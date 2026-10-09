import socket
import time
import os
import numpy as np
import torch
import moocore

from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples

import sys

# Sibling module imports must work both when running this file as a script and
# when loading it from another working directory (e.g. the test suite).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import context_support
import bo_normalize
# Warm-start loading, model construction, context bookkeeping and the observation-log append,
# shared with bo.py.
import botorch_common
# Socket, NDJSON, init parsing and log-writing plumbing shared by every backend.
from bo_protocol import (
    SOCKET_ACCEPT_TIMEOUT_SEC, SOCKET_TIMEOUT_SEC, StopRequested,
    accept_unity_connection, append_csv_rows, close_connection, configure_listener_socket,
    create_csv_file, flush_pending_logs, get_cfg_int, get_unique_folder, log_exists,
    parse_obj_init, parse_param_init, parse_user_ids, read_observation_log,
    receive_init_message, recv_objectives_blocking, send_json_line,
    validate_objective_bounds, validate_parameter_bounds, validate_raw_samples,
    write_csv_rows, write_data_to_csv, write_dataframe_csv,
)

# -------------------- defaults (overwritten by Unity init) --------------------
N_INITIAL = 5
N_ITERATIONS = 10
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 1024
# Matches Unity's default (was 512); used only when the init message omits mcSamples.
MC_SAMPLES = 128
SEED = 3

PROBLEM_DIM = None
NUM_OBJS = None

# derived at init
ref_point = None
problem_bounds = None

# paths/state
PROJECT_PATH = ""
OBSERVATIONS_LOG_PATH = ""

# warm start placeholders
WARM_START = False
CSV_PATH_PARAMETERS = ""
CSV_PATH_OBJECTIVES = ""
WARM_START_OBJECTIVE_FORMAT = "auto"  # auto|raw|normalized_max|normalized_native

# study info
USER_ID = ""
CONDITION_ID = ""
GROUP_ID = ""
USER_LOG_ID = ""
CONDITION_LOG_ID = ""

# names and meta parsed from init
parameter_names = []
objective_names = []
parameters_info = []   # [(lo, hi)]
objectives_info = []   # [(lo, hi, minimizeFlag)]

# contextual optimization (LCE-M GP); None => plain SingleTaskGP behaviour
CONTEXT_SETUP = None

# device
tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
device = torch.device("cpu")

# -------------------- Pareto / hypervolume helpers --------------------
def as_numpy_array(values):
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=np.float64)

def as_objective_matrix(values):
    arr = as_numpy_array(values)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Objective values must be a 2D array, got shape {arr.shape}")
    return arr

def bool_mask_like_input(mask, source):
    mask = np.asarray(mask, dtype=bool)
    # Decide by type: NumPy 2 arrays also have a .device, and a one-element torch bool
    # mask indexing a numpy array is read as the integer index 1.
    if isinstance(source, torch.Tensor) and hasattr(torch, "as_tensor") and hasattr(torch, "bool"):
        return torch.as_tensor(mask, dtype=torch.bool, device=source.device)
    return mask

def first_duplicate_mask(values):
    keep = np.zeros(values.shape[0], dtype=bool)
    seen = set()
    for i, row in enumerate(values):
        key = tuple(row.tolist())
        if key in seen:
            continue
        seen.add(key)
        keep[i] = True
    return keep

def is_non_dominated(values):
    arr = as_objective_matrix(values)
    finite_rows = np.all(np.isfinite(arr), axis=1)
    mask = np.zeros(arr.shape[0], dtype=bool)
    if np.any(finite_rows):
        finite_arr = arr[finite_rows]
        finite_mask = moocore.is_nondominated(
            finite_arr,
            maximise=True,
            keep_weakly=True,
        )
        finite_mask &= first_duplicate_mask(finite_arr)
        mask[finite_rows] = finite_mask
    return bool_mask_like_input(mask, values)

class Hypervolume:
    def __init__(self, ref_point):
        self.ref_point = as_numpy_array(ref_point)

    def compute(self, values):
        arr = as_objective_matrix(values)
        finite_rows = np.all(np.isfinite(arr), axis=1)
        if not np.any(finite_rows):
            return 0.0
        return float(moocore.hypervolume(arr[finite_rows], ref=self.ref_point, maximise=True))

# Frame transforms live in bo_normalize so that this backend, bo.py and the offline
# Meta-TAF source generators cannot drift apart. Thin wrappers keep the call sites
# (and the test suite) unchanged.
def denormalize_to_original_param(val01, lo, hi, decimals="log"):
    return bo_normalize.denormalize_to_original_param(val01, lo, hi, decimals)

def denormalize_to_original_obj(v_m1p1, lo, hi, smaller_is_better):
    return bo_normalize.denormalize_to_original_obj(v_m1p1, lo, hi, smaller_is_better)

def fixed_observation_columns():
    return botorch_common.fixed_observation_columns(CONTEXT_SETUP, 'IsPareto')


def expected_observation_columns():
    return fixed_observation_columns() + objective_names + parameter_names


def observation_context_cells():
    """Extra CSV cells for the context column (empty when contexts are disabled)."""
    return botorch_common.observation_context_cells(CONTEXT_SETUP)


def current_context_mask(x_sample):
    """Boolean numpy row mask of observations that belong to the current context."""
    return botorch_common.current_context_mask(x_sample, CONTEXT_SETUP)


def current_context_hypervolume(hv_util, train_x, train_y):
    """Hypervolume of the current context's non-dominated observations.

    Warm-start rows from other contexts inform the model but do not contribute
    to this run's Pareto front / hypervolume metric.
    """
    mask = current_context_mask(train_x)
    y_np = as_objective_matrix(train_y)[mask]
    if y_np.shape[0] == 0:
        print(
            "Warning: no current-context observations yet; reporting hypervolume 0.0.",
            flush=True,
        )
        return 0.0
    pareto_mask = is_non_dominated(y_np)
    return hv_util.compute(y_np[pareto_mask])

def normalize_param_column(col, lo, hi):
    return bo_normalize.normalize_param_column(col, lo, hi)

def normalize_obj_column(col, lo, hi, minflag):
    # The warm-start format stays a module global here (it is set once from the Unity
    # init message); bo_normalize takes it explicitly so offline generators can request
    # "raw" without mutating shared state.
    return bo_normalize.normalize_obj_column(
        col, lo, hi, minflag, fmt=WARM_START_OBJECTIVE_FORMAT
    )

# -------------------- objective evaluation --------------------
def objective_function(conn, x_tensor, iteration):
    """Send one design to Unity and block for its objectives (normalized, maximized).

    ``iteration`` is the Iteration the design is logged under in ObservationsPerEvaluation.csv;
    Unity receives it with the parameters.
    """
    x = x_tensor.cpu().numpy()
    values = {}
    for i, name in enumerate(parameter_names):
        lo, hi = parameters_info[i]
        # Keep full precision for optimizer-proposed points sent to Unity.
        values[name] = denormalize_to_original_param(x[i], lo, hi, decimals=None)

    payload = {"type": "parameters", "values": values, "iteration": int(iteration)}
    print("Send parameters:", payload, flush=True)
    send_json_line(conn, payload)

    resp = recv_objectives_blocking(conn)  # a dict, or None once Unity disconnected
    if resp is None:
        raise RuntimeError("No objectives received from Unity.")

    fs = []
    missing = [name for name in objective_names if name not in resp]
    if missing:
        raise KeyError(f"Unity objectives missing required key(s): {missing}")
    unexpected = sorted([k for k in resp.keys() if k not in set(objective_names)])
    if unexpected:
        raise KeyError(f"Unity objectives payload contains unexpected key(s): {unexpected}")

    # normalize to [-1,1] and maximize: the transform every backend applies to Unity's values
    for i, name in enumerate(objective_names):
        lo, hi, minflag = objectives_info[i]
        fs.append(bo_normalize.normalize_objective_value(resp[name], lo, hi, minflag, name=name))

    return torch.tensor(fs, dtype=torch.double)

# -------------------- data IO --------------------
def generate_initial_data(conn, n_samples, hv_util=None, hvs=None):
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1 for non-warm-start runs.")

    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    if not log_exists(obs_csv):
        write_csv_rows(obs_csv, expected_observation_columns(), [])

    # n_samples points of a d-dimensional Sobol sequence (n=N, q=1). The former n=1, q=N
    # drew ONE point of an N*d-dimensional sequence, which spreads no better than
    # independent uniform draws.
    train_x = draw_sobol_samples(bounds=problem_bounds, n=n_samples, q=1, seed=SEED).squeeze(1)
    print("Initial Sobol X in [0,1]:", train_x, flush=True)

    train_obj = []
    try:
        for i, x in enumerate(train_x):
            print(f"---- Initial Sample {i+1}", flush=True)
            y = objective_function(conn, x, i + 1)
            train_obj.append(y)

            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()
            x_den = [denormalize_to_original_param(x_np[j], parameters_info[j][0], parameters_info[j][1]) for j in range(PROBLEM_DIM)]
            y_den = [denormalize_to_original_obj(y_np[j], objectives_info[j][0], objectives_info[j][1], objectives_info[j][2]) for j in range(NUM_OBJS)]
            # IsPareto is provisional (FALSE) until sampling ends.
            row = [USER_ID, CONDITION_ID, GROUP_ID,
                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                   i+1, 'sampling', *observation_context_cells(), 'FALSE', *y_den, *x_den]
            append_csv_rows(obs_csv, [row])
            send_json_line(conn, {"type": "tempCoverage", "value": float(i+1)/float(max(1,n_samples))})
            if hv_util is not None and hvs is not None:
                y_so_far = torch.stack(train_obj, dim=0).to(dtype=torch.double)
                volume = hv_util.compute(y_so_far[is_non_dominated(y_so_far)])
                hvs.append(volume)
                save_hypervolume_to_file(hvs, i + 1)
                send_json_line(conn, {"type": "coverage", "value": float(volume)})
    except StopRequested:
        # Unity ended the study during the sampling phase (perfect-rating stop): the rows logged
        # so far must not keep their provisional flags.
        finalize_sampling_flags(obs_csv, train_obj)
        raise

    # Sampling-only runs (N_ITERATIONS=0) end here, so the flags must be final now.
    Y = finalize_sampling_flags(obs_csv, train_obj)

    if CONTEXT_SETUP is not None:
        # All freshly sampled observations belong to the current context.
        train_x = context_support.append_task_column(train_x, CONTEXT_SETUP.current_index)
    return train_x, Y

def finalize_sampling_flags(obs_csv, train_obj):
    """Write the final IsPareto flags of the sampling rows; returns their objectives (n x M)."""
    if not train_obj:
        return None
    Y = torch.stack(train_obj, dim=0).to(dtype=torch.double)
    pareto_flags = ['TRUE' if b else 'FALSE' for b in is_non_dominated(Y).tolist()]
    df = read_observation_log(obs_csv)
    if len(df) >= len(pareto_flags):
        df.loc[df.index[:len(pareto_flags)], 'IsPareto'] = pareto_flags
        write_dataframe_csv(obs_csv, df)
    return Y

def load_data():
    """Warm-start CSVs -> (train_x, train_y) in the canonical frame (see botorch_common)."""
    return botorch_common.load_warm_start_data(
        CSV_PATH_PARAMETERS, CSV_PATH_OBJECTIVES, parameter_names, parameters_info,
        objective_names, objectives_info, CONTEXT_SETUP,
        normalize_param_column=normalize_param_column, normalize_obj_column=normalize_obj_column,
    )

# -------------------- model --------------------
def initialize_model(train_x, train_obj):
    return botorch_common.initialize_model(train_x, train_obj, CONTEXT_SETUP)

def acquisition_baseline(train_x):
    """Baseline design points for the acquisition function (task column stripped)."""
    return botorch_common.acquisition_baseline(train_x, CONTEXT_SETUP)

# -------------------- acquisition --------------------
def reference_point(num_objs):
    """Hypervolume reference point, shared by qLogNEHVI, the coverage metric and the log.

    It lies beyond the worst normalized value (-1): with exactly [-1]^M a Pareto-optimal
    design rated worst on one objective added no hypervolume, so it counted for nothing in
    the coverage metric and in the acquisition function.
    """
    return torch.full((num_objs,), bo_normalize.HYPERVOLUME_REFERENCE_VALUE, dtype=torch.double)

def optimize_qnehvi(model, sampler, X_baseline):
    if X_baseline.dim() == 3:
        X_baseline = X_baseline[0]
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=X_baseline,
        sampler=sampler,
        prune_baseline=True,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=problem_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        # init_batch_limit scores the raw samples 128 at a time instead of batch_limit's 5:
        # same candidates, a fraction of the wait. batch_limit stays 5: polishing all restarts
        # in one batch saved < 0.1 s but ended at a different local optimum in ~1% of problems.
        options={"batch_limit": 5, "init_batch_limit": 128, "maxiter": 200},
        sequential=True,
    )
    return candidates.detach()

# -------------------- logging --------------------
def save_xy(x_sample, y_sample, iteration):
    """Log the newest evaluation to ObservationsPerEvaluation.csv; returns its Iteration."""
    def is_pareto_flags(y, ctx_mask):
        # Only current-context observations compete for the Pareto front; warm-start
        # rows from other contexts inform the model but are not part of this run.
        y_current = as_objective_matrix(y)[ctx_mask]
        return ['TRUE' if b else 'FALSE' for b in is_non_dominated(y_current).tolist()]

    return botorch_common.append_optimization_observation(
        os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv"), x_sample, y_sample,
        context_setup=CONTEXT_SETUP, parameters_info=parameters_info, objectives_info=objectives_info,
        id_cells=[USER_ID, CONDITION_ID, GROUP_ID], expected_columns=expected_observation_columns(),
        flag_column='IsPareto', current_context_flags=is_pareto_flags,
    )

def save_hypervolume_to_file(hvs, iteration):
    append_csv_rows(
        os.path.join(PROJECT_PATH, "HypervolumePerEvaluation.csv"),
        [[
            hvs[-1],
            iteration,
            "normalized maximize-space [-1,1] per objective",
            "[" + ",".join(str(float(value)) for value in as_numpy_array(ref_point)) + "]",
        ]],
        header=["Hypervolume", "Iteration", "Scale", "ReferencePoint"],
    )

# -------------------- main loop --------------------
def mobo_execute(conn, seed, iterations, initial_samples):
    global PROJECT_PATH, OBSERVATIONS_LOG_PATH
    base = os.environ.get("BO_LOG_ROOT") or os.path.join(os.getcwd(), "LogData")
    condition_base = os.path.join(base, USER_LOG_ID, CONDITION_LOG_ID)
    os.makedirs(condition_base, exist_ok=True)
    PROJECT_PATH = get_unique_folder(condition_base, "run")
    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")

    exec_csv = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(exec_csv, ['Optimization', 'Execution_Time'])

    torch.manual_seed(seed)
    hv_util = Hypervolume(ref_point=ref_point)
    hvs = []

    if WARM_START:
        train_x, train_y = load_data()
    else:
        train_x, train_y = generate_initial_data(
            conn, n_samples=initial_samples, hv_util=hv_util, hvs=hvs
        )

    expected_x_dim = PROBLEM_DIM + (1 if CONTEXT_SETUP is not None else 0)
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError(f"Training X/Y row mismatch: X={train_x.shape[0]}, Y={train_y.shape[0]}")
    if train_x.shape[1] != expected_x_dim:
        raise ValueError(f"Training X has wrong dimension: got {train_x.shape[1]}, expected {expected_x_dim}")
    if train_y.shape[1] != NUM_OBJS:
        raise ValueError(f"Training Y has wrong objective count: got {train_y.shape[1]}, expected {NUM_OBJS}")

    mll, model = initialize_model(train_x, train_y)

    if WARM_START:
        volume = current_context_hypervolume(hv_util, train_x, train_y)
        hvs.append(volume)
        # Baseline at the Iteration of the last current-context warm-start row: the axis
        # ObservationsPerEvaluation.csv uses (0 if none belongs to the current context).
        save_hypervolume_to_file(hvs, int(np.sum(current_context_mask(train_x))))
        send_json_line(conn, {"type": "coverage", "value": float(volume)})

    for it in range(1, iterations + 1):
        t0 = time.time()
        fit_gpytorch_mll(mll)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]), seed=SEED)
        new_x = optimize_qnehvi(model, sampler, X_baseline=acquisition_baseline(train_x))
        t_elapsed = time.time() - t0
        write_data_to_csv(exec_csv, ['Optimization', 'Execution_Time'],
                          [{'Optimization': it, 'Execution_Time': t_elapsed}])

        # The Iteration save_xy will log this design under: this run's rows so far + 1.
        iteration = int(np.sum(current_context_mask(train_x))) + 1
        new_y = objective_function(conn, new_x[0], iteration)
        if CONTEXT_SETUP is not None:
            new_x = context_support.append_task_column(new_x, CONTEXT_SETUP.current_index)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y.unsqueeze(0)])

        volume = current_context_hypervolume(hv_util, train_x, train_y)
        hvs.append(volume)
        # The metric row carries the same Iteration as the evaluation's observation row.
        save_hypervolume_to_file(hvs, save_xy(train_x, train_y, it))
        send_json_line(conn, {"type": "coverage", "value": float(volume)})
        mll, model = initialize_model(train_x, train_y)

    send_json_line(conn, {"type": "optimization_finished"})
    return hvs, train_x, train_y

# -------------------- boot --------------------
def main():
    global N_INITIAL, N_ITERATIONS, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, SEED
    global PROBLEM_DIM, NUM_OBJS, ref_point, problem_bounds
    global WARM_START, CSV_PATH_PARAMETERS, CSV_PATH_OBJECTIVES, WARM_START_OBJECTIVE_FORMAT
    global USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID
    global parameter_names, objective_names, parameters_info, objectives_info
    global CONTEXT_SETUP

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    configure_listener_socket(s)
    conn = None
    try:
        conn = accept_unity_connection(s, SOCKET_ACCEPT_TIMEOUT_SEC)
        init_msg = receive_init_message(conn, SOCKET_TIMEOUT_SEC)

        cfg = init_msg.get("config", {}) or {}
        N_INITIAL      = get_cfg_int(cfg, "numSamplingIterations", default=N_INITIAL)
        N_ITERATIONS   = get_cfg_int(cfg, "numOptimizationIterations", default=N_ITERATIONS)
        BATCH_SIZE     = get_cfg_int(cfg, "batchSize", default=BATCH_SIZE)
        NUM_RESTARTS   = get_cfg_int(cfg, "numRestarts", default=NUM_RESTARTS)
        RAW_SAMPLES    = get_cfg_int(cfg, "rawSamples", default=RAW_SAMPLES)
        MC_SAMPLES     = get_cfg_int(cfg, "mcSamples", default=MC_SAMPLES)
        SEED           = get_cfg_int(cfg, "seed", default=SEED)
        PROBLEM_DIM    = get_cfg_int(cfg, "nParameters", required=True)
        NUM_OBJS       = get_cfg_int(cfg, "nObjectives", required=True)
        WARM_START     = bool(cfg.get("warmStart", False))

        CSV_PATH_PARAMETERS = str(cfg.get("initialParametersDataPath") or "")
        CSV_PATH_OBJECTIVES = str(cfg.get("initialObjectivesDataPath") or "")
        WARM_START_OBJECTIVE_FORMAT = str(cfg.get("warmStartObjectiveFormat", WARM_START_OBJECTIVE_FORMAT) or "auto").strip().lower()

        if WARM_START_OBJECTIVE_FORMAT not in ("auto", "raw", "normalized_max", "normalized_native"):
            raise ValueError(
                "warmStartObjectiveFormat must be one of: auto, raw, normalized_max, normalized_native; "
                f"got '{WARM_START_OBJECTIVE_FORMAT}'"
            )

        if PROBLEM_DIM < 1:
            raise ValueError(f"nParameters must be >= 1, got {PROBLEM_DIM}")
        if NUM_OBJS < 2:
            raise ValueError(f"mobo.py expects at least 2 objectives, got {NUM_OBJS}")
        if N_INITIAL < 0 or N_ITERATIONS < 0:
            raise ValueError(f"Iteration counts must be non-negative, got sampling={N_INITIAL}, optimization={N_ITERATIONS}")
        if (not WARM_START) and N_INITIAL < 1:
            raise ValueError(
                "numSamplingIterations must be >= 1 when warmStart is disabled. "
                f"Got sampling={N_INITIAL}, warmStart={WARM_START}."
            )
        if NUM_RESTARTS < 1 or RAW_SAMPLES < 1 or MC_SAMPLES < 1:
            raise ValueError(
                f"numRestarts/rawSamples/mcSamples must be >=1, got {NUM_RESTARTS}/{RAW_SAMPLES}/{MC_SAMPLES}"
            )
        validate_raw_samples(NUM_RESTARTS, RAW_SAMPLES)
        if BATCH_SIZE != 1:
            print(f"Warning: batchSize={BATCH_SIZE} is not supported in this HITL loop; forcing batchSize=1.", flush=True)
            BATCH_SIZE = 1

        USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID = parse_user_ids(init_msg)

        parameters = init_msg.get("parameters", []) or []
        objectives = init_msg.get("objectives", []) or []

        parameter_names = [p.get("key") for p in parameters]
        objective_names = [o.get("key") for o in objectives]

        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("Duplicate parameter keys detected in init message.")
        if len(set(objective_names)) != len(objective_names):
            raise ValueError("Duplicate objective keys detected in init message.")
        overlap = sorted(set(parameter_names).intersection(set(objective_names)))
        if overlap:
            raise ValueError(f"Parameter and objective keys must be distinct. Overlap: {overlap}")

        if len(parameter_names) != PROBLEM_DIM:
            raise ValueError(f"parameter_names len {len(parameter_names)} != nParameters {PROBLEM_DIM}")
        if len(objective_names) != NUM_OBJS:
            raise ValueError(f"objective_names len {len(objective_names)} != nObjectives {NUM_OBJS}")

        parameters_info = [parse_param_init(p.get("init")) for p in parameters]
        objectives_info = [parse_obj_init(o.get("init")) for o in objectives]

        validate_parameter_bounds(parameter_names, parameters_info, "BoTorch")
        validate_objective_bounds(objective_names, objectives_info)

        CONTEXT_SETUP = context_support.parse_context_config(init_msg)
        if CONTEXT_SETUP is not None:
            init_root = os.environ.get("BO_INIT_ROOT") or os.path.join(os.getcwd(), "InitData")
            context_support.resolve_embeddings(CONTEXT_SETUP, init_root=init_root)
            print("Contextual optimization:", context_support.describe(CONTEXT_SETUP), flush=True)

        # Keys become CSV columns next to the fixed ones; a key such as "Phase" would
        # duplicate a column and break the log rewrite after the sampling phase.
        reserved = sorted(set(parameter_names + objective_names).intersection(fixed_observation_columns()))
        if reserved:
            raise ValueError(f"Parameter/objective keys collide with log columns: {reserved}. Rename them.")

        ref_point = reference_point(NUM_OBJS)
        problem_bounds = torch.stack(
            [torch.zeros(PROBLEM_DIM, dtype=torch.double),
             torch.ones(PROBLEM_DIM, dtype=torch.double)],
            dim=0
        )

        print("Init OK:", dict(
            BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES,
            N_ITERATIONS=N_ITERATIONS, MC_SAMPLES=MC_SAMPLES,
            N_INITIAL=N_INITIAL, SEED=SEED, PROBLEM_DIM=PROBLEM_DIM, NUM_OBJS=NUM_OBJS,
            SOCKET_TIMEOUT_SEC=SOCKET_TIMEOUT_SEC, WARM_START_OBJECTIVE_FORMAT=WARM_START_OBJECTIVE_FORMAT
        ), flush=True)

        mobo_execute(conn, SEED, N_ITERATIONS, N_INITIAL)
    except StopRequested:
        pass  # Unity ended the study on purpose; every completed evaluation is logged
    finally:
        close_connection(conn)
        s.close()
        flush_pending_logs()

if __name__ == "__main__":
    main()
