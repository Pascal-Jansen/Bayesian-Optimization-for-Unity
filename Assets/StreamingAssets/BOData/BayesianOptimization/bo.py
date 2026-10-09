# bo.py — single-objective Bayesian Optimization (NDJSON protocol)
# Uses qLogNoisyExpectedImprovement.
# Logs observations to ObservationsPerEvaluation.csv ('IsBest' flag) and
# best-so-far metric to BestObjectivePerEvaluation.csv.
# Also mirrors the metric to legacy HypervolumePerEvaluation.csv for compatibility.

import socket
import time
import os
import numpy as np
import torch

from botorch.acquisition.logei import qLogNoisyExpectedImprovement  # LogNEI
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
# shared with mobo.py.
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
NUM_OBJS = None  # must be 1

# derived at init
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
objectives_info = []   # [(lo, hi, minimizeFlag)]  # minimizeFlag==1 means minimize in original scale

# contextual optimization (LCE-M GP); None => plain SingleTaskGP behaviour
CONTEXT_SETUP = None

# device
tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
device = torch.device("cpu")

# -------------------- threading --------------------
# Single-objective GPs at study sizes are too small for intra-op parallelism: one torch
# thread was 1.4-1.6x faster per suggestion with identical candidates (d=2..8, n=12..80),
# and leaves the remaining cores to Unity. BO_TORCH_THREADS overrides it.
TORCH_THREADS = int(os.environ.get("BO_TORCH_THREADS", "1"))

# Frame transforms live in bo_normalize so that this backend, mobo.py and the offline
# Meta-TAF source generators cannot drift apart. Thin wrappers keep the call sites
# (and the test suite) unchanged.
def denormalize_to_original_param(val01, lo, hi, decimals="log"):
    return bo_normalize.denormalize_to_original_param(val01, lo, hi, decimals)

def denormalize_to_original_obj(v_m1p1, lo, hi, smaller_is_better):
    return bo_normalize.denormalize_to_original_obj(v_m1p1, lo, hi, smaller_is_better)


def normalize_param_column(col, lo, hi):
    return bo_normalize.normalize_param_column(col, lo, hi)


def normalize_obj_column(col, lo, hi, minflag):
    # The warm-start format stays a module global here (it is set once from the Unity
    # init message); bo_normalize takes it explicitly so offline generators can request
    # "raw" without mutating shared state.
    return bo_normalize.normalize_obj_column(
        col, lo, hi, minflag, fmt=WARM_START_OBJECTIVE_FORMAT
    )


def fixed_observation_columns():
    return botorch_common.fixed_observation_columns(CONTEXT_SETUP, 'IsBest')


def expected_observation_columns():
    return fixed_observation_columns() + objective_names + parameter_names


def observation_context_cells():
    """Extra CSV cells for the context column (empty when contexts are disabled)."""
    return botorch_common.observation_context_cells(CONTEXT_SETUP)


def current_context_mask(x_sample):
    """Boolean numpy row mask of observations that belong to the current context."""
    return botorch_common.current_context_mask(x_sample, CONTEXT_SETUP)

# -------------------- objective evaluation --------------------
def objective_function(conn, x_tensor, iteration):
    """Send one design to Unity and block for its objective (normalized, maximized).

    ``iteration`` is the Iteration the design is logged under in ObservationsPerEvaluation.csv;
    Unity receives it with the parameters.
    """
    x = x_tensor.cpu().numpy()
    values = {}
    for i, name in enumerate(parameter_names):
        lo, hi = parameters_info[i]
        values[name] = denormalize_to_original_param(x[i], lo, hi, decimals=None)
    payload = {"type": "parameters", "values": values, "iteration": int(iteration)}
    print("Send parameters:", payload, flush=True)
    send_json_line(conn, payload)

    resp = recv_objectives_blocking(conn)  # a dict, or None once Unity disconnected
    if resp is None:
        raise RuntimeError("No objectives received from Unity.")

    name = objective_names[0]
    missing = [k for k in objective_names if k not in resp]
    if missing:
        raise KeyError(f"Unity objectives missing required key(s): {missing}")
    unexpected = sorted([k for k in resp.keys() if k not in set(objective_names)])
    if unexpected:
        raise KeyError(f"Unity objectives payload contains unexpected key(s): {unexpected}")
    # normalize to [-1,1] and maximize: the transform every backend applies to Unity's values
    lo, hi, minflag = objectives_info[0]
    f = bo_normalize.normalize_objective_value(resp[name], lo, hi, minflag, name=name)
    return torch.tensor([f], dtype=torch.double)

# -------------------- data IO --------------------
def generate_initial_data(conn, n_samples, metric_values=None):
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1 for non-warm-start runs.")

    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    if not log_exists(obs_csv):
        # NOTE: 'IsBest' replaces 'IsPareto'
        write_csv_rows(obs_csv, expected_observation_columns(), [])

    # n_samples points of a d-dimensional Sobol sequence (n=N, q=1). The former n=1, q=N
    # drew ONE point of an N*d-dimensional sequence, which spreads no better than
    # independent uniform draws.
    train_x = draw_sobol_samples(bounds=problem_bounds, n=n_samples, q=1, seed=SEED).squeeze(1)
    print("Initial Sobol X in [0,1]:", train_x, flush=True)

    train_obj = []
    best_so_far = -1e9
    try:
        for i, x in enumerate(train_x):
            print(f"---- Initial Sample {i+1}", flush=True)
            y = objective_function(conn, x, i + 1)  # shape [1]
            train_obj.append(y)

            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()  # normalized [-1,1]
            # denormalize objective back to original scale for logging
            y_den = denormalize_to_original_obj(y_np[0], objectives_info[0][0], objectives_info[0][1], objectives_info[0][2])
            x_den = [denormalize_to_original_param(x_np[j], parameters_info[j][0], parameters_info[j][1]) for j in range(PROBLEM_DIM)]

            # Provisional IsBest flag (best-so-far); finalized once sampling ends.
            is_best = float(y_np[0]) > best_so_far + 1e-12
            if is_best:
                best_so_far = float(y_np[0])

            row = [USER_ID, CONDITION_ID, GROUP_ID,
                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                   i+1, 'sampling', *observation_context_cells(),
                   'TRUE' if is_best else 'FALSE', y_den, *x_den]
            append_csv_rows(obs_csv, [row])

            if metric_values is not None:
                metric_values.append(best_so_far)
                save_metric_to_file(metric_values, i + 1)

            send_json_line(conn, {"type": "tempCoverage", "value": float(i+1)/float(max(1,n_samples))})
    except StopRequested:
        # Unity ended the study during the sampling phase (perfect-rating stop): the rows logged
        # so far must not keep their provisional flags.
        finalize_sampling_flags(obs_csv, train_obj)
        raise

    # Sampling-only runs (N_ITERATIONS=0) end here, so the flags must be final now.
    finalize_sampling_flags(obs_csv, train_obj)

    Y = torch.tensor(np.stack([t.numpy() for t in train_obj], axis=0), dtype=torch.double)  # shape [n,1]
    if CONTEXT_SETUP is not None:
        # All freshly sampled observations belong to the current context.
        train_x = context_support.append_task_column(train_x, CONTEXT_SETUP.current_index)
    return train_x, Y

def finalize_sampling_flags(obs_csv, train_obj):
    """Replace the provisional (best-so-far) IsBest flags of the sampling rows by the final ones."""
    if not train_obj:
        return
    vals_norm = [float(t.item()) for t in train_obj]
    best_norm = max(vals_norm)
    flags = ['TRUE' if abs(v - best_norm) < 1e-12 else 'FALSE' for v in vals_norm]
    df = read_observation_log(obs_csv)
    if len(df) >= len(flags):
        df.loc[df.index[:len(flags)], 'IsBest'] = flags
        write_dataframe_csv(obs_csv, df)

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

# -------------------- acquisition (single-objective, LogNEI) --------------------
def optimize_candidates(model, sampler, X_baseline):
    if X_baseline.dim() == 3:
        X_baseline = X_baseline[0]
    acq = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=X_baseline,
        sampler=sampler,
        # tau=1e-3,  # optional smoothing
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
    return candidates.detach()  # in [0,1]

def acquisition_baseline(train_x):
    """Baseline design points for the acquisition function (task column stripped)."""
    return botorch_common.acquisition_baseline(train_x, CONTEXT_SETUP)

def best_current_context_objective(train_x, train_y):
    """Best normalized objective among current-context observations.

    Returns -1.0 (the normalized minimum) when the current context has no
    observations yet (e.g. a warm start that only contains other contexts).
    """
    mask = current_context_mask(train_x)
    y_flat = np.asarray(train_y.detach().cpu().numpy()).reshape(-1)
    vals = [float(v) for v, keep in zip(y_flat.tolist(), mask.tolist()) if keep]
    if not vals:
        print(
            "Warning: no current-context observations yet; reporting metric -1.0.",
            flush=True,
        )
        return -1.0
    return max(vals)

# -------------------- logging --------------------
def save_xy(x_sample, y_sample, iteration):
    """Log the newest evaluation to ObservationsPerEvaluation.csv; returns its Iteration."""
    def is_best_flags(y, ctx_mask):
        # Only current-context observations compete for IsBest; warm-start rows from
        # other contexts inform the model but are not part of this run's metric.
        vals_norm = [
            float(v)
            for v, keep in zip(np.asarray(y.detach().cpu().numpy()).reshape(-1).tolist(), ctx_mask.tolist())
            if keep
        ]
        best_norm = max(vals_norm) if len(vals_norm) > 0 else float(y[-1].item())
        return ['TRUE' if abs(v - best_norm) < 1e-12 else 'FALSE' for v in vals_norm]

    return botorch_common.append_optimization_observation(
        os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv"), x_sample, y_sample,
        context_setup=CONTEXT_SETUP, parameters_info=parameters_info, objectives_info=objectives_info,
        id_cells=[USER_ID, CONDITION_ID, GROUP_ID], expected_columns=expected_observation_columns(),
        flag_column='IsBest', current_context_flags=is_best_flags,
    )

def metric_scale():
    """`Scale` column of the metric logs: the frame of BestObjective and the objective's direction."""
    if int(objectives_info[0][2]) == 1:
        return "normalized maximize-space [-1,1] (objective minimized: sign-flipped)"
    return "normalized maximize-space [-1,1] (objective maximized)"

def save_metric_to_file(metric_values, iteration, observed=True):
    """Append the best-so-far metric of one evaluation to the metric logs.

    BestObjective is the normalized [-1,1] maximize-space value Unity's coverage reports (kept
    for existing analysis scripts); BestObjectiveRaw is the same best observation in the
    objective's own units and direction, like every other log column. ``observed=False`` (no
    current-context observation yet) leaves BestObjectiveRaw empty.
    """
    best = metric_values[-1]
    lo, hi, minflag = objectives_info[0]
    raw = denormalize_to_original_obj(best, lo, hi, minflag) if observed else ""
    scale = metric_scale()
    append_csv_rows(os.path.join(PROJECT_PATH, "BestObjectivePerEvaluation.csv"),
                    [[best, iteration, scale, raw]],
                    header=["BestObjective", "Iteration", "Scale", "BestObjectiveRaw"])
    # Legacy mirror for older analysis scripts that still read this file.
    append_csv_rows(os.path.join(PROJECT_PATH, "HypervolumePerEvaluation.csv"),
                    [[best, iteration, scale, raw]],
                    header=["Hypervolume", "Iteration", "Scale", "BestObjectiveRaw"])

# -------------------- main loop --------------------
def bo_execute(conn, seed, iterations, initial_samples):
    global PROJECT_PATH, OBSERVATIONS_LOG_PATH
    base = os.environ.get("BO_LOG_ROOT") or os.path.join(os.getcwd(), "LogData")
    condition_base = os.path.join(base, USER_LOG_ID, CONDITION_LOG_ID)
    os.makedirs(condition_base, exist_ok=True)
    PROJECT_PATH = get_unique_folder(condition_base, "run")
    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")

    exec_csv = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(exec_csv, ['Optimization', 'Execution_Time'])

    torch.manual_seed(seed)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]), seed=SEED)

    metric_values = []  # best normalized objective per evaluation (current context only)

    if WARM_START:
        train_x, train_y = load_data()
    else:
        train_x, train_y = generate_initial_data(
            conn, n_samples=initial_samples, metric_values=metric_values
        )

    mll, model = initialize_model(train_x, train_y)

    best = best_current_context_objective(train_x, train_y)
    if WARM_START:
        # Baseline over the warm-start data, at the Iteration of the last current-context
        # warm-start row: the axis ObservationsPerEvaluation.csv uses (0 if none).
        n_current = int(np.sum(current_context_mask(train_x)))
        metric_values.append(best)
        save_metric_to_file(metric_values, n_current, observed=n_current > 0)
    send_json_line(conn, {"type": "coverage", "value": float(best)})

    for it in range(1, iterations + 1):
        t0 = time.time()
        fit_gpytorch_mll(mll)
        new_x = optimize_candidates(model, sampler, X_baseline=acquisition_baseline(train_x))
        t_elapsed = time.time() - t0
        write_data_to_csv(exec_csv, ['Optimization', 'Execution_Time'],
                          [{'Optimization': it, 'Execution_Time': t_elapsed}])

        # The Iteration save_xy will log this design under: this run's rows so far + 1.
        iteration = int(np.sum(current_context_mask(train_x))) + 1
        new_y = objective_function(conn, new_x[0], iteration)  # shape [1]
        if CONTEXT_SETUP is not None:
            new_x = context_support.append_task_column(new_x, CONTEXT_SETUP.current_index)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y.unsqueeze(0)])  # shape [n+1,1]

        best = best_current_context_objective(train_x, train_y)
        metric_values.append(best)
        # The metric row carries the same Iteration as the evaluation's observation row.
        save_metric_to_file(metric_values, save_xy(train_x, train_y, it))
        send_json_line(conn, {"type": "coverage", "value": float(best)})

        mll, model = initialize_model(train_x, train_y)

    send_json_line(conn, {"type": "optimization_finished"})
    return metric_values, train_x, train_y

# -------------------- boot --------------------
def main():
    global N_INITIAL, N_ITERATIONS, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, SEED
    global PROBLEM_DIM, NUM_OBJS, problem_bounds
    global WARM_START, CSV_PATH_PARAMETERS, CSV_PATH_OBJECTIVES, WARM_START_OBJECTIVE_FORMAT
    global USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID
    global parameter_names, objective_names, parameters_info, objectives_info
    global CONTEXT_SETUP

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    configure_listener_socket(s)
    conn = None
    try:
        if TORCH_THREADS < 1:
            raise ValueError(f"BO_TORCH_THREADS must be >= 1, got {TORCH_THREADS}")
        torch.set_num_threads(TORCH_THREADS)
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
        WARM_START_OBJECTIVE_FORMAT = str(
            cfg.get("warmStartObjectiveFormat", WARM_START_OBJECTIVE_FORMAT) or "auto"
        ).strip().lower()

        if PROBLEM_DIM < 1:
            raise ValueError(f"nParameters must be >= 1, got {PROBLEM_DIM}")
        if NUM_OBJS != 1:
            raise ValueError(f"bo.py expects exactly 1 objective, got {NUM_OBJS}")
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
        if WARM_START_OBJECTIVE_FORMAT not in ("auto", "raw", "normalized_max", "normalized_native"):
            raise ValueError(
                "warmStartObjectiveFormat must be one of: auto, raw, normalized_max, normalized_native; "
                f"got '{WARM_START_OBJECTIVE_FORMAT}'"
            )
        if BATCH_SIZE != 1:
            print(f"Warning: batchSize={BATCH_SIZE} is not supported in this HITL loop; forcing batchSize=1.", flush=True)
            BATCH_SIZE = 1

        USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID = parse_user_ids(init_msg)

        parameters = init_msg.get("parameters", []) or []
        objectives = init_msg.get("objectives", []) or []

        parameter_names = [p.get("key") for p in parameters]
        objective_names = [o.get("key") for o in objectives]
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("Duplicate parameter keys in init message.")
        if len(set(objective_names)) != len(objective_names):
            raise ValueError("Duplicate objective keys in init message.")
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

        # normalized search box [0,1]^d
        problem_bounds = torch.stack(
            [torch.zeros(PROBLEM_DIM, dtype=torch.double),
             torch.ones (PROBLEM_DIM, dtype=torch.double)],
            dim=0
        )

        print("Init OK:", dict(
            BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES,
            N_ITERATIONS=N_ITERATIONS, MC_SAMPLES=MC_SAMPLES,
            N_INITIAL=N_INITIAL, SEED=SEED, PROBLEM_DIM=PROBLEM_DIM, NUM_OBJS=NUM_OBJS
        ), flush=True)

        bo_execute(conn, SEED, N_ITERATIONS, N_INITIAL)
    except StopRequested:
        pass  # Unity ended the study on purpose; every completed evaluation is logged
    finally:
        close_connection(conn)
        s.close()
        flush_pending_logs()

if __name__ == "__main__":
    main()
