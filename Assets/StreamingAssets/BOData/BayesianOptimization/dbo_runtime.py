# dbo_runtime.py — single-objective Dynamic Bayesian Optimization (NDJSON protocol).
#
# Speaks the same wire protocol as bo.py (TCP server on port 56001, newline-delimited
# JSON: one init message, then a parameters -> objectives loop with tempCoverage /
# coverage / optimization_finished updates) and writes the same CSV family, but the
# optimizer is dbo_torch's DynamicBO: a GP whose covariance is the product of a spatial
# factor over the design parameters and a temporal factor alpha ** |t - t'| over an
# appended TIME column. alpha is fitted jointly with the other hyperparameters, so the
# model infers how fast the participant is drifting (fatigue, adaptation, learning)
# instead of assuming the objective is stationary.
#
# TIME IS THE ITERATION INDEX, AND IT IS CONTINUOUS ACROSS BOTH PHASES.
# BOforUnity runs a sampling phase (numSamplingIterations) followed by an optimization
# phase (numOptimizationIterations). Both phases are fed to ONE DynamicBO instance via
# suggest()/observe(), so the sampling points are iterations 1..N_INITIAL and the
# optimization points continue at N_INITIAL+1 .. N_INITIAL+N_ITERATIONS on the same
# axis. That matters: restarting the clock at the optimization phase would tell the
# model the sampling observations were taken at the same time as the optimization ones,
# which is precisely the drift the temporal kernel exists to model. The sampling phase
# is supplied through DBOConfig.seed_points (Sobol, same draw as bo.py), which is the
# package's own mechanism for "fixed inputs at the first iterations" and keeps the
# ask/tell loop uniform across phases.
#
# SIGN CONVENTION. BOforUnity's canonical frame (bo_normalize.py) is: parameters in
# [0,1]^d, objectives in [-1,1] as a MAXIMIZATION problem, with the per-objective
# 'minimize' flag from the init message already folded in via a sign flip. DynamicBO
# MINIMISES cost. The bridge is therefore cost = -f, applied at exactly two places
# (observe() and the diagnostics log); everything else — CSV denormalization, the
# coverage metric, IsBest — stays in the canonical maximization frame so the existing
# analysis tooling keeps working unchanged.
#
# Deliberate scope limits (validated, not silently ignored):
#   - single objective only (nObjectives == 1): DBO models one drifting cost,
#   - no contextual optimization (the LCE-M context pipeline stays BoTorch-only).
#
# dbo_torch is VENDORED into ./dbo_torch/ (kernels.py, model.py, optimizer.py,
# __init__.py, LICENSE) from https://github.com/M-Colley/dbo-torch so a study machine
# needs nothing beyond the BoTorch stack bo.py already requires. The upstream
# unity_bridge.py is deliberately NOT vendored: it implements a different, incompatible
# request/response protocol.

import csv
import json
import os
import socket
import sys
import time

import numpy as np
import pandas as pd
import torch

from botorch.utils.sampling import draw_sobol_samples

# Sibling module imports (and the vendored dbo_torch package) must work both when
# running this file as a script and when loading it from another working directory
# (e.g. the test suite).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import bo_normalize
from dbo_torch import DBOConfig, DBOModelConfig, DynamicBO

# -------------------- defaults (overwritten by Unity init) --------------------
N_INITIAL = 5
N_ITERATIONS = 10
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 1024
MC_SAMPLES = 512   # read for parity only: DBO uses analytic LogEI, not an MC sampler
SEED = 3

PROBLEM_DIM = None
NUM_OBJS = None  # must be 1

# DBO configuration (overwritten from the init message's dbo* fields).
DBO_SPATIAL_KERNEL = "rbf"              # rbf | matern52
DBO_ALPHA_PARAMETERIZATION = "decay"    # decay | direct
DBO_INITIAL_ALPHA = 0.99
DBO_ACQUISITION_TIME_OFFSET = 0.0       # 0 scores candidates now, 1 at their eval time
DBO_VALIDATION_EVERY = 0                # 0/negative disables validation iterations
DBO_VALIDATION_CONFIDENCE = 0.01        # tail mass, not confidence level
DBO_VALIDATION_VISITED_ONLY = True
DBO_STATIONARY_BASELINE = False         # True freezes alpha=1 -> plain stationary BO
EXPLORATION_RATIO = 0.1                 # 0 disables the over-exploitation guard

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

# run bookkeeping. ALL_NORM_MAX holds the canonical [-1,1] MAXIMIZATION value of every
# observation handed to the optimizer (warm-start rows included); OBSERVATION_ROWS holds
# only this run's live evaluations, which are the rows that reach the CSV.
ALL_NORM_MAX = []
OBSERVATION_ROWS = []  # [(iteration, phase, timestamp, y_denormalized, [x_denormalized])]

# Dtype/device are owned by DBOConfig (float64 on CPU), so there is no tkwargs here.

# -------------------- TCP server helpers --------------------
HOST = ''
PORT = 56001
SOCKET_TIMEOUT_SEC = float(os.environ.get("BO_SOCKET_TIMEOUT_SEC", "3600"))
SOCKET_ACCEPT_TIMEOUT_SEC = float(os.environ.get("BO_ACCEPT_TIMEOUT_SEC", "300"))
SOCKET_MAX_RECV_BUF_BYTES = int(os.environ.get("BO_MAX_RECV_BUF_BYTES", "1048576"))
SOCKET_RECV_BUF = ""


def normalize_user_token(value, default="-1"):
    token = str(value).strip() if value is not None else ""
    return token if token else default


def normalize_log_folder_token(value, default="-1"):
    token = normalize_user_token(value, default=default)
    invalid_chars = set('/\\:*?"<>|')
    cleaned_chars = []
    for ch in token:
        if ch in invalid_chars or ord(ch) < 32:
            cleaned_chars.append("_")
        else:
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars).strip().strip(".")
    if cleaned in ("", ".", ".."):
        return default
    return cleaned


def send_json_line(conn, obj):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    try:
        conn.sendall(line.encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        t = obj.get("type") if isinstance(obj, dict) else "unknown"
        raise ConnectionError(f"Failed to send message to Unity (type={t}): {e}") from e


def recv_json_message(conn):
    """Receive one NDJSON message while preserving unread bytes across calls."""
    global SOCKET_RECV_BUF
    while True:
        idx = SOCKET_RECV_BUF.find("\n")
        if idx >= 0:
            line = SOCKET_RECV_BUF[:idx].rstrip("\r")
            SOCKET_RECV_BUF = SOCKET_RECV_BUF[idx + 1:]
            if not line.strip():
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                preview = line[:200]
                # Keep the reader tolerant to non-critical malformed lines.
                print(
                    f"Warning: skipping malformed JSON line from Unity: {e}. Payload preview: {preview!r}",
                    flush=True,
                )
                continue
        try:
            chunk = conn.recv(4096)
        except socket.timeout as e:
            raise TimeoutError(f"Socket receive timed out after {SOCKET_TIMEOUT_SEC} seconds.") from e
        if not chunk:
            trailing = SOCKET_RECV_BUF.strip()
            SOCKET_RECV_BUF = ""
            if trailing:
                print("Warning: discarding trailing unterminated socket data:", trailing, flush=True)
            return None
        SOCKET_RECV_BUF += chunk.decode("utf-8", errors="replace")
        if len(SOCKET_RECV_BUF) > SOCKET_MAX_RECV_BUF_BYTES:
            preview = SOCKET_RECV_BUF[-200:].replace("\n", "\\n")
            SOCKET_RECV_BUF = ""
            raise RuntimeError(
                f"Socket receive buffer exceeded {SOCKET_MAX_RECV_BUF_BYTES} bytes without a newline; "
                f"possible framing error or oversized message. Tail preview: {preview}"
            )


# -------------------- IO utils --------------------
def get_unique_folder(parent, folder_name):
    base_path = os.path.join(parent, folder_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path
    if os.path.isdir(base_path):
        visible_entries = [
            name for name in os.listdir(base_path)
            if name != ".DS_Store" and not name.endswith(".meta")
        ]
        if not visible_entries:
            return base_path
    k = 1
    while True:
        p = os.path.join(parent, f"{folder_name}_{k}")
        if not os.path.exists(p):
            os.makedirs(p)
            return p
        k += 1


def create_csv_file(csv_file_path, fieldnames):
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    write_header = not os.path.exists(csv_file_path)
    with open(csv_file_path, 'a+', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        if write_header:
            w.writeheader()


def write_data_to_csv(csv_file_path, fieldnames, rows):
    with open(csv_file_path, 'a+', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        w.writerows(rows)

# Frame transforms live in bo_normalize so that this backend, bo.py, mobo.py and the
# offline Meta-TAF source generators cannot drift apart. Thin wrappers keep the call
# sites (and the test suite) unchanged.
def denormalize_to_original_param(val01, lo, hi, decimals=3):
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


def expected_observation_columns():
    # Same layout as bo.py minus the context column (contexts are BoTorch-only).
    return (['UserID', 'ConditionID', 'GroupID', 'Timestamp', 'Iteration', 'Phase', 'IsBest']
            + objective_names + parameter_names)


# Per-iteration DBO diagnostics. Alpha is the one that matters: it is the fitted
# temporal decay rate, i.e. how much the model believes the objective moved between
# consecutive iterations. alpha == 1 means "no drift detected"; the further below 1,
# the more aggressively old observations are being discounted.
DBO_DIAGNOSTICS_FIELDS = [
    'Iteration', 'Phase', 'IsValidation', 'Alpha',
    'PredictedCost', 'PredictedSd', 'ObservedCost', 'ObservedObjective', 'SuggestSeconds',
]

# -------------------- protocol parsing --------------------
def parse_param_init(init_val):
    if isinstance(init_val, dict):
        if "low" not in init_val or "high" not in init_val:
            raise ValueError(f"Parameter init parse error (missing 'low'/'high'): {init_val}")
        return float(init_val["low"]), float(init_val["high"])
    parts = [p.strip() for p in str(init_val).split(",")]
    if len(parts) < 2:
        raise ValueError(f"Parameter init parse error: '{init_val}'")
    return float(parts[0]), float(parts[1])


def parse_obj_init(init_val):
    if isinstance(init_val, dict):
        if "low" not in init_val or "high" not in init_val:
            raise ValueError(f"Objective init parse error (missing 'low'/'high'): {init_val}")
        if "minimize" not in init_val:
            raise ValueError(f"Objective init parse error (missing 'minimize'): {init_val}")
        return float(init_val["low"]), float(init_val["high"]), int(init_val["minimize"])
    parts = [p.strip() for p in str(init_val).split(",")]
    if len(parts) < 3:
        raise ValueError(f"Objective init parse error: '{init_val}'")
    return float(parts[0]), float(parts[1]), int(float(parts[2]))


def get_cfg_int(cfg, key, default=None, required=False):
    if key in cfg and cfg.get(key) is not None:
        try:
            return int(cfg.get(key))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Config field '{key}' must be an integer, got {cfg.get(key)!r}") from e
    if required:
        raise ValueError(f"Missing required config field '{key}'")
    return int(default) if default is not None else None


def get_cfg_float(cfg, key, default=None, required=False):
    if key in cfg and cfg.get(key) is not None:
        try:
            return float(cfg.get(key))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Config field '{key}' must be a number, got {cfg.get(key)!r}") from e
    if required:
        raise ValueError(f"Missing required config field '{key}'")
    return float(default) if default is not None else None


def get_cfg_bool(cfg, key, default=None, required=False):
    if key in cfg and cfg.get(key) is not None:
        val = cfg.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)) and float(val) in (0.0, 1.0):
            return bool(val)
        if isinstance(val, str):
            token = val.strip().lower()
            if token in ("true", "1"):
                return True
            if token in ("false", "0"):
                return False
        raise ValueError(f"Config field '{key}' must be a boolean, got {cfg.get(key)!r}")
    if required:
        raise ValueError(f"Missing required config field '{key}'")
    return default


def get_cfg_str(cfg, key, default=""):
    val = cfg.get(key)
    if val is None:
        return default
    token = str(val).strip().lower()
    return token if token else default

# -------------------- objective evaluation --------------------
def recv_objectives_blocking(conn):
    while True:
        msg = recv_json_message(conn)
        if msg is None:
            return None
        if not isinstance(msg, dict):
            continue
        t = msg.get("type")
        if t == "objectives":
            return msg.get("values")
        continue


def objective_function(conn, x_norm):
    """Send one design to Unity, block for its objective, return the canonical frame.

    ``x_norm`` is a list of d values in [0,1]. Returns ``(f_max, raw_value)`` where
    ``f_max`` is in [-1,1] and points UP (maximization, minimize flag folded in) and
    ``raw_value`` is what Unity reported, in the objective's original units. The caller
    negates ``f_max`` before handing it to DynamicBO, which minimises cost.
    """
    values = {}
    for i, name in enumerate(parameter_names):
        lo, hi = parameters_info[i]
        values[name] = denormalize_to_original_param(x_norm[i], lo, hi, decimals=None)
    payload = {"type": "parameters", "values": values}
    print("Send parameters:", payload, flush=True)
    send_json_line(conn, payload)

    resp = recv_objectives_blocking(conn)
    if resp is None:
        raise RuntimeError("No objectives received from Unity.")
    if not isinstance(resp, dict):
        raise TypeError(f"Unity objectives payload must be a dict, got {type(resp).__name__}")

    name = objective_names[0]
    missing = [k for k in objective_names if k not in resp]
    if missing:
        raise KeyError(f"Unity objectives missing required key(s): {missing}")
    unexpected = sorted([k for k in resp.keys() if k not in set(objective_names)])
    if unexpected:
        raise KeyError(f"Unity objectives payload contains unexpected key(s): {unexpected}")
    try:
        val = float(resp[name])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Objective '{name}' must be numeric, got {resp[name]!r}") from e

    lo, hi, minflag = objectives_info[0]
    # Identical transform to bo.py: bounds check, map to [-1,1], flip if smaller is better.
    f = bo_normalize.normalize_objective_value(val, lo, hi, minflag, name=name)
    return float(f), float(val)

# -------------------- optimizer construction --------------------
def sobol_seed_points(n_samples):
    """Sampling-phase designs, drawn exactly as bo.py draws its initial data.

    Same Sobol call and same seed, so a DBO run and a BoTorch run configured
    identically visit the same sampling points and only diverge once the model
    takes over.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1 for non-warm-start runs.")
    bounds = torch.stack(
        [torch.zeros(PROBLEM_DIM, dtype=torch.double),
         torch.ones(PROBLEM_DIM, dtype=torch.double)],
        dim=0
    )
    pts = draw_sobol_samples(bounds=bounds, n=1, q=n_samples, seed=SEED).squeeze(0)
    print("Initial Sobol X in [0,1]:", pts, flush=True)
    return [[float(v) for v in row] for row in pts]


def build_optimizer(seed_points, num_seed_points):
    """Map the Unity init settings onto DynamicBO.

    The search box is the canonical [0,1]^d, so the frame is identical to bo.py's;
    dbo_torch appends the time column itself. Validation scheduling starts disabled
    and is switched on after the sampling phase (see ``dbo_execute``).
    """
    model_config = DBOModelConfig(
        spatial_kernel=DBO_SPATIAL_KERNEL,
        alpha_parameterization=DBO_ALPHA_PARAMETERIZATION,
        initial_alpha=DBO_INITIAL_ALPHA,
        stationary=DBO_STATIONARY_BASELINE,
    )
    config = DBOConfig(
        model=model_config,
        seed_points=seed_points,
        num_seed_points=num_seed_points,
        validation_every=None,
        exploration_ratio=EXPLORATION_RATIO,
        validation_confidence=DBO_VALIDATION_CONFIDENCE,
        validation_visited_only=DBO_VALIDATION_VISITED_ONLY,
        acquisition_time_offset=DBO_ACQUISITION_TIME_OFFSET,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        seed=SEED,
    )
    return DynamicBO(bounds=[(0.0, 1.0)] * PROBLEM_DIM, config=config)


def load_warm_start(dbo):
    """Replay warm-start CSV rows as the leading iterations of the time axis.

    Warm-start rows come from an EARLIER session and carry no timestamp, so the only
    coherent placement on a per-iteration time axis is in front of the live run: they
    become iterations 1..k and the live run continues at k+1. The temporal kernel then
    discounts them exactly as it discounts any other old observation, which is the
    behaviour you want — prior-session data should inform the model but not pin it.
    They are not written to ObservationsPerEvaluation.csv, matching bo.py: that file
    records the evaluations of this run.
    """
    if not CSV_PATH_PARAMETERS or not CSV_PATH_OBJECTIVES:
        raise ValueError("Warm start is enabled, but initial CSV paths are missing.")

    init_root = os.environ.get("BO_INIT_ROOT") or os.path.join(os.getcwd(), "InitData")
    x_path = os.path.join(init_root, CSV_PATH_PARAMETERS)
    y_path = os.path.join(init_root, CSV_PATH_OBJECTIVES)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Warm-start parameter CSV not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Warm-start objective CSV not found: {y_path}")

    x_df = pd.read_csv(x_path, delimiter=';')
    y_df = pd.read_csv(y_path, delimiter=';')

    missing_param_cols = [k for k in parameter_names if k not in x_df.columns]
    missing_obj_cols = [k for k in objective_names if k not in y_df.columns]
    if missing_param_cols:
        raise ValueError(f"Warm-start parameter CSV is missing columns: {missing_param_cols}")
    if missing_obj_cols:
        raise ValueError(f"Warm-start objective CSV is missing columns: {missing_obj_cols}")

    x_raw = x_df[parameter_names].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    y_raw = y_df[objective_names].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if x_raw.shape[0] != y_raw.shape[0]:
        raise ValueError(f"Warm-start rows mismatch: parameters={x_raw.shape[0]}, objectives={y_raw.shape[0]}")
    if x_raw.shape[0] < 1:
        raise ValueError("Warm-start CSVs must contain at least one data row.")
    if not np.all(np.isfinite(x_raw)):
        raise ValueError("Warm-start parameter CSV contains NaN/Inf values.")
    if not np.all(np.isfinite(y_raw)):
        raise ValueError("Warm-start objective CSV contains NaN/Inf values.")

    x_norm = np.zeros_like(x_raw, dtype=np.float64)
    for j in range(PROBLEM_DIM):
        lo, hi = parameters_info[j]
        x_norm[:, j] = normalize_param_column(x_raw[:, j], lo, hi)

    lo, hi, minflag = objectives_info[0]
    y_norm = normalize_obj_column(y_raw[:, 0], lo, hi, minflag)

    if not np.all(np.isfinite(x_norm)):
        raise ValueError("Warm-start normalized parameters contain non-finite values.")
    if not np.all(np.isfinite(y_norm)):
        raise ValueError("Warm-start normalized objectives contain non-finite values.")

    for i in range(x_norm.shape[0]):
        f = float(y_norm[i])
        dbo.observe([float(v) for v in x_norm[i]], -f)  # cost = -f, DynamicBO minimises
        ALL_NORM_MAX.append(f)
    print(f"Warm start: replayed {x_norm.shape[0]} prior observation(s) as iterations "
          f"1..{x_norm.shape[0]} of the time axis.", flush=True)

# -------------------- logging --------------------
def best_objective_so_far():
    """Best normalized objective over every observation of this run (maximization).

    Kept bit-identical to bo.py's 'coverage' metric so the Unity progress display and
    the downstream analysis scripts behave the same. Note this is a best-EVER record:
    under drift the value that produced it may no longer be attainable, which is what
    the per-iteration alpha in DboDiagnosticsPerEvaluation.csv is there to expose.
    """
    if not ALL_NORM_MAX:
        print("Warning: no observations yet; reporting metric -1.0.", flush=True)
        return -1.0
    return max(ALL_NORM_MAX)


def record_observation(iteration, phase, f_max, x_norm):
    """Buffer one evaluation for ObservationsPerEvaluation.csv, then rewrite the file."""
    lo, hi, minflag = objectives_info[0]
    y_den = denormalize_to_original_obj(f_max, lo, hi, minflag)
    x_den = [denormalize_to_original_param(x_norm[j], parameters_info[j][0], parameters_info[j][1])
             for j in range(PROBLEM_DIM)]
    OBSERVATION_ROWS.append((
        iteration, phase,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        y_den, x_den,
    ))
    write_observations_csv()


def write_observations_csv():
    """Rewrite ObservationsPerEvaluation.csv with globally correct IsBest flags.

    PROJECT_PATH is a freshly created run folder, so this file only ever holds the
    current run; rewriting it in full is simpler than bo.py's append-then-patch dance
    and produces the same artifact. IsBest competes over ALL observations, warm-start
    rows included, so a run whose best point came from the warm-start data has no
    TRUE row — same as bo.py.
    """
    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    os.makedirs(PROJECT_PATH, exist_ok=True)

    flags = []
    if ALL_NORM_MAX:
        best_norm = max(ALL_NORM_MAX)
        flags = ['TRUE' if abs(v - best_norm) < 1e-12 else 'FALSE' for v in ALL_NORM_MAX]
    tail = flags[-len(OBSERVATION_ROWS):] if OBSERVATION_ROWS else []

    with open(obs_csv, 'w', newline='') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(expected_observation_columns())
        for row, is_best in zip(OBSERVATION_ROWS, tail):
            iteration, phase, timestamp, y_den, x_den = row
            w.writerow([USER_ID, CONDITION_ID, GROUP_ID, timestamp,
                        iteration, phase, is_best, y_den, *x_den])


def save_metric_to_file(metric_values, iteration):
    os.makedirs(PROJECT_PATH, exist_ok=True)
    best_csv = os.path.join(PROJECT_PATH, "BestObjectivePerEvaluation.csv")
    legacy_csv = os.path.join(PROJECT_PATH, "HypervolumePerEvaluation.csv")

    write_best_header = not os.path.exists(best_csv) or os.path.getsize(best_csv) == 0
    with open(best_csv, 'a', newline='') as f:
        w = csv.writer(f, delimiter=';')
        if write_best_header:
            w.writerow(["BestObjective", "Run"])
        w.writerow([metric_values[-1], iteration])

    # Legacy mirror for older analysis scripts that still read this file.
    write_legacy_header = not os.path.exists(legacy_csv) or os.path.getsize(legacy_csv) == 0
    with open(legacy_csv, 'a', newline='') as f:
        w = csv.writer(f, delimiter=';')
        if write_legacy_header:
            w.writerow(["Hypervolume", "Run"])
        w.writerow([metric_values[-1], iteration])


def save_dbo_diagnostics(iteration, phase, observation, alpha, raw_value, suggest_seconds):
    """Append the DBO-specific per-iteration diagnostics, alpha first among them."""
    diag_csv = os.path.join(PROJECT_PATH, "DboDiagnosticsPerEvaluation.csv")
    write_data_to_csv(diag_csv, DBO_DIAGNOSTICS_FIELDS, [{
        'Iteration': iteration,
        'Phase': phase,
        'IsValidation': 'TRUE' if observation.is_validation else 'FALSE',
        # Empty during the sampling phase: no model has been fitted yet. Kept at 8
        # decimals because the interesting regime is just below 1: 0.99 already means
        # a 1% discount per iteration, which is a lot over a 40-iteration study.
        'Alpha': '' if alpha is None else round(float(alpha), 8),
        # Predicted/observed values are COSTS (= -objective in the canonical frame).
        'PredictedCost': '' if observation.predicted_y is None else round(float(observation.predicted_y), 6),
        'PredictedSd': '' if observation.predicted_sd is None else round(float(observation.predicted_sd), 6),
        'ObservedCost': round(float(observation.y), 6),
        'ObservedObjective': raw_value,
        'SuggestSeconds': round(float(suggest_seconds), 4),
    }])

# -------------------- main loop --------------------
def dbo_execute(conn, iterations, initial_samples):
    global PROJECT_PATH, OBSERVATIONS_LOG_PATH
    base = os.environ.get("BO_LOG_ROOT") or os.path.join(os.getcwd(), "LogData")
    condition_base = os.path.join(base, USER_LOG_ID, CONDITION_LOG_ID)
    os.makedirs(condition_base, exist_ok=True)
    PROJECT_PATH = get_unique_folder(condition_base, "run")
    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")

    exec_csv = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(exec_csv, ['Optimization', 'Execution_Time'])
    create_csv_file(os.path.join(PROJECT_PATH, 'DboDiagnosticsPerEvaluation.csv'),
                    DBO_DIAGNOSTICS_FIELDS)

    torch.manual_seed(SEED)

    if WARM_START:
        # Warm-start rows already occupy the leading time slots, so no seed points.
        dbo = build_optimizer(None, 0)
        load_warm_start(dbo)
    else:
        dbo = build_optimizer(sobol_seed_points(initial_samples), initial_samples)

    metric_values = []  # best normalized objective per evaluation

    # ---- sampling phase: global iterations 1..initial_samples ----------------
    # suggest() replays the Sobol seed points here; observe() stamps each one with
    # time = its global iteration index. Nothing resets between phases.
    if not WARM_START:
        for i in range(initial_samples):
            print(f"---- Initial Sample {i+1}", flush=True)
            t0 = time.time()
            x = dbo.suggest()
            suggest_seconds = time.time() - t0
            f, raw_value = objective_function(conn, x)
            obs = dbo.observe(x, -f)  # cost = -f
            ALL_NORM_MAX.append(f)
            record_observation(obs.iteration, 'sampling', f, x)
            save_dbo_diagnostics(obs.iteration, 'sampling', obs, dbo.alpha, raw_value, suggest_seconds)
            send_json_line(conn, {"type": "tempCoverage",
                                  "value": float(i + 1) / float(max(1, initial_samples))})

    # Validation iterations need a fitted model, so scheduling only starts once the
    # sampling phase is done. The schedule is evaluated on the GLOBAL iteration index,
    # which is also the model's time coordinate: with 5 sampling iterations and
    # dboValidationEvery=5, the first validation step is global iteration 10.
    dbo.config.validation_every = DBO_VALIDATION_EVERY if DBO_VALIDATION_EVERY > 0 else None

    best = best_objective_so_far()
    metric_values.append(best)
    save_metric_to_file(metric_values, 0)
    send_json_line(conn, {"type": "coverage", "value": float(best)})

    # ---- optimization phase: global iterations continue from the sampling phase ----
    for it in range(1, iterations + 1):
        is_validation = dbo.is_validation_iteration()
        t0 = time.time()
        x = dbo.suggest()  # fits the GP (alpha included) and maximises LogEI at time t
        suggest_seconds = time.time() - t0
        write_data_to_csv(exec_csv, ['Optimization', 'Execution_Time'],
                          [{'Optimization': it, 'Execution_Time': suggest_seconds}])

        alpha = dbo.alpha
        print(f"---- Optimization {it} (global iteration {dbo.next_iteration}"
              f"{', validation' if is_validation else ''}): "
              f"alpha={'unfitted' if alpha is None else f'{alpha:.6f}'}", flush=True)

        f, raw_value = objective_function(conn, x)
        obs = dbo.observe(x, -f)  # cost = -f
        ALL_NORM_MAX.append(f)

        best = best_objective_so_far()
        metric_values.append(best)
        record_observation(obs.iteration, 'optimization', f, x)
        save_metric_to_file(metric_values, it)
        save_dbo_diagnostics(obs.iteration, 'optimization', obs, alpha, raw_value, suggest_seconds)
        send_json_line(conn, {"type": "coverage", "value": float(best)})

    final_alpha = dbo.alpha
    print("DBO finished:", dict(
        observations=dbo.num_observations,
        alpha='unfitted' if final_alpha is None else round(float(final_alpha), 8),
        best_normalized_objective=best_objective_so_far(),
    ), flush=True)

    send_json_line(conn, {"type": "optimization_finished"})
    return metric_values, dbo

# -------------------- boot --------------------
def main():
    global N_INITIAL, N_ITERATIONS, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, SEED
    global PROBLEM_DIM, NUM_OBJS
    global WARM_START, CSV_PATH_PARAMETERS, CSV_PATH_OBJECTIVES, WARM_START_OBJECTIVE_FORMAT
    global USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID
    global parameter_names, objective_names, parameters_info, objectives_info
    global DBO_SPATIAL_KERNEL, DBO_ALPHA_PARAMETERIZATION, DBO_INITIAL_ALPHA
    global DBO_ACQUISITION_TIME_OFFSET, DBO_VALIDATION_EVERY, DBO_VALIDATION_CONFIDENCE
    global DBO_VALIDATION_VISITED_ONLY, DBO_STATIONARY_BASELINE, EXPLORATION_RATIO
    global SOCKET_ACCEPT_TIMEOUT_SEC
    global SOCKET_RECV_BUF

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    conn = None
    try:
        if SOCKET_ACCEPT_TIMEOUT_SEC <= 0:
            raise ValueError(f"BO_ACCEPT_TIMEOUT_SEC must be > 0, got {SOCKET_ACCEPT_TIMEOUT_SEC}")
        s.settimeout(SOCKET_ACCEPT_TIMEOUT_SEC)
        s.bind((HOST, PORT))
        s.listen(1)
        print('Server starts, waiting for connection...', flush=True)
        try:
            conn, addr = s.accept()
        except socket.timeout as e:
            raise TimeoutError(f"Socket accept timed out after {SOCKET_ACCEPT_TIMEOUT_SEC} seconds.") from e
        print('Connected by', addr, flush=True)
        if SOCKET_TIMEOUT_SEC <= 0:
            raise ValueError(f"BO_SOCKET_TIMEOUT_SEC must be > 0, got {SOCKET_TIMEOUT_SEC}")
        conn.settimeout(SOCKET_TIMEOUT_SEC)
        SOCKET_RECV_BUF = ""

        # receive init
        init_msg = None
        while True:
            msg = recv_json_message(conn)
            if msg is None:
                break
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "init":
                init_msg = msg
                break
            continue
        if init_msg is None:
            raise RuntimeError("Did not receive init message.")

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

        # DBO-specific settings. Every one of them is optional: a Unity build that does
        # not know about this backend yet still produces a valid, sensible run.
        DBO_SPATIAL_KERNEL = get_cfg_str(cfg, "dboSpatialKernel", DBO_SPATIAL_KERNEL)
        DBO_ALPHA_PARAMETERIZATION = get_cfg_str(cfg, "dboAlphaParameterization", DBO_ALPHA_PARAMETERIZATION)
        DBO_INITIAL_ALPHA = get_cfg_float(cfg, "dboInitialAlpha", default=DBO_INITIAL_ALPHA)
        DBO_ACQUISITION_TIME_OFFSET = get_cfg_float(cfg, "dboAcquisitionTimeOffset", default=DBO_ACQUISITION_TIME_OFFSET)
        DBO_VALIDATION_EVERY = get_cfg_int(cfg, "dboValidationEvery", default=DBO_VALIDATION_EVERY)
        DBO_VALIDATION_CONFIDENCE = get_cfg_float(cfg, "dboValidationConfidence", default=DBO_VALIDATION_CONFIDENCE)
        DBO_VALIDATION_VISITED_ONLY = get_cfg_bool(cfg, "dboValidationVisitedOnly", default=DBO_VALIDATION_VISITED_ONLY)
        DBO_STATIONARY_BASELINE = get_cfg_bool(cfg, "dboStationaryBaseline", default=DBO_STATIONARY_BASELINE)
        EXPLORATION_RATIO = get_cfg_float(cfg, "explorationRatio", default=EXPLORATION_RATIO)

        if PROBLEM_DIM < 1:
            raise ValueError(f"nParameters must be >= 1, got {PROBLEM_DIM}")
        if NUM_OBJS != 1:
            raise ValueError(
                f"The DBO backend is single-objective: it models ONE drifting cost. "
                f"Got nObjectives={NUM_OBJS}. Configure exactly one objective, or use the "
                f"BoTorch/MetaTAF backends for multi-objective studies."
            )
        if N_INITIAL < 0 or N_ITERATIONS < 0:
            raise ValueError(f"Iteration counts must be non-negative, got sampling={N_INITIAL}, optimization={N_ITERATIONS}")
        if (not WARM_START) and N_INITIAL < 1:
            raise ValueError(
                "numSamplingIterations must be >= 1 when warmStart is disabled. "
                f"Got sampling={N_INITIAL}, warmStart={WARM_START}."
            )
        if NUM_RESTARTS < 1 or RAW_SAMPLES < 1:
            raise ValueError(
                f"numRestarts/rawSamples must be >=1, got {NUM_RESTARTS}/{RAW_SAMPLES}"
            )
        if WARM_START_OBJECTIVE_FORMAT not in ("auto", "raw", "normalized_max", "normalized_native"):
            raise ValueError(
                "warmStartObjectiveFormat must be one of: auto, raw, normalized_max, normalized_native; "
                f"got '{WARM_START_OBJECTIVE_FORMAT}'"
            )
        if BATCH_SIZE != 1:
            print(f"Warning: batchSize={BATCH_SIZE} is not supported in this HITL loop; forcing batchSize=1.", flush=True)
            BATCH_SIZE = 1

        if DBO_SPATIAL_KERNEL not in ("rbf", "matern52"):
            raise ValueError(f"dboSpatialKernel must be 'rbf' or 'matern52', got '{DBO_SPATIAL_KERNEL}'")
        if DBO_ALPHA_PARAMETERIZATION not in ("decay", "direct"):
            raise ValueError(
                f"dboAlphaParameterization must be 'decay' or 'direct', got '{DBO_ALPHA_PARAMETERIZATION}'"
            )
        if not (0.0 < DBO_INITIAL_ALPHA <= 1.0):
            raise ValueError(f"dboInitialAlpha must lie in (0, 1], got {DBO_INITIAL_ALPHA}")
        if not np.isfinite(DBO_ACQUISITION_TIME_OFFSET):
            raise ValueError(f"dboAcquisitionTimeOffset must be finite, got {DBO_ACQUISITION_TIME_OFFSET}")
        if not (0.0 < DBO_VALIDATION_CONFIDENCE < 1.0):
            raise ValueError(
                "dboValidationConfidence is a tail mass and must lie in (0, 1); "
                f"got {DBO_VALIDATION_CONFIDENCE} (pass 0.01, not 0.99)."
            )
        if not np.isfinite(EXPLORATION_RATIO) or EXPLORATION_RATIO < 0.0:
            raise ValueError(f"explorationRatio must be >= 0, got {EXPLORATION_RATIO}")
        if DBO_STATIONARY_BASELINE:
            print("Warning: dboStationaryBaseline is on; alpha is frozen at 1 and this run is a "
                  "stationary-BO control, not a DBO run.", flush=True)

        user = init_msg.get("user", {}) or {}
        USER_ID      = normalize_user_token(user.get("userId"), default="-1")
        CONDITION_ID = normalize_user_token(user.get("conditionId"), default="-1")
        GROUP_ID     = normalize_user_token(user.get("groupId"), default="-1")
        USER_LOG_ID  = normalize_log_folder_token(USER_ID, default="-1")
        CONDITION_LOG_ID = normalize_log_folder_token(CONDITION_ID, default="-1")
        if USER_LOG_ID != USER_ID:
            print(
                f"Warning: userId '{USER_ID}' was normalized to safe log-folder token '{USER_LOG_ID}'.",
                flush=True,
            )
        if CONDITION_LOG_ID != CONDITION_ID:
            print(
                f"Warning: conditionId '{CONDITION_ID}' was normalized to safe log-folder token '{CONDITION_LOG_ID}'.",
                flush=True,
            )

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
        for i, (lo, hi) in enumerate(parameters_info):
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"Parameter '{parameter_names[i]}' bounds must be finite, got ({lo}, {hi})")
            if hi < lo:
                raise ValueError(f"Parameter '{parameter_names[i]}' has invalid bounds: low={lo} > high={hi}")
            if hi == lo:
                # DynamicBO requires hi > lo on every axis; a frozen parameter has no
                # search dimension and must be removed from the study configuration.
                raise ValueError(
                    f"Parameter '{parameter_names[i]}' has a degenerate range [{lo}, {hi}]. "
                    "The DBO backend needs a non-empty range on every parameter."
                )
        for i, (lo, hi, minflag) in enumerate(objectives_info):
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"Objective '{objective_names[i]}' bounds must be finite, got ({lo}, {hi})")
            if hi < lo:
                raise ValueError(f"Objective '{objective_names[i]}' has invalid bounds: low={lo} > high={hi}")
            if int(minflag) not in (0, 1):
                raise ValueError(f"Objective '{objective_names[i]}' minimize flag must be 0 or 1, got {minflag}")

        context_cfg = init_msg.get("context") or {}
        if isinstance(context_cfg, dict) and bool(context_cfg.get("enabled", False)):
            raise ValueError(
                "Contextual optimization (LCE-M GP) is not supported by the DBO backend: "
                "DBO's extra input dimension is time, not a context embedding. "
                "Disable contextual optimization or use the BoTorch backend."
            )

        print("Init OK:", dict(
            BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES,
            N_ITERATIONS=N_ITERATIONS, MC_SAMPLES=MC_SAMPLES,
            N_INITIAL=N_INITIAL, SEED=SEED, PROBLEM_DIM=PROBLEM_DIM, NUM_OBJS=NUM_OBJS,
            WARM_START=WARM_START,
        ), flush=True)
        print("DBO settings:", dict(
            spatialKernel=DBO_SPATIAL_KERNEL,
            alphaParameterization=DBO_ALPHA_PARAMETERIZATION,
            initialAlpha=DBO_INITIAL_ALPHA,
            acquisitionTimeOffset=DBO_ACQUISITION_TIME_OFFSET,
            validationEvery=DBO_VALIDATION_EVERY,
            validationConfidence=DBO_VALIDATION_CONFIDENCE,
            validationVisitedOnly=DBO_VALIDATION_VISITED_ONLY,
            stationaryBaseline=DBO_STATIONARY_BASELINE,
            explorationRatio=EXPLORATION_RATIO,
        ), flush=True)
        if WARM_START:
            print(f"Time axis: replayed warm-start rows first, then live iterations; "
                  f"one continuous clock, optimization spans {N_ITERATIONS} live iterations.",
                  flush=True)
        else:
            print(f"Time axis: iterations 1..{N_INITIAL + N_ITERATIONS} "
                  f"(sampling 1..{N_INITIAL}, "
                  f"optimization {N_INITIAL + 1}..{N_INITIAL + N_ITERATIONS}); "
                  "one continuous clock across both phases.", flush=True)

        dbo_execute(conn, N_ITERATIONS, N_INITIAL)
    finally:
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        s.close()


if __name__ == "__main__":
    main()
