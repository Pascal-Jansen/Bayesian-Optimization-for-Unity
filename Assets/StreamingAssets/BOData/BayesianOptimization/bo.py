# bo.py — single-objective Bayesian Optimization (NDJSON protocol)
# Uses qLogNoisyExpectedImprovement.
# Logs observations to ObservationsPerEvaluation.csv ('IsBest' flag) and
# best-so-far metric to BestObjectivePerEvaluation.csv.
# Also mirrors the metric to legacy HypervolumePerEvaluation.csv for compatibility.

import json
import socket
import time
import csv
import os
import numpy as np
import pandas as pd
import torch

from botorch.acquisition.logei import qLogNoisyExpectedImprovement  # LogNEI
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

# -------------------- defaults (overwritten by Unity init) --------------------
N_INITIAL = 5
N_ITERATIONS = 10
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 1024
MC_SAMPLES = 512
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

# names and meta parsed from init
parameter_names = []
objective_names = []
parameters_info = []   # [(lo, hi)]
objectives_info = []   # [(lo, hi, minimizeFlag)]  # minimizeFlag==1 means minimize in original scale

# device
tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
device = torch.device("cpu")

# -------------------- TCP server helpers --------------------
HOST = ''
PORT = 56001
SOCKET_TIMEOUT_SEC = float(os.environ.get("BO_SOCKET_TIMEOUT_SEC", "3600"))
SOCKET_ACCEPT_TIMEOUT_SEC = float(os.environ.get("BO_ACCEPT_TIMEOUT_SEC", "300"))
SOCKET_MAX_RECV_BUF_BYTES = int(os.environ.get("BO_MAX_RECV_BUF_BYTES", "1048576"))
SOCKET_RECV_BUF = ""

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
                print("JSON decode error:", e, "line:", line, flush=True)
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


def ndjson_reader(conn):
    while True:
        msg = recv_json_message(conn)
        if msg is None:
            return
        yield msg

# -------------------- IO utils --------------------
def get_unique_folder(parent, folder_name):
    base_path = os.path.join(parent, folder_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
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

def denormalize_to_original_param(val01, lo, hi, decimals=3):
    v = lo + val01 * (hi - lo)
    if decimals is None:
        return float(v)
    return np.round(v, decimals)

def denormalize_to_original_obj(v_m1p1, lo, hi, smaller_is_better):
    v = -v_m1p1 if int(smaller_is_better) == 1 else v_m1p1
    return np.round(lo + (v + 1) * 0.5 * (hi - lo), 3)


def normalize_param_column(col, lo, hi):
    col = np.asarray(col, dtype=np.float64)
    if hi == lo:
        if not np.allclose(col, lo, rtol=0.0, atol=1e-12):
            raise ValueError(f"Parameter values out of bounds for degenerate interval [{lo}, {hi}]")
        return np.zeros_like(col, dtype=np.float64)
    eps = 1e-9
    if np.any(col < (lo - eps)) or np.any(col > (hi + eps)):
        raise ValueError(f"Parameter values out of bounds [{lo}, {hi}]")
    x01 = (col - lo) / (hi - lo)
    return np.clip(x01, 0.0, 1.0)


def normalize_obj_column(col, lo, hi, minflag):
    col = np.asarray(col, dtype=np.float64)
    raw_range_detected = np.all((lo - 1e-8 <= col) & (col <= hi + 1e-8))
    norm_range_detected = np.all((-1.0 - 1e-8 <= col) & (col <= 1.0 + 1e-8))
    in_raw_range = raw_range_detected
    in_norm_range = norm_range_detected

    if WARM_START_OBJECTIVE_FORMAT == "raw":
        if not raw_range_detected:
            raise ValueError(
                f"warmStartObjectiveFormat=raw requires values in [{lo},{hi}], "
                f"but received range [{np.min(col)}, {np.max(col)}]"
            )
        in_raw_range = True
        in_norm_range = False
    elif WARM_START_OBJECTIVE_FORMAT == "normalized_max":
        if not norm_range_detected:
            raise ValueError(
                f"warmStartObjectiveFormat=normalized_max requires values in [-1,1], "
                f"but received range [{np.min(col)}, {np.max(col)}]"
            )
        in_raw_range = False
        in_norm_range = True
    elif WARM_START_OBJECTIVE_FORMAT == "normalized_native":
        if not norm_range_detected:
            raise ValueError(
                f"warmStartObjectiveFormat=normalized_native requires values in [-1,1], "
                f"but received range [{np.min(col)}, {np.max(col)}]"
            )
        in_raw_range = False
        in_norm_range = True

    if in_raw_range:
        if WARM_START_OBJECTIVE_FORMAT == "auto" and in_norm_range:
            print(
                "Warning: warm-start objective values are ambiguous (fit both raw bounds and [-1,1]); assuming raw scale.",
                flush=True,
            )
        if hi == lo:
            y = np.zeros_like(col)
        else:
            y = (col - lo) / (hi - lo) * 2.0 - 1.0
            if int(minflag) == 1:
                y = -y
    elif in_norm_range:
        # already normalized
        y = np.clip(col, -1.0, 1.0)
        if WARM_START_OBJECTIVE_FORMAT == "normalized_native" and int(minflag) == 1:
            y = -y
    else:
        raise ValueError(
            f"Warm-start objective values must be within raw bounds [{lo}, {hi}] or normalized [-1,1], "
            f"got range [{np.min(col)}, {np.max(col)}]"
        )
    return np.clip(y, -1.0, 1.0)


def expected_observation_columns():
    return ['UserID','ConditionID','GroupID','Timestamp','Iteration','Phase','IsBest'] + objective_names + parameter_names

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
            return msg.get("values") or {}
        elif t in ("coverage", "tempCoverage", "optimization_finished"):
            continue
        elif t == "log":
            print("LOG:", msg.get("message", ""), flush=True)
            continue

def objective_function(conn, x_tensor):
    x = x_tensor.cpu().numpy()
    values = {}
    for i, name in enumerate(parameter_names):
        lo, hi = parameters_info[i]
        values[name] = denormalize_to_original_param(x[i], lo, hi, decimals=None)
    payload = {"type": "parameters", "values": values}
    print("Send parameters:", payload, flush=True)
    send_json_line(conn, payload)

    resp = recv_objectives_blocking(conn)
    if resp is None:
        raise RuntimeError("No objectives received from Unity.")
    if not isinstance(resp, dict):
        raise TypeError(f"Unity objectives payload must be a dict, got {type(resp).__name__}")

    # normalize to [-1,1] and maximize
    name = objective_names[0]
    if name not in resp:
        raise KeyError(f"Unity objectives missing required key(s): {[name]}")
    try:
        val = float(resp[name])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Objective '{name}' must be numeric, got {resp[name]!r}") from e
    if not np.isfinite(val):
        raise ValueError(f"Objective '{name}' is non-finite: {val}")
    lo, hi, minflag = objectives_info[0]
    eps = 1e-9
    if hi == lo:
        if not np.isclose(val, lo, rtol=0.0, atol=eps):
            raise ValueError(f"Objective '{name}' value {val} is out of bounds for degenerate interval [{lo}, {hi}]")
        f = 0.0
    else:
        if val < (lo - eps) or val > (hi + eps):
            raise ValueError(f"Objective '{name}' value {val} is out of bounds [{lo}, {hi}]")
        f = (val - lo) / (hi - lo) * 2 - 1
    if int(minflag) == 1:
        f *= -1
    f = float(np.clip(f, -1.0, 1.0))
    return torch.tensor([f], dtype=torch.double)

# -------------------- data IO --------------------
def generate_initial_data(conn, n_samples):
    global PROJECT_PATH
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1 for non-warm-start runs.")

    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    if not os.path.exists(obs_csv):
        # NOTE: 'IsBest' replaces 'IsPareto'
        header = expected_observation_columns()
        with open(obs_csv, 'w', newline='') as f:
            csv.writer(f, delimiter=';').writerow(header)

    train_x = draw_sobol_samples(bounds=problem_bounds, n=1, q=n_samples, seed=SEED).squeeze(0)
    print("Initial Sobol X in [0,1]:", train_x, flush=True)

    train_obj = []
    best_so_far = -1e9
    for i, x in enumerate(train_x):
        print(f"---- Initial Sample {i+1}", flush=True)
        y = objective_function(conn, x)  # shape [1]
        train_obj.append(y)

        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()  # normalized [-1,1]
        # denormalize objective back to original scale for logging
        y_den = denormalize_to_original_obj(y_np[0], objectives_info[0][0], objectives_info[0][1], objectives_info[0][2])
        x_den = [denormalize_to_original_param(x_np[j], parameters_info[j][0], parameters_info[j][1]) for j in range(PROBLEM_DIM)]

        # IsBest for this row compared to best_so_far
        is_best = float(y_np[0]) > best_so_far + 1e-12
        if is_best:
            best_so_far = float(y_np[0])

        row = [USER_ID, CONDITION_ID, GROUP_ID,
               time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
               i+1, 'sampling', 'TRUE' if is_best else 'FALSE', y_den, *x_den]
        with open(obs_csv, 'a', newline='') as f:
            csv.writer(f, delimiter=';').writerow(row)

        send_json_line(conn, {"type": "tempCoverage", "value": float(i+1)/float(max(1,n_samples))})

    Y = torch.tensor(np.stack([t.numpy() for t in train_obj], axis=0), dtype=torch.double)  # shape [n,1]
    return train_x, Y

def load_data():
    cur = os.getcwd()
    if not CSV_PATH_PARAMETERS or not CSV_PATH_OBJECTIVES:
        raise ValueError("Warm start is enabled, but initial CSV paths are missing.")

    x_path = os.path.join(cur, "InitData", CSV_PATH_PARAMETERS)
    y_path = os.path.join(cur, "InitData", CSV_PATH_OBJECTIVES)
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

    y_norm = np.zeros_like(y_raw, dtype=np.float64)
    for j in range(NUM_OBJS):
        lo, hi, minflag = objectives_info[j]
        y_norm[:, j] = normalize_obj_column(y_raw[:, j], lo, hi, minflag)

    if not np.all(np.isfinite(x_norm)):
        raise ValueError("Warm-start normalized parameters contain non-finite values.")
    if not np.all(np.isfinite(y_norm)):
        raise ValueError("Warm-start normalized objectives contain non-finite values.")

    return torch.tensor(x_norm, dtype=torch.double), torch.tensor(y_norm, dtype=torch.double)

# -------------------- model --------------------
def initialize_model(train_x, train_obj):
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# -------------------- acquisition (single-objective, LogNEI) --------------------
def optimize_candidates(model, sampler):
    X_baseline = model.train_inputs[0]
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
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    return candidates.detach()  # in [0,1]

# -------------------- logging --------------------
def save_xy(x_sample, y_sample, iteration):
    # Reuse ObservationsPerEvaluation.csv with 'IsBest'
    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    x_np = x_sample.clone().cpu().numpy()
    y_np = y_sample.clone().cpu().numpy()

    # denormalize last row
    for j in range(PROBLEM_DIM):
        x_np[-1][j] = denormalize_to_original_param(x_np[-1][j], parameters_info[j][0], parameters_info[j][1])
    y_np[-1][0] = denormalize_to_original_obj(y_np[-1][0], objectives_info[0][0], objectives_info[0][1], objectives_info[0][2])

    if os.path.exists(obs_csv):
        df = pd.read_csv(obs_csv, delimiter=';')
        expected_cols = expected_observation_columns()
        if list(df.columns) != expected_cols:
            raise ValueError(
                f"ObservationsPerEvaluation.csv columns mismatch. "
                f"Expected {expected_cols}, got {list(df.columns)}"
            )
    else:
        cols = expected_observation_columns()
        df = pd.DataFrame(columns=cols)

    new_row = pd.DataFrame([[USER_ID, CONDITION_ID, GROUP_ID,
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                             int(x_sample.shape[0]), 'optimization', 'FALSE',
                             y_np[-1][0], *x_np[-1]]], columns=df.columns)
    if df.empty:
        df = new_row.copy()
    else:
        df = pd.concat([df, new_row], ignore_index=True)

    # Recompute IsBest across all rows: mark rows whose normalized objective equals current best
    vals_norm = y_sample.squeeze(-1).detach().cpu().tolist()  # length == number of rows in y_sample
    best_norm = max(vals_norm) if len(vals_norm) > 0 else float(y_sample[-1].item())

    # df rows correspond to all past + current rows. The number of objective entries equals df length.
    # Build flags from vals_norm.
    flags = ['TRUE' if abs(v - best_norm) < 1e-12 else 'FALSE' for v in vals_norm]
    if len(flags) == len(df):
        df['IsBest'] = flags
    elif len(flags) > 0 and len(df) >= len(flags):
        start = len(df) - len(flags)
        df.loc[df.index[start:], 'IsBest'] = flags
        print("Warning: CSV row count mismatch; updated IsBest only for current run rows.", flush=True)
    elif len(df) > 0:
        df.loc[df.index[-1], 'IsBest'] = 'TRUE' if abs(float(y_sample[-1][0]) - best_norm) < 1e-12 else 'FALSE'

    df.to_csv(obs_csv, sep=';', index=False)

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

# -------------------- main loop --------------------
def bo_execute(conn, seed, iterations, initial_samples):
    global PROJECT_PATH, OBSERVATIONS_LOG_PATH
    base = os.path.join(os.getcwd(), "LogData")
    os.makedirs(base, exist_ok=True)
    PROJECT_PATH = get_unique_folder(base, USER_ID)
    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")

    exec_csv = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(exec_csv, ['Optimization', 'Execution_Time'])

    torch.manual_seed(seed)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]), seed=SEED)

    metric_values = []  # best normalized objective per evaluation

    if WARM_START:
        train_x, train_y = load_data()
    else:
        train_x, train_y = generate_initial_data(conn, n_samples=initial_samples)

    mll, model = initialize_model(train_x, train_y)

    best = torch.max(train_y).item()
    metric_values.append(best)
    save_metric_to_file(metric_values, 0)
    send_json_line(conn, {"type": "coverage", "value": float(best)})

    for it in range(1, iterations + 1):
        t0 = time.time()
        fit_gpytorch_mll(mll)
        new_x = optimize_candidates(model, sampler)
        t_elapsed = time.time() - t0
        write_data_to_csv(exec_csv, ['Optimization', 'Execution_Time'],
                          [{'Optimization': it, 'Execution_Time': t_elapsed}])

        new_y = objective_function(conn, new_x[0])  # shape [1]
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y.unsqueeze(0)])  # shape [n+1,1]

        best = torch.max(train_y).item()
        metric_values.append(best)
        save_xy(train_x, train_y, it)
        save_metric_to_file(metric_values, it)
        send_json_line(conn, {"type": "coverage", "value": float(best)})

        mll, model = initialize_model(train_x, train_y)

    send_json_line(conn, {"type": "optimization_finished"})
    return metric_values, train_x, train_y

# -------------------- boot --------------------
def main():
    global N_INITIAL, N_ITERATIONS, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, SEED
    global PROBLEM_DIM, NUM_OBJS, problem_bounds
    global WARM_START, CSV_PATH_PARAMETERS, CSV_PATH_OBJECTIVES, WARM_START_OBJECTIVE_FORMAT
    global USER_ID, CONDITION_ID, GROUP_ID
    global parameter_names, objective_names, parameters_info, objectives_info
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

        if PROBLEM_DIM < 1:
            raise ValueError(f"nParameters must be >= 1, got {PROBLEM_DIM}")
        if NUM_OBJS != 1:
            raise ValueError(f"bo.py expects exactly 1 objective, got {NUM_OBJS}")
        if N_INITIAL < 0 or N_ITERATIONS < 0:
            raise ValueError(f"Iteration counts must be non-negative, got sampling={N_INITIAL}, optimization={N_ITERATIONS}")
        if NUM_RESTARTS < 1 or RAW_SAMPLES < 1 or MC_SAMPLES < 1:
            raise ValueError(
                f"numRestarts/rawSamples/mcSamples must be >=1, got {NUM_RESTARTS}/{RAW_SAMPLES}/{MC_SAMPLES}"
            )
        if WARM_START_OBJECTIVE_FORMAT not in ("auto", "raw", "normalized_max", "normalized_native"):
            raise ValueError(
                "warmStartObjectiveFormat must be one of: auto, raw, normalized_max, normalized_native; "
                f"got '{WARM_START_OBJECTIVE_FORMAT}'"
            )
        if BATCH_SIZE != 1:
            print(f"Warning: batchSize={BATCH_SIZE} is not supported in this HITL loop; forcing batchSize=1.", flush=True)
            BATCH_SIZE = 1

        user = init_msg.get("user", {}) or {}
        USER_ID      = str(user.get("userId", "user"))
        CONDITION_ID = str(user.get("conditionId", "cond"))
        GROUP_ID     = str(user.get("groupId", "grp"))

        parameters = init_msg.get("parameters", []) or []
        objectives = init_msg.get("objectives", []) or []

        parameter_names = [p.get("key") for p in parameters]
        objective_names = [o.get("key") for o in objectives]
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("Duplicate parameter keys in init message.")
        if len(set(objective_names)) != len(objective_names):
            raise ValueError("Duplicate objective keys in init message.")

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
        for i, (lo, hi, minflag) in enumerate(objectives_info):
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"Objective '{objective_names[i]}' bounds must be finite, got ({lo}, {hi})")
            if hi < lo:
                raise ValueError(f"Objective '{objective_names[i]}' has invalid bounds: low={lo} > high={hi}")
            if int(minflag) not in (0, 1):
                raise ValueError(f"Objective '{objective_names[i]}' minimize flag must be 0 or 1, got {minflag}")

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
