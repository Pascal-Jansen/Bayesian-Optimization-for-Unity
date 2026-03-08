# cabop_runtime.py — Unity NDJSON runtime for CABOP backend (single + scalarized multi-objective)
import csv
import json
import os
import socket
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cabop.bayesopt import BayesOpt, BOSpace

# -------------------- defaults (overwritten by Unity init) --------------------
N_INITIAL = 5
N_ITERATIONS = 10
SEED = 3

PROBLEM_DIM = None
NUM_OBJS = None

WARM_START = False
CSV_PATH_PARAMETERS = ""
CSV_PATH_OBJECTIVES = ""
WARM_START_OBJECTIVE_FORMAT = "auto"  # auto|raw|normalized_max|normalized_native

USER_ID = ""
CONDITION_ID = ""
GROUP_ID = ""
USER_LOG_ID = ""

OPTIMIZER_BACKEND = "cabop"
CABOP_MODE = "single"  # single|multi
CABOP_USE_COST_AWARE = True
CABOP_UPDATE_RULE = "actual"  # actual|intended|both
CABOP_ENABLE_COST_BUDGET = False
CABOP_MAX_CUMULATIVE_COST = -1.0

parameter_names: List[str] = []
objective_names: List[str] = []
parameters_info: List[Tuple[float, float, str, float, List[float]]] = []  # (lo, hi, group, tolerance, prefab)
objectives_info: List[Tuple[float, float, int, float]] = []  # (lo, hi, minimizeFlag, weight)

cabop_group_costs: Dict[str, Dict[str, Dict[str, float]]] = {}

# paths/state
PROJECT_PATH = ""
OBSERVATIONS_LOG_PATH = ""
EXECUTION_LOG_PATH = ""
METRICS_LOG_PATH = ""
COMPAT_METRIC_LOG_PATH = ""

# -------------------- TCP server helpers --------------------
HOST = ""
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
                raise RuntimeError(
                    f"Received malformed JSON line from Unity: {e}. Payload preview: {preview!r}"
                ) from e

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


def recv_objectives_blocking(conn):
    while True:
        msg = recv_json_message(conn)
        if msg is None:
            return None
        if not isinstance(msg, dict):
            raise RuntimeError(f"Received non-object message while waiting for objectives: {msg!r}")
        msg_type = msg.get("type")
        if msg_type == "objectives":
            values = msg.get("values")
            if not isinstance(values, dict):
                raise RuntimeError("Received malformed 'objectives' message: missing or non-dict 'values'.")
            return values
        raise RuntimeError(f"Received unexpected message type while waiting for objectives: {msg_type!r}")


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


def safe_float(value, fallback):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(f):
        return float(fallback)
    return float(f)


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


def normalize_update_rule(value):
    update_rule = str(value or "actual").strip().lower()
    if update_rule not in ("actual", "intended", "both"):
        raise ValueError(
            "cabopUpdateRule must be one of: actual, intended, both; "
            f"got '{update_rule}'"
        )
    return update_rule


def normalize_mode(value):
    mode = str(value or "single").strip().lower()
    if mode not in ("single", "multi"):
        raise ValueError(
            "cabopObjectiveMode must be one of: single, multi; "
            f"got '{mode}'"
        )
    return mode


def normalize_prefab_values(values):
    if not isinstance(values, (list, tuple)):
        return []
    out = []
    for raw in values:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        out.append(v)
    # deterministic order + de-duplication
    return sorted(set(out))


def normalize_cost_triplet(raw_triplet):
    if not isinstance(raw_triplet, dict):
        raw_triplet = {}

    unchanged = safe_float(raw_triplet.get("unchanged"), 1.0)
    swapped = safe_float(raw_triplet.get("swapped"), 10.0)
    acquired = safe_float(raw_triplet.get("acquired"), 100.0)

    if unchanged < 0:
        unchanged = 1.0
    if swapped < 0:
        swapped = 10.0
    if acquired < 0:
        acquired = 100.0

    return {
        "unchanged": float(unchanged),
        "swapped": float(swapped),
        "acquired": float(acquired),
    }


def init_parameter_and_objective_metadata(init_msg):
    global PROBLEM_DIM, NUM_OBJS
    global parameter_names, objective_names, parameters_info, objectives_info

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

    parameters_info = []
    for i, p in enumerate(parameters):
        lo, hi = parse_param_init(p.get("init"))
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Parameter '{parameter_names[i]}' bounds must be finite, got ({lo}, {hi})")
        if hi < lo:
            raise ValueError(f"Parameter '{parameter_names[i]}' has invalid bounds: low={lo} > high={hi}")

        group = str(p.get("group") or "default").strip() or "default"
        tol = safe_float(p.get("tolerance"), 0.05)
        if tol < 0:
            tol = 0.0
        prefab_values = normalize_prefab_values(p.get("prefabValues", []))

        parameters_info.append((float(lo), float(hi), group, float(tol), prefab_values))

    objectives_info = []
    for i, o in enumerate(objectives):
        lo, hi, minflag = parse_obj_init(o.get("init"))
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Objective '{objective_names[i]}' bounds must be finite, got ({lo}, {hi})")
        if hi < lo:
            raise ValueError(f"Objective '{objective_names[i]}' has invalid bounds: low={lo} > high={hi}")
        if int(minflag) not in (0, 1):
            raise ValueError(f"Objective '{objective_names[i]}' minimize flag must be 0 or 1, got {minflag}")

        weight = safe_float(o.get("weight"), 1.0)
        if weight <= 0:
            weight = 1.0
        objectives_info.append((float(lo), float(hi), int(minflag), float(weight)))


def init_group_costs(init_msg):
    global cabop_group_costs

    cabop_group_costs = {}
    group_cost_payload = init_msg.get("cabopGroupCosts", []) or []

    if isinstance(group_cost_payload, list):
        for entry in group_cost_payload:
            if not isinstance(entry, dict):
                continue
            group = str(entry.get("group") or "").strip()
            if not group:
                continue
            if group in cabop_group_costs:
                continue
            cabop_group_costs[group] = {
                "cost": normalize_cost_triplet(entry.get("cost")),
                "actual_cost": normalize_cost_triplet(entry.get("actualCost")),
            }

    groups_from_parameters = []
    for _, _, group, _, _ in parameters_info:
        if group not in groups_from_parameters:
            groups_from_parameters.append(group)

    for group in groups_from_parameters:
        if group in cabop_group_costs:
            continue
        cabop_group_costs[group] = {
            "cost": normalize_cost_triplet({}),
            "actual_cost": normalize_cost_triplet({}),
        }


def build_cabop_space_dict():
    groups = []
    params = {}

    for idx, name in enumerate(parameter_names):
        lo, hi, group, tol, _ = parameters_info[idx]
        if group not in groups:
            groups.append(group)
        params[name] = {
            "bound": np.asarray([lo, hi], dtype=float),
            "tolerance": float(tol),
            "group": group,
        }

    cost = {}
    actual_cost = {}
    for group in groups:
        group_data = cabop_group_costs.get(group, None)
        if group_data is None:
            group_data = {
                "cost": normalize_cost_triplet({}),
                "actual_cost": normalize_cost_triplet({}),
            }
        cost[group] = dict(group_data["cost"])
        actual_cost[group] = dict(group_data["actual_cost"])

    return {
        "groups": groups,
        "cost": cost,
        "actual_cost": actual_cost,
        "parameters": params,
    }


def build_prefab_dict():
    prefab = {}
    for idx, name in enumerate(parameter_names):
        values = parameters_info[idx][4]
        if values:
            prefab[name] = list(values)
    return prefab if prefab else None


def normalize_obj_column_to_raw(col, lo, hi, minflag):
    col = np.asarray(col, dtype=np.float64)
    raw_range_detected = np.all((lo - 1e-8 <= col) & (col <= hi + 1e-8))
    norm_range_detected = np.all((-1.0 - 1e-8 <= col) & (col <= 1.0 + 1e-8))

    mode = WARM_START_OBJECTIVE_FORMAT

    if mode == "raw":
        if not raw_range_detected:
            raise ValueError(
                f"warmStartObjectiveFormat=raw requires values in [{lo},{hi}], "
                f"but received range [{np.min(col)}, {np.max(col)}]"
            )
        return np.asarray(col, dtype=np.float64)

    if mode == "normalized_max":
        if not norm_range_detected:
            raise ValueError(
                "warmStartObjectiveFormat=normalized_max requires values in [-1,1], "
                f"but received range [{np.min(col)}, {np.max(col)}]"
            )
        y_max = np.clip(col, -1.0, 1.0)
        y_native = -y_max if int(minflag) == 1 else y_max
        if hi == lo:
            return np.full_like(y_native, lo)
        return lo + (y_native + 1.0) * 0.5 * (hi - lo)

    if mode == "normalized_native":
        if not norm_range_detected:
            raise ValueError(
                "warmStartObjectiveFormat=normalized_native requires values in [-1,1], "
                f"but received range [{np.min(col)}, {np.max(col)}]"
            )
        y_native = np.clip(col, -1.0, 1.0)
        if hi == lo:
            return np.full_like(y_native, lo)
        return lo + (y_native + 1.0) * 0.5 * (hi - lo)

    # auto
    if raw_range_detected:
        if norm_range_detected:
            print(
                "Warning: warm-start objective values are ambiguous (fit both raw bounds and [-1,1]); assuming raw scale.",
                flush=True,
            )
        return np.asarray(col, dtype=np.float64)

    if norm_range_detected:
        # auto mirrors previous behavior: normalized values are treated as normalized_max
        y_max = np.clip(col, -1.0, 1.0)
        y_native = -y_max if int(minflag) == 1 else y_max
        if hi == lo:
            return np.full_like(y_native, lo)
        return lo + (y_native + 1.0) * 0.5 * (hi - lo)

    raise ValueError(
        f"Warm-start objective values must be within raw bounds [{lo}, {hi}] or normalized [-1,1], "
        f"got range [{np.min(col)}, {np.max(col)}]"
    )


def normalize_param_column_to_raw(col, lo, hi):
    col = np.asarray(col, dtype=np.float64)
    eps = 1e-8
    in_raw_range = np.all((lo - eps <= col) & (col <= hi + eps))
    in_norm_range = np.all((-eps <= col) & (col <= 1.0 + eps))

    if hi == lo:
        if np.allclose(col, lo, rtol=0.0, atol=1e-8):
            return np.full_like(col, lo)
        if in_norm_range and np.allclose(col, 0.0, rtol=0.0, atol=1e-8):
            return np.full_like(col, lo)
        raise ValueError(f"Warm-start parameter values out of bounds for degenerate interval [{lo}, {hi}]")

    if in_raw_range:
        return np.asarray(col, dtype=np.float64)
    if in_norm_range:
        return lo + np.clip(col, 0.0, 1.0) * (hi - lo)

    raise ValueError(
        f"Warm-start parameter values must be within raw bounds [{lo}, {hi}] or normalized [0,1], "
        f"got range [{np.min(col)}, {np.max(col)}]"
    )


def load_warm_start_raw():
    cur = os.getcwd()
    if not CSV_PATH_PARAMETERS or not CSV_PATH_OBJECTIVES:
        raise ValueError("Warm start is enabled, but initial CSV paths are missing.")

    x_path = os.path.join(cur, "InitData", CSV_PATH_PARAMETERS)
    y_path = os.path.join(cur, "InitData", CSV_PATH_OBJECTIVES)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Warm-start parameter CSV not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Warm-start objective CSV not found: {y_path}")

    x_df = pd.read_csv(x_path, delimiter=";")
    y_df = pd.read_csv(y_path, delimiter=";")

    missing_param_cols = [k for k in parameter_names if k not in x_df.columns]
    missing_obj_cols = [k for k in objective_names if k not in y_df.columns]
    if missing_param_cols:
        raise ValueError(f"Warm-start parameter CSV is missing columns: {missing_param_cols}")
    if missing_obj_cols:
        raise ValueError(f"Warm-start objective CSV is missing columns: {missing_obj_cols}")

    x_in = x_df[parameter_names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    y_in = y_df[objective_names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)

    if x_in.shape[0] != y_in.shape[0]:
        raise ValueError(f"Warm-start rows mismatch: parameters={x_in.shape[0]}, objectives={y_in.shape[0]}")
    if x_in.shape[0] < 1:
        raise ValueError("Warm-start CSVs must contain at least one data row.")
    if not np.all(np.isfinite(x_in)):
        raise ValueError("Warm-start parameter CSV contains NaN/Inf values.")
    if not np.all(np.isfinite(y_in)):
        raise ValueError("Warm-start objective CSV contains NaN/Inf values.")

    x_raw = np.zeros_like(x_in, dtype=np.float64)
    for j in range(PROBLEM_DIM):
        lo, hi, _, _, _ = parameters_info[j]
        x_raw[:, j] = normalize_param_column_to_raw(x_in[:, j], lo, hi)

    y_raw = np.zeros_like(y_in, dtype=np.float64)
    for j in range(NUM_OBJS):
        lo, hi, minflag, _ = objectives_info[j]
        y_raw[:, j] = normalize_obj_column_to_raw(y_in[:, j], lo, hi, minflag)

    if not np.all(np.isfinite(x_raw)):
        raise ValueError("Warm-start normalized parameters contain non-finite values.")
    if not np.all(np.isfinite(y_raw)):
        raise ValueError("Warm-start normalized objectives contain non-finite values.")

    return x_raw, y_raw


def objective_to_minimize_unit(value, lo, hi, minflag):
    eps = 1e-9
    if hi == lo:
        if not np.isclose(value, lo, rtol=0.0, atol=eps):
            raise ValueError(f"Objective value {value} is out of bounds for degenerate interval [{lo}, {hi}]")
        return 0.0

    if value < lo - eps or value > hi + eps:
        raise ValueError(f"Objective value {value} is out of bounds [{lo}, {hi}]")

    if int(minflag) == 1:
        unit = (value - lo) / (hi - lo)
    else:
        unit = (hi - value) / (hi - lo)

    return float(np.clip(unit, 0.0, 1.0))


def scalarize_objectives(raw_values):
    if len(raw_values) != len(objectives_info):
        raise ValueError("Objective value count mismatch while scalarizing.")

    normalized_min = []
    weights = []
    for j, raw in enumerate(raw_values):
        lo, hi, minflag, weight = objectives_info[j]
        normalized_min.append(objective_to_minimize_unit(float(raw), lo, hi, minflag))
        weights.append(max(float(weight), 1e-9))

    if CABOP_MODE == "single":
        return float(normalized_min[0])

    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return float(np.mean(normalized_min))
    return float(np.average(np.asarray(normalized_min, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))


def expected_observation_columns():
    marker_col = "IsBest" if CABOP_MODE == "single" else "IsPareto"
    return [
        "UserID",
        "ConditionID",
        "GroupID",
        "Timestamp",
        "Iteration",
        "Phase",
        marker_col,
    ] + objective_names + parameter_names


def create_observations_file_if_missing(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter=";").writerow(expected_observation_columns())


def append_execution_time(iteration, elapsed_sec):
    write_header = not os.path.exists(EXECUTION_LOG_PATH) or os.path.getsize(EXECUTION_LOG_PATH) == 0
    with open(EXECUTION_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if write_header:
            w.writerow(["Optimization", "Execution_Time"])
        w.writerow([iteration, elapsed_sec])


def append_observation_row(iteration, phase, scalarized_value, objective_raw, parameter_raw):
    create_observations_file_if_missing(OBSERVATIONS_LOG_PATH)

    marker_col = "IsBest" if CABOP_MODE == "single" else "IsPareto"
    row = {
        "UserID": USER_ID,
        "ConditionID": CONDITION_ID,
        "GroupID": GROUP_ID,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "Iteration": int(iteration),
        "Phase": phase,
        marker_col: "FALSE",
    }

    for j, name in enumerate(objective_names):
        row[name] = np.round(float(objective_raw[j]), 3)
    for i, name in enumerate(parameter_names):
        row[name] = np.round(float(parameter_raw[i]), 3)

    df = pd.read_csv(OBSERVATIONS_LOG_PATH, delimiter=";") if os.path.exists(OBSERVATIONS_LOG_PATH) else pd.DataFrame(columns=expected_observation_columns())
    expected_cols = expected_observation_columns()
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"ObservationsPerEvaluation.csv columns mismatch. Expected {expected_cols}, got {list(df.columns)}"
        )

    new_row_df = pd.DataFrame([[row[c] for c in df.columns]], columns=df.columns)
    if df.empty:
        df = new_row_df
    else:
        df = pd.concat([df, new_row_df], ignore_index=True)

    # Mark the currently best scalarized (lowest) row(s) as candidates.
    values = []
    for _, r in df.iterrows():
        raw_vals = [float(r[name]) for name in objective_names]
        values.append(scalarize_objectives(raw_vals))
    values = np.asarray(values, dtype=float)

    best = float(np.min(values)) if len(values) > 0 else float(scalarized_value)
    flags = ["TRUE" if abs(v - best) <= 1e-12 else "FALSE" for v in values]
    df[marker_col] = flags

    df.to_csv(OBSERVATIONS_LOG_PATH, sep=";", index=False)


def append_metrics_row(iteration, phase, scalarized_value, best_scalarized, coverage, realized_cost, cumulative_cost):
    write_header = not os.path.exists(METRICS_LOG_PATH) or os.path.getsize(METRICS_LOG_PATH) == 0
    with open(METRICS_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if write_header:
            w.writerow([
                "Iteration",
                "Phase",
                "ScalarizedObjective",
                "BestScalarizedObjective",
                "Coverage",
                "EvaluationCost",
                "CumulativeCost",
            ])
        w.writerow([
            int(iteration),
            phase,
            np.round(float(scalarized_value), 6),
            np.round(float(best_scalarized), 6),
            np.round(float(coverage), 6),
            np.round(float(realized_cost), 6),
            np.round(float(cumulative_cost), 6),
        ])


def append_compat_metric(iteration, coverage):
    write_header = not os.path.exists(COMPAT_METRIC_LOG_PATH) or os.path.getsize(COMPAT_METRIC_LOG_PATH) == 0
    with open(COMPAT_METRIC_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if write_header:
            if CABOP_MODE == "single":
                w.writerow(["BestObjective", "Run"])
            else:
                w.writerow(["Hypervolume", "Run"])
        w.writerow([np.round(float(coverage), 6), int(iteration)])


def evaluate_design(conn, x_realized):
    values = {}
    for i, name in enumerate(parameter_names):
        values[name] = float(x_realized[i])

    payload = {"type": "parameters", "values": values}
    send_json_line(conn, payload)

    resp = recv_objectives_blocking(conn)
    if resp is None:
        raise RuntimeError("No objectives received from Unity.")
    if not isinstance(resp, dict):
        raise TypeError(f"Unity objectives payload must be a dict, got {type(resp).__name__}")

    missing = [k for k in objective_names if k not in resp]
    if missing:
        raise KeyError(f"Unity objectives missing required key(s): {missing}")

    unexpected = sorted([k for k in resp.keys() if k not in set(objective_names)])
    if unexpected:
        raise KeyError(f"Unity objectives payload contains unexpected key(s): {unexpected}")

    raw_values = []
    for j, name in enumerate(objective_names):
        try:
            val = float(resp[name])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Objective '{name}' must be numeric, got {resp[name]!r}") from e
        if not np.isfinite(val):
            raise ValueError(f"Objective '{name}' is non-finite: {val}")

        lo, hi, _, _ = objectives_info[j]
        eps = 1e-9
        if hi == lo:
            if not np.isclose(val, lo, rtol=0.0, atol=eps):
                raise ValueError(f"Objective '{name}' value {val} is out of bounds for degenerate interval [{lo}, {hi}]")
        elif val < lo - eps or val > hi + eps:
            raise ValueError(f"Objective '{name}' value {val} is out of bounds [{lo}, {hi}]")

        raw_values.append(val)

    scalarized = scalarize_objectives(raw_values)
    return float(scalarized), raw_values


def boot_optimizer_with_warm_start(optimizer):
    if not WARM_START:
        return

    x_raw, y_raw = load_warm_start_raw()
    for i in range(x_raw.shape[0]):
        x = x_raw[i]
        y_scalar = scalarize_objectives(y_raw[i].tolist())
        optimizer.tell(x, float(y_scalar), x, update_rule="actual")


def run_cabop(conn):
    global PROJECT_PATH, OBSERVATIONS_LOG_PATH, EXECUTION_LOG_PATH, METRICS_LOG_PATH, COMPAT_METRIC_LOG_PATH

    base = os.path.join(os.getcwd(), "LogData", "CABOP", CABOP_MODE)
    os.makedirs(base, exist_ok=True)
    PROJECT_PATH = get_unique_folder(base, USER_LOG_ID)

    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    EXECUTION_LOG_PATH = os.path.join(PROJECT_PATH, "ExecutionTimes.csv")
    METRICS_LOG_PATH = os.path.join(PROJECT_PATH, "CABOPMetricsPerEvaluation.csv")
    COMPAT_METRIC_LOG_PATH = os.path.join(
        PROJECT_PATH,
        "BestObjectivePerEvaluation.csv" if CABOP_MODE == "single" else "HypervolumePerEvaluation.csv",
    )

    create_observations_file_if_missing(OBSERVATIONS_LOG_PATH)

    space_dict = build_cabop_space_dict()
    space = BOSpace(parameters=space_dict)
    optimizer = BayesOpt(space, ifCost=bool(CABOP_USE_COST_AWARE), random_state=SEED)

    prefab = build_prefab_dict()

    # Seed optimizer from warm-start CSV data.
    boot_optimizer_with_warm_start(optimizer)

    planned_evaluations = int(N_ITERATIONS if WARM_START else (N_INITIAL + N_ITERATIONS))
    planned_evaluations = max(0, planned_evaluations)

    cumulative_cost = 0.0
    best_scalarized = float(optimizer.current_best.get("y", np.inf))
    if not np.isfinite(best_scalarized):
        best_scalarized = np.inf

    iteration = 0
    while iteration < planned_evaluations:
        if CABOP_ENABLE_COST_BUDGET and CABOP_MAX_CUMULATIVE_COST > 0 and cumulative_cost >= CABOP_MAX_CUMULATIVE_COST:
            print(
                f"CABOP cost budget reached before iteration {iteration + 1}: "
                f"cumulative_cost={cumulative_cost}, limit={CABOP_MAX_CUMULATIVE_COST}",
                flush=True,
            )
            break

        t0 = time.time()
        x_candidate, _ = optimizer.ask(n_init=max(0, int(N_INITIAL)))
        elapsed = time.time() - t0

        costs, x_realized = optimizer.select_sample(x_candidate, prefab=prefab)
        scalarized, objective_raw = evaluate_design(conn, x_realized)

        optimizer.tell(x_realized, float(scalarized), x_candidate, update_rule=CABOP_UPDATE_RULE)

        realized_cost = float(np.sum(costs))
        cumulative_cost += realized_cost
        best_scalarized = min(best_scalarized, float(scalarized))

        # 1.0 means best possible according to scalarized minimization objective.
        coverage = float(np.clip(1.0 - best_scalarized, -1e9, 1.0)) if np.isfinite(best_scalarized) else 0.0

        absolute_iteration = iteration + 1
        phase = "sampling" if (not WARM_START and absolute_iteration <= N_INITIAL) else "optimization"

        append_execution_time(absolute_iteration, elapsed)
        append_observation_row(absolute_iteration, phase, scalarized, objective_raw, x_realized)
        append_metrics_row(absolute_iteration, phase, scalarized, best_scalarized, coverage, realized_cost, cumulative_cost)
        append_compat_metric(absolute_iteration, coverage)

        send_json_line(conn, {"type": "coverage", "value": float(coverage)})
        send_json_line(
            conn,
            {
                "type": "tempCoverage",
                "value": float(absolute_iteration) / float(max(1, planned_evaluations)),
            },
        )

        iteration += 1

    send_json_line(conn, {"type": "optimization_finished"})


def parse_init_and_validate(init_msg, forced_mode):
    global N_INITIAL, N_ITERATIONS, SEED, PROBLEM_DIM, NUM_OBJS
    global WARM_START, CSV_PATH_PARAMETERS, CSV_PATH_OBJECTIVES, WARM_START_OBJECTIVE_FORMAT
    global USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID
    global OPTIMIZER_BACKEND, CABOP_MODE, CABOP_USE_COST_AWARE
    global CABOP_UPDATE_RULE, CABOP_ENABLE_COST_BUDGET, CABOP_MAX_CUMULATIVE_COST

    cfg = init_msg.get("config", {}) or {}

    N_INITIAL = get_cfg_int(cfg, "numSamplingIterations", default=N_INITIAL)
    N_ITERATIONS = get_cfg_int(cfg, "numOptimizationIterations", default=N_ITERATIONS)
    SEED = get_cfg_int(cfg, "seed", default=SEED)
    PROBLEM_DIM = get_cfg_int(cfg, "nParameters", required=True)
    NUM_OBJS = get_cfg_int(cfg, "nObjectives", required=True)

    WARM_START = bool(cfg.get("warmStart", False))
    CSV_PATH_PARAMETERS = str(cfg.get("initialParametersDataPath") or "")
    CSV_PATH_OBJECTIVES = str(cfg.get("initialObjectivesDataPath") or "")
    WARM_START_OBJECTIVE_FORMAT = str(cfg.get("warmStartObjectiveFormat", "auto") or "auto").strip().lower()

    OPTIMIZER_BACKEND = str(cfg.get("optimizerBackend") or "cabop").strip().lower()
    CABOP_MODE = normalize_mode(forced_mode if forced_mode else cfg.get("cabopObjectiveMode", "single"))
    CABOP_USE_COST_AWARE = bool(cfg.get("cabopUseCostAwareAcquisition", True))
    CABOP_UPDATE_RULE = normalize_update_rule(cfg.get("cabopUpdateRule", "actual"))
    CABOP_ENABLE_COST_BUDGET = bool(cfg.get("cabopEnableCostBudget", False))
    CABOP_MAX_CUMULATIVE_COST = safe_float(cfg.get("cabopMaxCumulativeCost"), -1.0)

    if OPTIMIZER_BACKEND not in ("cabop", "botorch"):
        raise ValueError(f"optimizerBackend must be 'cabop' or 'botorch', got '{OPTIMIZER_BACKEND}'")

    if PROBLEM_DIM < 1:
        raise ValueError(f"nParameters must be >= 1, got {PROBLEM_DIM}")
    if NUM_OBJS < 1:
        raise ValueError(f"nObjectives must be >= 1, got {NUM_OBJS}")
    if N_INITIAL < 0 or N_ITERATIONS < 0:
        raise ValueError(f"Iteration counts must be non-negative, got sampling={N_INITIAL}, optimization={N_ITERATIONS}")
    if (not WARM_START) and N_INITIAL < 1:
        raise ValueError(
            "numSamplingIterations must be >= 1 when warmStart is disabled. "
            f"Got sampling={N_INITIAL}, warmStart={WARM_START}."
        )

    if WARM_START_OBJECTIVE_FORMAT not in ("auto", "raw", "normalized_max", "normalized_native"):
        raise ValueError(
            "warmStartObjectiveFormat must be one of: auto, raw, normalized_max, normalized_native; "
            f"got '{WARM_START_OBJECTIVE_FORMAT}'"
        )

    user = init_msg.get("user", {}) or {}
    USER_ID = normalize_user_token(user.get("userId"), default="-1")
    CONDITION_ID = normalize_user_token(user.get("conditionId"), default="-1")
    GROUP_ID = normalize_user_token(user.get("groupId"), default="-1")
    USER_LOG_ID = normalize_log_folder_token(USER_ID, default="-1")

    init_parameter_and_objective_metadata(init_msg)
    init_group_costs(init_msg)

    if CABOP_MODE == "single" and NUM_OBJS != 1:
        raise ValueError(f"CABOP single mode requires exactly one objective, got {NUM_OBJS}")
    if CABOP_MODE == "multi" and NUM_OBJS < 2:
        raise ValueError(f"CABOP multi mode requires at least two objectives, got {NUM_OBJS}")


def main(forced_mode=None):
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
        print("Server starts, waiting for connection...", flush=True)
        try:
            conn, addr = s.accept()
        except socket.timeout as e:
            raise TimeoutError(f"Socket accept timed out after {SOCKET_ACCEPT_TIMEOUT_SEC} seconds.") from e

        print("Connected by", addr, flush=True)
        if SOCKET_TIMEOUT_SEC <= 0:
            raise ValueError(f"BO_SOCKET_TIMEOUT_SEC must be > 0, got {SOCKET_TIMEOUT_SEC}")
        conn.settimeout(SOCKET_TIMEOUT_SEC)
        SOCKET_RECV_BUF = ""

        init_msg = None
        while True:
            msg = recv_json_message(conn)
            if msg is None:
                break
            if not isinstance(msg, dict):
                raise RuntimeError(f"Received non-object message while waiting for init: {msg!r}")
            if msg.get("type") == "init":
                init_msg = msg
                break
            raise RuntimeError(
                f"Received unexpected message type while waiting for init: {msg.get('type')!r}"
            )

        if init_msg is None:
            raise RuntimeError("Did not receive init message.")

        parse_init_and_validate(init_msg, forced_mode=forced_mode)

        print(
            "Init OK:",
            dict(
                mode=CABOP_MODE,
                updateRule=CABOP_UPDATE_RULE,
                costAware=CABOP_USE_COST_AWARE,
                samplingIterations=N_INITIAL,
                optimizationIterations=N_ITERATIONS,
                warmStart=WARM_START,
                nParameters=PROBLEM_DIM,
                nObjectives=NUM_OBJS,
            ),
            flush=True,
        )

        run_cabop(conn)
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
