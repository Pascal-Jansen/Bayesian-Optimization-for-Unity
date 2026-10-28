# cabop_runtime.py — Unity NDJSON runtime for CABOP backend (single + scalarized multi-objective)
import os
import socket
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import bo_normalize
# Socket, NDJSON, init parsing and log-writing plumbing shared by every backend.
from bo_protocol import (
    SOCKET_ACCEPT_TIMEOUT_SEC, SOCKET_TIMEOUT_SEC, StopRequested,
    accept_unity_connection, append_csv_rows, close_connection, configure_listener_socket,
    flush_pending_logs, get_cfg_int, get_unique_folder, log_exists, parse_obj_init,
    parse_param_init, parse_user_ids, read_observation_log, receive_init_message,
    recv_objectives_blocking, send_json_line, validate_objective_bounds,
    validate_parameter_bounds, write_csv_rows, write_dataframe_csv,
)
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
CONDITION_LOG_ID = ""

OPTIMIZER_BACKEND = "cabop"
CABOP_MODE = "single"  # single|multi
CABOP_USE_COST_AWARE = True
CABOP_UPDATE_RULE = "actual"  # actual|intended|both
CABOP_ENABLE_COST_BUDGET = False
CABOP_MAX_CUMULATIVE_COST = -1.0

parameter_names: List[str] = []
objective_names: List[str] = []
# tolerance is a fraction of the parameter's range (0.05 = 5% of hi - lo)
parameters_info: List[Tuple[float, float, str, float, List[float]]] = []  # (lo, hi, group, tolerance, prefab)
objectives_info: List[Tuple[float, float, int, float]] = []  # (lo, hi, minimizeFlag, weight)

DEFAULT_CABOP_TOLERANCE = 0.05

cabop_group_costs: Dict[str, Dict[str, Dict[str, float]]] = {}

# paths/state
PROJECT_PATH = ""
OBSERVATIONS_LOG_PATH = ""
EXECUTION_LOG_PATH = ""
METRICS_LOG_PATH = ""
COMPAT_METRIC_LOG_PATH = ""

# Every observation handed to the optimizer, at full precision: the warm-start rows first
# (N_WARM_START_ROWS of them), then this run's evaluations -- the rows that reach the CSV.
# IsBest/IsPareto compete over all of them, as in bo.py/mobo.py, and are not derived from the
# rounded values written to the CSV.
SCALARIZED_HISTORY: List[float] = []
OBJECTIVE_HISTORY: List[List[float]] = []  # raw objective values, Unity's units
N_WARM_START_ROWS = 0


# -------------------- init parsing --------------------
def safe_float(value, fallback):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(f):
        return float(fallback)
    return float(f)


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

    # Keys become CSV columns next to the fixed ones; a key such as "Phase" would duplicate
    # a column and break the log rewrite after the first evaluation.
    reserved = sorted(
        set(parameter_names + objective_names).intersection(
            {"UserID", "ConditionID", "GroupID", "Timestamp", "Iteration", "Phase", "IsBest", "IsPareto"}
        )
    )
    if reserved:
        raise ValueError(f"Parameter/objective keys collide with log columns: {reserved}. Rename them.")

    if len(parameter_names) != PROBLEM_DIM:
        raise ValueError(f"parameter_names len {len(parameter_names)} != nParameters {PROBLEM_DIM}")
    if len(objective_names) != NUM_OBJS:
        raise ValueError(f"objective_names len {len(objective_names)} != nObjectives {NUM_OBJS}")

    bounds = [parse_param_init(p.get("init")) for p in parameters]
    # CABOP maps parameters to [0,1] by (x - lo) / (hi - lo); a frozen parameter would turn
    # the first GP fit, after the sampling phase, into NaN.
    validate_parameter_bounds(parameter_names, bounds, "CABOP")

    parameters_info = []
    # Unity matches CABOP groups case-insensitively; collapse spellings onto the first seen
    # so a group and its configured costs cannot drift apart over letter case.
    group_spellings = {}
    for name, p, (lo, hi) in zip(parameter_names, parameters, bounds):
        group = str(p.get("group") or "default").strip() or "default"
        group = group_spellings.setdefault(group.casefold(), group)
        # The reuse tolerance is a fraction of the parameter's range, so it means the same for
        # a parameter on [0, 0.1] and one on [0, 100]. A value outside [0, 1] cannot be such a
        # fraction -- most likely a tolerance in parameter units from an older configuration.
        raw_tol = p.get("tolerance")
        tol = DEFAULT_CABOP_TOLERANCE if raw_tol is None else safe_float(raw_tol, float("nan"))
        if not 0.0 <= tol <= 1.0:
            raise ValueError(
                f"CABOP tolerance of parameter '{name}' must lie in [0, 1] (a fraction of its range "
                f"[{lo}, {hi}]), got {raw_tol!r}. To keep a tolerance given in parameter units, "
                f"divide it by the range ({hi - lo})."
            )
        prefab_values = normalize_prefab_values(p.get("prefabValues", []))
        # Proposals snap to the nearest prefabricated value, and Unity applies what it receives:
        # a value outside the bounds would be shown to the participant as is.
        outside = [v for v in prefab_values if v < lo - 1e-9 or v > hi + 1e-9]
        if outside:
            raise ValueError(
                f"CABOP prefabricated value(s) {outside} of parameter '{name}' lie outside its bounds "
                f"[{lo}, {hi}]; designs snapped to them would leave the configured range."
            )

        parameters_info.append((float(lo), float(hi), group, float(tol), prefab_values))

    objective_inits = [parse_obj_init(o.get("init")) for o in objectives]
    validate_objective_bounds(objective_names, objective_inits)

    objectives_info = []
    for o, (lo, hi, minflag) in zip(objectives, objective_inits):
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
            if group.casefold() in cabop_group_costs:
                continue
            cabop_group_costs[group.casefold()] = {
                "cost": normalize_cost_triplet(entry.get("cost")),
                "actual_cost": normalize_cost_triplet(entry.get("actualCost")),
            }

    groups_from_parameters = []
    for _, _, group, _, _ in parameters_info:
        if group not in groups_from_parameters:
            groups_from_parameters.append(group)

    for group in groups_from_parameters:
        if group.casefold() in cabop_group_costs:
            continue
        cabop_group_costs[group.casefold()] = {
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
        group_data = cabop_group_costs.get(group.casefold(), None)
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
    # Same scale inference as the BoTorch/DBO backends (raw vs already-normalized, the shared rounding
    # tolerance, announced fallbacks), mapped back to the raw units CABOP works in. The former copy
    # here accepted raw values only within 1e-8 of the bounds, so 3-decimal logs on small ranges were
    # silently re-read as normalized.
    y_max = bo_normalize.normalize_obj_column(col, lo, hi, minflag, fmt=WARM_START_OBJECTIVE_FORMAT)
    y_native = -y_max if int(minflag) == 1 else y_max
    if hi == lo:
        return np.full_like(y_native, lo)
    return lo + (y_native + 1.0) * 0.5 * (hi - lo)


def normalize_param_column_to_raw(col, lo, hi):
    # Same scale inference as the BoTorch/DBO backends (raw vs already-normalized, rounding
    # tolerance, warnings), mapped back to the raw units CABOP works in.
    unit = bo_normalize.normalize_param_column(col, lo, hi)
    return lo + unit * (hi - lo)


def load_warm_start_raw():
    if not CSV_PATH_PARAMETERS or not CSV_PATH_OBJECTIVES:
        raise ValueError("Warm start is enabled, but initial CSV paths are missing.")

    init_root = os.environ.get("BO_INIT_ROOT") or os.path.join(os.getcwd(), "InitData")
    x_path = os.path.join(init_root, CSV_PATH_PARAMETERS)
    y_path = os.path.join(init_root, CSV_PATH_OBJECTIVES)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Warm-start parameter CSV not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Warm-start objective CSV not found: {y_path}")

    x_df = pd.read_csv(x_path, delimiter=";", encoding="utf-8")
    y_df = pd.read_csv(y_path, delimiter=";", encoding="utf-8")

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
    if log_exists(path):
        return
    write_csv_rows(path, expected_observation_columns(), [])


def append_execution_time(iteration, elapsed_sec):
    append_csv_rows(EXECUTION_LOG_PATH, [[iteration, elapsed_sec]], header=["Optimization", "Execution_Time"])


def record_history(scalarized_value, objective_raw):
    """Remember one observation (warm-start or live) for the IsBest/IsPareto flags."""
    SCALARIZED_HISTORY.append(float(scalarized_value))
    OBJECTIVE_HISTORY.append([float(v) for v in objective_raw])


def non_dominated_mask(values_max):
    """Rows of a maximization matrix that no other row dominates.

    Of identical rows only the first counts, as in mobo.py/MetaTAF (moocore's
    is_nondominated with keep_weakly=True, then first copy of each duplicate only).
    """
    arr = np.asarray(values_max, dtype=np.float64)
    keep = np.zeros(arr.shape[0], dtype=bool)
    seen = set()
    for i, row in enumerate(arr):
        key = tuple(row.tolist())
        if key in seen:
            continue
        seen.add(key)
        dominated_by = np.all(arr >= row, axis=1) & np.any(arr > row, axis=1)
        keep[i] = not np.any(dominated_by)
    return keep


def marker_flags():
    """IsBest (single) / IsPareto (multi) flag of every observation in the history.

    Warm-start rows compete like in bo.py/mobo.py: a run whose best (or a dominating) design
    came from the warm-start data marks no live row for it. In multi mode IsPareto is the real
    non-dominated set of the raw objective vectors in each objective's direction -- the
    candidates FinalDesignSelector chooses from -- not the rows with the lowest weighted score.
    """
    if not SCALARIZED_HISTORY:
        return []
    if CABOP_MODE == "single":
        best = min(SCALARIZED_HISTORY)
        flags = [abs(v - best) <= 1e-12 for v in SCALARIZED_HISTORY]
    else:
        signs = np.array([1.0 if int(minflag) == 1 else -1.0 for _, _, minflag, _ in objectives_info])
        # Minimized objectives flip sign, so larger is better in every column.
        flags = non_dominated_mask(-signs * np.asarray(OBJECTIVE_HISTORY, dtype=np.float64)).tolist()
    return ["TRUE" if f else "FALSE" for f in flags]


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

    # 10 significant digits, as every other backend logs (3 decimals erased small ranges).
    for j, name in enumerate(objective_names):
        row[name] = bo_normalize.round_for_log(objective_raw[j])
    for i, name in enumerate(parameter_names):
        row[name] = bo_normalize.round_for_log(parameter_raw[i])

    # Read back as text: type inference would rewrite IDs such as "007" as 7 in every
    # earlier row, and FinalDesignSelector matches IDs as exact strings.
    df = (
        read_observation_log(OBSERVATIONS_LOG_PATH)
        if log_exists(OBSERVATIONS_LOG_PATH)
        else pd.DataFrame(columns=expected_observation_columns())
    )
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

    # Flags from the full-precision in-memory history (warm-start rows included), assigned to
    # this run's rows at the end of the log, as bo.py/mobo.py do.
    record_history(scalarized_value, objective_raw)
    flags = marker_flags()[N_WARM_START_ROWS:]
    df[marker_col] = df[marker_col].astype(str)
    if len(flags) >= len(df):
        df[marker_col] = flags[-len(df):]
    elif flags:
        df.loc[df.index[-len(flags):], marker_col] = flags

    write_dataframe_csv(OBSERVATIONS_LOG_PATH, df)


def append_metrics_row(iteration, phase, scalarized_value, best_scalarized, coverage, realized_cost, cumulative_cost):
    append_csv_rows(
        METRICS_LOG_PATH,
        [[
            int(iteration),
            phase,
            np.round(float(scalarized_value), 6),
            np.round(float(best_scalarized), 6),
            np.round(float(coverage), 6),
            np.round(float(realized_cost), 6),
            np.round(float(cumulative_cost), 6),
        ]],
        header=[
            "Iteration",
            "Phase",
            "ScalarizedObjective",
            "BestScalarizedObjective",
            "Coverage",
            "EvaluationCost",
            "CumulativeCost",
        ],
    )


def append_compat_metric(iteration, coverage):
    append_csv_rows(
        COMPAT_METRIC_LOG_PATH,
        [[np.round(float(coverage), 6), int(iteration)]],
        header=["BestObjective" if CABOP_MODE == "single" else "Hypervolume", "Iteration"],
    )


def evaluate_design(conn, x_realized, iteration):
    """Send one realized design to Unity (logged under ``iteration``) and block for its objectives."""
    values = {}
    for i, name in enumerate(parameter_names):
        values[name] = float(x_realized[i])

    payload = {"type": "parameters", "values": values, "iteration": int(iteration)}
    send_json_line(conn, payload)

    resp = recv_objectives_blocking(conn)  # a dict, or None once Unity disconnected
    if resp is None:
        raise RuntimeError("No objectives received from Unity.")

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


def coverage_from(best_scalarized):
    """Unity's coverage: 1 - best scalarized objective (1.0 = best possible)."""
    return float(np.clip(1.0 - best_scalarized, -1e9, 1.0)) if np.isfinite(best_scalarized) else 0.0


def boot_optimizer_with_warm_start(optimizer):
    """Tell the optimizer the warm-start rows; returns how many there were (0 without warm start).

    The rows are earlier evaluations of the same problem, as in bo.py/DBO: they are not written
    to ObservationsPerEvaluation.csv, but they occupy Iterations 1..k, so this run's first
    evaluation is Iteration k + 1, and they compete for IsBest/IsPareto.
    """
    if not WARM_START:
        return 0

    x_raw, y_raw = load_warm_start_raw()
    for i in range(x_raw.shape[0]):
        x = x_raw[i]
        y_scalar = scalarize_objectives(y_raw[i].tolist())
        optimizer.tell(x, float(y_scalar), x, update_rule="actual")
        record_history(y_scalar, y_raw[i].tolist())
    print(f"Warm start: {x_raw.shape[0]} prior observation(s) loaded as Iterations 1..{x_raw.shape[0]}; "
          f"skipping the sampling phase.", flush=True)
    return int(x_raw.shape[0])


def run_cabop(conn):
    global PROJECT_PATH, OBSERVATIONS_LOG_PATH, EXECUTION_LOG_PATH, METRICS_LOG_PATH, COMPAT_METRIC_LOG_PATH
    global SCALARIZED_HISTORY, OBJECTIVE_HISTORY, N_WARM_START_ROWS

    SCALARIZED_HISTORY = []
    OBJECTIVE_HISTORY = []
    N_WARM_START_ROWS = 0
    log_root = os.environ.get("BO_LOG_ROOT") or os.path.join(os.getcwd(), "LogData")
    base = os.path.join(log_root, USER_LOG_ID, CONDITION_LOG_ID, "CABOP", CABOP_MODE)
    os.makedirs(base, exist_ok=True)
    PROJECT_PATH = get_unique_folder(base, "run")

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
    # numpy rejects negative seeds, which Unity allows; wrap into numpy's seed range.
    optimizer = BayesOpt(space, ifCost=bool(CABOP_USE_COST_AWARE), random_state=SEED % (2**32))

    prefab = build_prefab_dict()

    # Seed optimizer from warm-start CSV data. Warm start replaces the sampling phase, as in
    # bo.py/DBO: the model takes over at the first evaluation (n_init = 0), instead of drawing
    # N_INITIAL - k Sobol points from the optimization budget.
    N_WARM_START_ROWS = boot_optimizer_with_warm_start(optimizer)
    n_init = 0 if WARM_START else max(0, int(N_INITIAL))

    planned_evaluations = int(N_ITERATIONS if WARM_START else (N_INITIAL + N_ITERATIONS))
    planned_evaluations = max(0, planned_evaluations)

    cumulative_cost = 0.0
    best_scalarized = float(optimizer.current_best.get("y", np.inf))
    if not np.isfinite(best_scalarized):
        best_scalarized = np.inf

    if WARM_START:
        # Baseline over the warm-start rows at the Iteration of the last of them, as bo.py
        # writes it (CABOPMetricsPerEvaluation.csv keeps one row per evaluation of this run).
        append_compat_metric(N_WARM_START_ROWS, coverage_from(best_scalarized))
        send_json_line(conn, {"type": "coverage", "value": coverage_from(best_scalarized)})

    iteration = 0
    while iteration < planned_evaluations:
        # Iteration this design is logged under: warm-start rows occupy 1..k.
        absolute_iteration = N_WARM_START_ROWS + iteration + 1
        if CABOP_ENABLE_COST_BUDGET and CABOP_MAX_CUMULATIVE_COST > 0 and cumulative_cost >= CABOP_MAX_CUMULATIVE_COST:
            print(
                f"CABOP cost budget reached before iteration {absolute_iteration}: "
                f"cumulative_cost={cumulative_cost}, limit={CABOP_MAX_CUMULATIVE_COST}",
                flush=True,
            )
            break

        t0 = time.time()
        x_candidate, _ = optimizer.ask(n_init=n_init)
        elapsed = time.time() - t0

        costs, x_realized = optimizer.select_sample(x_candidate, prefab=prefab)
        scalarized, objective_raw = evaluate_design(conn, x_realized, absolute_iteration)

        optimizer.tell(x_realized, float(scalarized), x_candidate, update_rule=CABOP_UPDATE_RULE)

        realized_cost = float(np.sum(costs))
        cumulative_cost += realized_cost
        best_scalarized = min(best_scalarized, float(scalarized))

        # 1.0 means best possible according to scalarized minimization objective.
        coverage = coverage_from(best_scalarized)

        phase = "sampling" if (not WARM_START and iteration < N_INITIAL) else "optimization"

        append_execution_time(absolute_iteration, elapsed)
        append_observation_row(absolute_iteration, phase, scalarized, objective_raw, x_realized)
        append_metrics_row(absolute_iteration, phase, scalarized, best_scalarized, coverage, realized_cost, cumulative_cost)
        append_compat_metric(absolute_iteration, coverage)

        send_json_line(conn, {"type": "coverage", "value": float(coverage)})
        send_json_line(
            conn,
            {
                "type": "tempCoverage",
                "value": float(iteration + 1) / float(max(1, planned_evaluations)),
            },
        )

        iteration += 1

    send_json_line(conn, {"type": "optimization_finished"})


def parse_init_and_validate(init_msg, forced_mode):
    global N_INITIAL, N_ITERATIONS, SEED, PROBLEM_DIM, NUM_OBJS
    global WARM_START, CSV_PATH_PARAMETERS, CSV_PATH_OBJECTIVES, WARM_START_OBJECTIVE_FORMAT
    global USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID
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

    USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID = parse_user_ids(init_msg)

    init_parameter_and_objective_metadata(init_msg)
    init_group_costs(init_msg)

    if CABOP_MODE == "single" and NUM_OBJS != 1:
        raise ValueError(f"CABOP single mode requires exactly one objective, got {NUM_OBJS}")
    if CABOP_MODE == "multi" and NUM_OBJS < 2:
        raise ValueError(f"CABOP multi mode requires at least two objectives, got {NUM_OBJS}")


def main(forced_mode=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    configure_listener_socket(s)
    conn = None
    try:
        conn = accept_unity_connection(s, SOCKET_ACCEPT_TIMEOUT_SEC)
        init_msg = receive_init_message(conn, SOCKET_TIMEOUT_SEC)

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
        # The tolerance is a fraction of each range; show what it means in parameter units.
        print(
            "CABOP reuse tolerance (fraction of range = +/- parameter units):",
            {name: f"{tol:g} = +/-{tol * (hi - lo):g}"
             for name, (lo, hi, _, tol, _) in zip(parameter_names, parameters_info)},
            flush=True,
        )

        run_cabop(conn)
    except StopRequested:
        pass  # Unity ended the study on purpose; every completed evaluation is logged
    finally:
        close_connection(conn)
        s.close()
        flush_pending_logs()


if __name__ == "__main__":
    main()
