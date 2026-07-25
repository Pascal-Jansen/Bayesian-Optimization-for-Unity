# meta_mobo_runtime.py — multi-objective Meta-BO backend (TAF-EHVI, NDJSON protocol).
#
# Speaks the same wire protocol as mobo.py (TCP server on port 56001, newline-delimited
# JSON: one init message, then a parameters -> objectives loop with tempCoverage /
# coverage / optimization_finished updates) and writes the same CSV family, but the
# optimizer is openbo's MOTAFSequentialOptimizer: BoTorch qLogNEHVI blended with
# hypervolume-improvement terms from "source" models built offline from PRIOR runs
# (population models in the sense of Liao et al., CHI '24). With zero valid sources the
# acquisition degenerates exactly to plain qLogNEHVI, i.e. this backend behaves like a
# (uniform-sampled) mobo.py run.
#
# Deliberate scope limits (validated, not silently ignored):
#   - multi-objective only (nObjectives >= 2),
#   - no warm start (population models are the transfer mechanism here),
#   - no contextual optimization (the LCE-M context pipeline stays BoTorch-only).
#
# openbo (https://github.com/M-Colley/openbo) is imported lazily inside the run so this
# module can be imported -- and its protocol/CSV logic tested -- without the heavy stack.

import csv
import json
import os
import shutil
import socket
import sys
import time

import numpy as np
import pandas as pd

# Sibling module imports must work both when running this file as a script and
# when loading it from another working directory (e.g. the test suite).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import bo_normalize
import meta_fingerprint

# -------------------- defaults (overwritten by Unity init) --------------------
N_INITIAL = 5
N_ITERATIONS = 10
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 1024
MC_SAMPLES = 512
SEED = 3

PROBLEM_DIM = None
NUM_OBJS = None  # must be >= 2

# Meta-TAF configuration (overwritten from the init message's meta* fields).
META_SOURCE_DIR = "MetaSources"
META_WEIGHT_MODE = "taf_r"
META_RHO = 1.0
META_TARGET_WEIGHT = 1.0
META_WARMUP_ITERS = 1
META_DECAY_START_ITER = 2
META_DECAY_RATE = 0.3

# paths/state
PROJECT_PATH = ""

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
FRAME = None           # canonical frame of THIS study (meta_fingerprint.canonical_frame)

REF_POINT_VALUE = -1.0  # objectives live in [-1, 1] maximization; ref point is [-1]^M

# -------------------- TCP server helpers (same contract as bo.py/mobo.py) ------------
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


def is_non_dominated_mask(values):
    """Boolean mask of Pareto-optimal rows (maximization). Small-n numpy, no deps."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(arr[j] >= arr[i]) and np.any(arr[j] > arr[i]):
                mask[i] = False
                break
    return mask

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

# -------------------- openbo import (lazy, fail-fast) --------------------
def _import_openbo():
    """Import the openbo pieces this backend needs, with an actionable error."""
    # BoTorch's acquisition functions try to JIT-compile a fused C++ kernel on first use.
    # torch serializes those builds with a lock file in a SHARED cache directory -- and a
    # backend process that gets killed mid-run (Unity stop, crash, task manager) leaves
    # the lock behind, after which every later run hangs forever waiting on it. Giving
    # each process its own extensions directory makes the attempt fail fast in isolation
    # (the compile is a ~3x-speedup nicety, not a requirement). Users who really compile
    # kernels can pre-set TORCH_EXTENSIONS_DIR themselves; we only fill it when unset.
    if "TORCH_EXTENSIONS_DIR" not in os.environ:
        import tempfile
        os.environ["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(prefix="bo_torch_ext_")
    try:
        from openbo.optimizers.mobo_botorch import compute_hypervolume
        from openbo.optimizers.mobo_taf import MOTAFConfig, MOTAFSequentialOptimizer
    except ImportError as e:
        raise RuntimeError(
            "The Meta-TAF backend needs the 'openbo' package (M-Colley fork with the "
            "multi-objective optimizers), which is not installed for this Python. Install it with:\n"
            "    python -m pip install \"open-bo @ git+https://github.com/M-Colley/openbo@main\"\n"
            "or, from a local clone:\n"
            "    python -m pip install -e path/to/openbo\n"
            "See docs/meta-taf-student-guide.md for details."
        ) from e
    return MOTAFConfig, MOTAFSequentialOptimizer, compute_hypervolume

# -------------------- source artifact validation --------------------
def resolve_meta_source_dir():
    """Resolve the configured source directory to an absolute path."""
    raw = str(META_SOURCE_DIR or "").strip() or "MetaSources"
    if os.path.isabs(raw):
        return raw
    root = os.environ.get("BO_META_ROOT") or os.getcwd()
    return os.path.join(root, raw)


def validate_and_stage_sources(source_dir, staging_dir):
    """Copy frame-compatible artifact pairs into ``staging_dir``; return kept names.

    Every source artifact must carry the canonical frame it was generated from
    (parameter names + bounds, objective names + bounds + minimize flags). Artifacts are
    stored already normalized, so a frame mismatch is invisible to shape checks -- a prior
    study with the same d and M but different bounds, or a flipped minimize flag, would
    silently transfer a rescaled or exactly INVERTED response surface. Mismatches are
    therefore skipped loudly, field by field. Unframed artifacts are skipped too unless
    BO_META_ALLOW_UNFRAMED=1 (escape hatch for hand-built fixtures).

    The staged copies live inside the run folder, which doubles as an audit trail of
    exactly which population models influenced this run.
    """
    gp_dir = os.path.join(source_dir, "gp_states")
    traj_dir = os.path.join(source_dir, "trajectories")
    os.makedirs(os.path.join(staging_dir, "gp_states"), exist_ok=True)
    os.makedirs(os.path.join(staging_dir, "trajectories"), exist_ok=True)

    if not os.path.isdir(gp_dir):
        print(f"Meta-TAF: source directory has no gp_states folder: {gp_dir}", flush=True)
        return []

    allow_unframed = os.environ.get("BO_META_ALLOW_UNFRAMED", "0") == "1"
    kept = []
    for fname in sorted(os.listdir(gp_dir)):
        if not fname.endswith(".json"):
            continue
        name = fname[:-len(".json")]
        gp_path = os.path.join(gp_dir, fname)
        traj_path = os.path.join(traj_dir, fname)
        if not os.path.exists(traj_path):
            print(f"Meta-TAF: source '{name}' has no trajectory file; skipping.", flush=True)
            continue
        try:
            with open(gp_path, "r", encoding="utf-8") as f:
                gp_payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Meta-TAF: source '{name}' gp_states unreadable ({e}); skipping.", flush=True)
            continue

        frame = gp_payload.get("frame")
        if frame is None:
            if not allow_unframed:
                print(
                    f"Meta-TAF: source '{name}' carries no frame block; skipping. "
                    "Regenerate it with meta_train.py (or set BO_META_ALLOW_UNFRAMED=1 "
                    "if you really know the frames match).",
                    flush=True,
                )
                continue
        else:
            diffs = meta_fingerprint.frame_differences(FRAME, frame)
            if diffs:
                print(
                    f"Meta-TAF: source '{name}' was built for a different study frame; skipping:",
                    flush=True,
                )
                for d in diffs:
                    print(f"    - {d}", flush=True)
                continue

        shutil.copyfile(gp_path, os.path.join(staging_dir, "gp_states", fname))
        shutil.copyfile(traj_path, os.path.join(staging_dir, "trajectories", fname))
        kept.append(name)

    return kept

# -------------------- objective evaluation over the socket --------------------
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


def objective_function(conn, x_unit):
    """Send one design (unit cube) to Unity; return its normalized objective row."""
    x = np.asarray(x_unit, dtype=np.float64).reshape(-1)
    values = {}
    for i, name in enumerate(parameter_names):
        lo, hi = parameters_info[i]
        # Keep full precision for optimizer-proposed points sent to Unity.
        values[name] = bo_normalize.denormalize_to_original_param(x[i], lo, hi, decimals=None)

    payload = {"type": "parameters", "values": values}
    print("Send parameters:", payload, flush=True)
    send_json_line(conn, payload)

    resp = recv_objectives_blocking(conn)
    if resp is None:
        raise RuntimeError("No objectives received from Unity.")
    if not isinstance(resp, dict):
        raise TypeError(f"Unity objectives payload must be a dict, got {type(resp).__name__}")

    missing = [name for name in objective_names if name not in resp]
    if missing:
        raise KeyError(f"Unity objectives missing required key(s): {missing}")
    unexpected = sorted([k for k in resp.keys() if k not in set(objective_names)])
    if unexpected:
        raise KeyError(f"Unity objectives payload contains unexpected key(s): {unexpected}")

    fs = []
    for i, name in enumerate(objective_names):
        lo, hi, minflag = objectives_info[i]
        fs.append(bo_normalize.normalize_objective_value(resp[name], lo, hi, minflag, name=name))
    return np.asarray(fs, dtype=np.float64)

# -------------------- logging --------------------
def expected_observation_columns():
    return (['UserID', 'ConditionID', 'GroupID', 'Timestamp', 'Iteration', 'Phase', 'IsPareto']
            + objective_names + parameter_names)


def append_observation_row(iteration, phase, y_norm_row, x_unit_row):
    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    if not os.path.exists(obs_csv):
        with open(obs_csv, 'w', newline='') as f:
            csv.writer(f, delimiter=';').writerow(expected_observation_columns())

    x_den = [
        bo_normalize.denormalize_to_original_param(x_unit_row[j], parameters_info[j][0], parameters_info[j][1])
        for j in range(PROBLEM_DIM)
    ]
    y_den = [
        bo_normalize.denormalize_to_original_obj(
            y_norm_row[j], objectives_info[j][0], objectives_info[j][1], objectives_info[j][2]
        )
        for j in range(NUM_OBJS)
    ]
    row = [USER_ID, CONDITION_ID, GROUP_ID,
           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
           iteration, phase, 'FALSE', *y_den, *x_den]
    with open(obs_csv, 'a', newline='') as f:
        csv.writer(f, delimiter=';').writerow(row)


def rewrite_pareto_flags(y_all_norm):
    """Recompute IsPareto over every logged row of this run (max sense)."""
    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    if not os.path.exists(obs_csv):
        return
    flags = ['TRUE' if b else 'FALSE' for b in is_non_dominated_mask(y_all_norm).tolist()]
    df = pd.read_csv(obs_csv, delimiter=';')
    expected_cols = expected_observation_columns()
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"ObservationsPerEvaluation.csv columns mismatch. "
            f"Expected {expected_cols}, got {list(df.columns)}"
        )
    if len(df) != len(flags):
        raise ValueError(
            f"ObservationsPerEvaluation.csv row count {len(df)} does not match "
            f"observation count {len(flags)}"
        )
    df['IsPareto'] = df['IsPareto'].astype(str)
    df['IsPareto'] = flags
    df.to_csv(obs_csv, sep=';', index=False)


def save_hypervolume_to_file(hvs, iteration):
    hv_csv = os.path.join(PROJECT_PATH, "HypervolumePerEvaluation.csv")
    os.makedirs(os.path.dirname(hv_csv), exist_ok=True)
    write_header = not os.path.exists(hv_csv) or os.path.getsize(hv_csv) == 0
    with open(hv_csv, 'a', newline='') as f:
        w = csv.writer(f, delimiter=';')
        if write_header:
            w.writerow(["Hypervolume", "Run"])
        w.writerow([hvs[-1], iteration])


def append_meta_weights_row(weights_csv, iteration, optimizer, source_names):
    target_w = float(getattr(optimizer, "last_target_weight", 1.0))
    weights = np.asarray(getattr(optimizer, "last_source_weights", np.zeros(0)), dtype=np.float64)
    decay = 1.0
    decay_fn = getattr(optimizer, "_decay_factor", None)
    if callable(decay_fn):
        decay = float(decay_fn())
    padded = list(weights) + [0.0] * (len(source_names) - len(weights))
    with open(weights_csv, 'a', newline='') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow([iteration, target_w, decay, *[float(v) for v in padded]])

# -------------------- sampling --------------------
def draw_initial_unit_samples(n_samples, d, seed):
    """Sobol samples in [0,1]^d (uniform fallback when scipy is unavailable)."""
    try:
        from scipy.stats import qmc
        engine = qmc.Sobol(d=d, scramble=True, seed=seed)
        return np.asarray(engine.random(n_samples), dtype=np.float64)
    except ImportError:
        rng = np.random.default_rng(seed)
        return rng.random((n_samples, d)).astype(np.float64)

# -------------------- main loop --------------------
def meta_execute(conn, seed, iterations, initial_samples):
    global PROJECT_PATH
    MOTAFConfig, MOTAFSequentialOptimizer, compute_hypervolume = _import_openbo()

    base = os.environ.get("BO_LOG_ROOT") or os.path.join(os.getcwd(), "LogData")
    condition_base = os.path.join(base, USER_LOG_ID, CONDITION_LOG_ID)
    os.makedirs(condition_base, exist_ok=True)
    PROJECT_PATH = get_unique_folder(condition_base, "run")

    exec_csv = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(exec_csv, ['Optimization', 'Execution_Time'])

    # Stage frame-validated sources into the run folder (also the audit trail).
    source_dir = resolve_meta_source_dir()
    staging_dir = os.path.join(PROJECT_PATH, "MetaSourcesUsed")
    kept = validate_and_stage_sources(source_dir, staging_dir)
    if kept:
        print(f"Meta-TAF: using {len(kept)} population model(s): {kept}", flush=True)
    else:
        print(
            "Meta-TAF: no valid population models found; "
            "running plain multi-objective BO (qLogNEHVI).",
            flush=True,
        )

    ref_point = [REF_POINT_VALUE] * NUM_OBJS
    optimizer = MOTAFSequentialOptimizer(MOTAFConfig(
        bounds=[(0.0, 1.0)] * PROBLEM_DIM,
        ref_point=ref_point,
        taf_run_dir=staging_dir,
        n_init=0,
        n_iter=iterations + 2,  # headroom: the sampling observe() consumes one slot
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        mc_samples=MC_SAMPLES,
        seed=seed,
        rho=META_RHO,
        taf_weight_mode=META_WEIGHT_MODE,
        target_weight=META_TARGET_WEIGHT,
        source_only_warmup_iters=META_WARMUP_ITERS,
        decay_start_iter=META_DECAY_START_ITER,
        decay_rate=META_DECAY_RATE,
    ))

    source_names = [s.name for s in getattr(optimizer, "source_surrogates", [])]
    weights_csv = os.path.join(PROJECT_PATH, "MetaWeightsPerEvaluation.csv")
    with open(weights_csv, 'w', newline='') as f:
        csv.writer(f, delimiter=';').writerow(
            ["Iteration", "TargetWeight", "DecayFactor", *source_names]
        )

    # ---- sampling phase (Sobol, evaluated one-by-one through Unity) ----
    if initial_samples < 1:
        raise ValueError("numSamplingIterations must be >= 1 for the Meta-TAF backend.")
    x_init = draw_initial_unit_samples(initial_samples, PROBLEM_DIM, seed)
    print("Initial Sobol X in [0,1]:", x_init, flush=True)

    y_rows = []
    for i in range(initial_samples):
        print(f"---- Initial Sample {i+1}", flush=True)
        y_row = objective_function(conn, x_init[i])
        y_rows.append(y_row)
        append_observation_row(i + 1, 'sampling', y_row, x_init[i])
        send_json_line(conn, {"type": "tempCoverage", "value": float(i + 1) / float(max(1, initial_samples))})

    y_all = np.vstack(y_rows)
    x_all = np.asarray(x_init, dtype=np.float64)
    rewrite_pareto_flags(y_all)
    optimizer.observe(x_all, y_all)

    hvs = [float(compute_hypervolume(y_all, np.asarray(ref_point, dtype=np.float64)))]
    save_hypervolume_to_file(hvs, 0)
    send_json_line(conn, {"type": "coverage", "value": float(hvs[-1])})

    # ---- optimization phase (ask/tell against openbo) ----
    for it in range(1, iterations + 1):
        t0 = time.time()
        x_next = optimizer.suggest()  # (1, d) in the unit cube
        t_elapsed = time.time() - t0
        write_data_to_csv(exec_csv, ['Optimization', 'Execution_Time'],
                          [{'Optimization': it, 'Execution_Time': t_elapsed}])
        append_meta_weights_row(weights_csv, it, optimizer, source_names)

        y_row = objective_function(conn, x_next[0])
        optimizer.observe(x_next, y_row.reshape(1, -1))
        x_all = np.vstack([x_all, x_next])
        y_all = np.vstack([y_all, y_row.reshape(1, -1)])

        append_observation_row(initial_samples + it, 'optimization', y_row, x_next[0])
        rewrite_pareto_flags(y_all)
        hvs.append(float(compute_hypervolume(y_all, np.asarray(ref_point, dtype=np.float64))))
        save_hypervolume_to_file(hvs, it)
        send_json_line(conn, {"type": "coverage", "value": float(hvs[-1])})

    send_json_line(conn, {"type": "optimization_finished"})
    return hvs, x_all, y_all

# -------------------- boot --------------------
def main():
    global N_INITIAL, N_ITERATIONS, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, SEED
    global PROBLEM_DIM, NUM_OBJS
    global META_SOURCE_DIR, META_WEIGHT_MODE, META_RHO, META_TARGET_WEIGHT
    global META_WARMUP_ITERS, META_DECAY_START_ITER, META_DECAY_RATE
    global USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID
    global parameter_names, objective_names, parameters_info, objectives_info, FRAME
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

        backend = str(cfg.get("optimizerBackend") or "").strip().lower()
        if backend and backend != "meta-taf":
            raise ValueError(
                f"meta_mobo_runtime was launched for optimizerBackend='{backend}'; expected 'meta-taf'."
            )

        N_INITIAL      = get_cfg_int(cfg, "numSamplingIterations", default=N_INITIAL)
        N_ITERATIONS   = get_cfg_int(cfg, "numOptimizationIterations", default=N_ITERATIONS)
        BATCH_SIZE     = get_cfg_int(cfg, "batchSize", default=BATCH_SIZE)
        NUM_RESTARTS   = get_cfg_int(cfg, "numRestarts", default=NUM_RESTARTS)
        RAW_SAMPLES    = get_cfg_int(cfg, "rawSamples", default=RAW_SAMPLES)
        MC_SAMPLES     = get_cfg_int(cfg, "mcSamples", default=MC_SAMPLES)
        SEED           = get_cfg_int(cfg, "seed", default=SEED)
        PROBLEM_DIM    = get_cfg_int(cfg, "nParameters", required=True)
        NUM_OBJS       = get_cfg_int(cfg, "nObjectives", required=True)

        META_SOURCE_DIR       = str(cfg.get("metaSourceDir") or META_SOURCE_DIR)
        META_WEIGHT_MODE      = str(cfg.get("metaWeightMode") or META_WEIGHT_MODE).strip().lower()
        META_RHO              = get_cfg_float(cfg, "metaRho", default=META_RHO)
        META_TARGET_WEIGHT    = get_cfg_float(cfg, "metaTargetWeight", default=META_TARGET_WEIGHT)
        META_WARMUP_ITERS     = get_cfg_int(cfg, "metaWarmupIters", default=META_WARMUP_ITERS)
        META_DECAY_START_ITER = get_cfg_int(cfg, "metaDecayStartIter", default=META_DECAY_START_ITER)
        META_DECAY_RATE       = get_cfg_float(cfg, "metaDecayRate", default=META_DECAY_RATE)

        if PROBLEM_DIM < 1:
            raise ValueError(f"nParameters must be >= 1, got {PROBLEM_DIM}")
        if NUM_OBJS < 2:
            raise ValueError(f"meta_mobo_runtime expects at least 2 objectives, got {NUM_OBJS}")
        if N_INITIAL < 1 or N_ITERATIONS < 0:
            raise ValueError(
                f"Iteration counts invalid: sampling={N_INITIAL} (must be >= 1), "
                f"optimization={N_ITERATIONS} (must be >= 0)"
            )
        if NUM_RESTARTS < 1 or RAW_SAMPLES < 1 or MC_SAMPLES < 1:
            raise ValueError(
                f"numRestarts/rawSamples/mcSamples must be >=1, got {NUM_RESTARTS}/{RAW_SAMPLES}/{MC_SAMPLES}"
            )
        if BATCH_SIZE != 1:
            print(f"Warning: batchSize={BATCH_SIZE} is not supported in this HITL loop; forcing batchSize=1.", flush=True)
            BATCH_SIZE = 1
        if META_WEIGHT_MODE not in ("taf_m", "taf_r"):
            raise ValueError(f"metaWeightMode must be 'taf_m' or 'taf_r', got '{META_WEIGHT_MODE}'")
        if META_RHO <= 0:
            raise ValueError(f"metaRho must be > 0, got {META_RHO}")
        if META_TARGET_WEIGHT <= 0:
            raise ValueError(f"metaTargetWeight must be > 0, got {META_TARGET_WEIGHT}")
        if META_WARMUP_ITERS < 0 or META_DECAY_START_ITER < 0:
            raise ValueError("metaWarmupIters and metaDecayStartIter must be >= 0")
        if not (0.0 <= META_DECAY_RATE <= 1.0):
            raise ValueError(f"metaDecayRate must be in [0, 1], got {META_DECAY_RATE}")

        # Explicit scope guards: fail fast instead of silently ignoring configuration.
        if bool(cfg.get("warmStart", False)):
            raise ValueError(
                "The Meta-TAF backend does not support warm start: population models are "
                "its transfer mechanism. Disable Warm Start or use the BoTorch backend."
            )
        context_cfg = init_msg.get("context") or {}
        if isinstance(context_cfg, dict) and bool(context_cfg.get("enabled", False)):
            raise ValueError(
                "Contextual optimization (LCE-M GP) is only supported with the BoTorch "
                "backend, not with Meta-TAF."
            )

        user = init_msg.get("user", {}) or {}
        USER_ID      = normalize_user_token(user.get("userId"), default="-1")
        CONDITION_ID = normalize_user_token(user.get("conditionId"), default="-1")
        GROUP_ID     = normalize_user_token(user.get("groupId"), default="-1")
        USER_LOG_ID  = normalize_log_folder_token(USER_ID, default="-1")
        CONDITION_LOG_ID = normalize_log_folder_token(CONDITION_ID, default="-1")

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

        FRAME = meta_fingerprint.canonical_frame(
            parameter_names, parameters_info, objective_names, objectives_info
        )

        print("Init OK:", dict(
            BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES,
            N_ITERATIONS=N_ITERATIONS, MC_SAMPLES=MC_SAMPLES,
            N_INITIAL=N_INITIAL, SEED=SEED, PROBLEM_DIM=PROBLEM_DIM, NUM_OBJS=NUM_OBJS,
            META_WEIGHT_MODE=META_WEIGHT_MODE, META_SOURCE_DIR=META_SOURCE_DIR,
            META_WARMUP_ITERS=META_WARMUP_ITERS,
            META_DECAY=(META_DECAY_START_ITER, META_DECAY_RATE),
            FRAME_DIGEST=meta_fingerprint.frame_digest(FRAME),
        ), flush=True)

        meta_execute(conn, SEED, N_ITERATIONS, N_INITIAL)
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
