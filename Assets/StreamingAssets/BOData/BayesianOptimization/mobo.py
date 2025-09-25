import json
import socket
import time
import csv
import os
import numpy as np
import pandas as pd
import torch

from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
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

# study info
USER_ID = ""
CONDITION_ID = ""
GROUP_ID = ""

# names and meta parsed from init
parameter_names = []
objective_names = []
parameters_info = []   # [(lo, hi)]
objectives_info = []   # [(lo, hi, minimizeFlag)]

# device
tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
device = torch.device("cpu")

# -------------------- TCP server helpers --------------------
HOST = ''
PORT = 56001

def send_json_line(conn, obj):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    conn.sendall(line.encode("utf-8"))

def ndjson_reader(conn):
    """Yield complete JSON objects from a TCP socket using newline framing."""
    buf = ""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            if buf.strip():
                try:
                    yield json.loads(buf)
                except Exception:
                    pass
            return
        buf += chunk.decode("utf-8", errors="replace")
        while True:
            idx = buf.find("\n")
            if idx < 0:
                break
            line = buf[:idx].rstrip("\r")
            buf = buf[idx + 1:]
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print("JSON decode error:", e, "line:", line, flush=True)

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
    try:
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        write_header = not os.path.exists(csv_file_path)
        with open(csv_file_path, 'a+', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            if write_header:
                w.writeheader()
    except Exception as e:
        print("Error creating file:", str(e), flush=True)

def write_data_to_csv(csv_file_path, fieldnames, rows):
    try:
        with open(csv_file_path, 'a+', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            w.writerows(rows)
    except Exception as e:
        print("Error writing to file:", str(e), flush=True)

def denormalize_to_original_param(val01, lo, hi):
    return np.round(lo + val01 * (hi - lo), 3)

def denormalize_to_original_obj(v_m1p1, lo, hi, smaller_is_better):
    v = -v_m1p1 if int(smaller_is_better) == 1 else v_m1p1
    return np.round(lo + (v + 1) * 0.5 * (hi - lo), 3)

# -------------------- protocol parsing --------------------
def parse_param_init(init_val):
    # Accept typed JSON: {"low": ..., "high": ...}
    if isinstance(init_val, dict):
        return float(init_val["low"]), float(init_val["high"])
    parts = [p.strip() for p in str(init_val).split(",")]
    if len(parts) < 2:
        raise ValueError(f"Parameter init parse error: '{init_val}'")
    return float(parts[0]), float(parts[1])

def parse_obj_init(init_val):
    # Accept typed JSON: {"low": ..., "high": ..., "minimize": 0/1}
    if isinstance(init_val, dict):
        return float(init_val["low"]), float(init_val["high"]), int(init_val.get("minimize", 0))
    parts = [p.strip() for p in str(init_val).split(",")]
    if len(parts) < 3:
        raise ValueError(f"Objective init parse error: '{init_val}'")
    return float(parts[0]), float(parts[1]), int(float(parts[2]))

# -------------------- objective evaluation --------------------
def recv_objectives_blocking(conn):
    for msg in ndjson_reader(conn):
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
        values[name] = float(denormalize_to_original_param(x[i], lo, hi))

    payload = {"type": "parameters", "values": values}
    print("Send parameters:", payload, flush=True)
    send_json_line(conn, payload)

    resp = recv_objectives_blocking(conn)
    if resp is None:
        raise RuntimeError("No objectives received from Unity.")

    fs = []
    rec_missing = []
    for i, name in enumerate(objective_names):
        val = float(resp.get(name, 0.0))
        if name not in resp:
            rec_missing.append(name)
        lo, hi, minflag = objectives_info[i]
        f = 0.0 if hi == lo else (val - lo) / (hi - lo) * 2 - 1
        if int(minflag) == 1:
            f *= -1
        fs.append(max(-1.0, min(1.0, f)))

    if rec_missing:
        print("Warning: missing objective(s) from Unity:", rec_missing, flush=True)

    return torch.tensor(fs, dtype=torch.double)

# -------------------- data IO --------------------
def generate_initial_data(conn, n_samples):
    global PROJECT_PATH
    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    if not os.path.exists(obs_csv):
        header = ['UserID','ConditionID','GroupID','Timestamp','Iteration','Phase','IsPareto'] + objective_names + parameter_names
        with open(obs_csv, 'w', newline='') as f:
            csv.writer(f, delimiter=';').writerow(header)

    train_x = draw_sobol_samples(bounds=problem_bounds, n=1, q=n_samples, seed=SEED).squeeze(0)
    print("Initial Sobol X in [0,1]:", train_x, flush=True)

    train_obj = []
    for i, x in enumerate(train_x):
        print(f"---- Initial Sample {i+1}", flush=True)
        y = objective_function(conn, x)
        train_obj.append(y)

        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        x_den = [denormalize_to_original_param(x_np[j], parameters_info[j][0], parameters_info[j][1]) for j in range(PROBLEM_DIM)]
        y_den = [denormalize_to_original_obj(y_np[j], objectives_info[j][0], objectives_info[j][1], objectives_info[j][2]) for j in range(NUM_OBJS)]
        row = [USER_ID, CONDITION_ID, GROUP_ID,
               time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
               i+1, 'sampling', 'FALSE', *y_den, *x_den]
        with open(obs_csv, 'a', newline='') as f:
            csv.writer(f, delimiter=';').writerow(row)
        send_json_line(conn, {"type": "tempCoverage", "value": float(i+1)/float(max(1,n_samples))})

    Y = torch.tensor(np.stack([t.numpy() for t in train_obj], axis=0), dtype=torch.double)
    return train_x, Y

def load_data():
    cur = os.getcwd()
    y = pd.read_csv(os.path.join(cur, "InitData", CSV_PATH_OBJECTIVES), delimiter=';').values
    x = pd.read_csv(os.path.join(cur, "InitData", CSV_PATH_PARAMETERS), delimiter=';').values
    return torch.tensor(x, dtype=torch.double), torch.tensor(y, dtype=torch.double)

# -------------------- model --------------------
def initialize_model(train_x, train_obj):
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# -------------------- acquisition --------------------
def optimize_qnehvi(model, sampler):
    X_baseline = model.train_inputs[0]
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
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    return candidates.detach()

# -------------------- logging --------------------
def save_xy(x_sample, y_sample, iteration):
    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    pareto_mask = is_non_dominated(y_sample).tolist()
    x_np = x_sample.clone().cpu().numpy()
    y_np = y_sample.clone().cpu().numpy()

    for j in range(PROBLEM_DIM):
        x_np[-1][j] = denormalize_to_original_param(x_np[-1][j], parameters_info[j][0], parameters_info[j][1])
    for j in range(NUM_OBJS):
        y_np[-1][j] = denormalize_to_original_obj(y_np[-1][j], objectives_info[j][0], objectives_info[j][1], objectives_info[j][2])

    if os.path.exists(obs_csv):
        df = pd.read_csv(obs_csv, delimiter=';')
    else:
        cols = ['UserID','ConditionID','GroupID','Timestamp','Iteration','Phase','IsPareto'] + objective_names + parameter_names
        df = pd.DataFrame(columns=cols)

    new_row = pd.DataFrame([[USER_ID, CONDITION_ID, GROUP_ID,
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                             iteration + N_INITIAL, 'optimization', 'FALSE',
                             *y_np[-1], *x_np[-1]]], columns=df.columns)

    df = pd.concat([df, new_row], ignore_index=True)
    flags = ['TRUE' if b else 'FALSE' for b in pareto_mask]
    if len(flags) == len(df):
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

# -------------------- main loop --------------------
def mobo_execute(conn, seed, iterations, initial_samples):
    global PROJECT_PATH, OBSERVATIONS_LOG_PATH
    base = os.path.join(os.getcwd(), "LogData")
    os.makedirs(base, exist_ok=True)
    PROJECT_PATH = get_unique_folder(base, USER_ID)
    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")

    exec_csv = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(exec_csv, ['Optimization', 'Execution_Time'])

    torch.manual_seed(seed)
    hv_util = Hypervolume(ref_point=ref_point)
    hvs = []

    if WARM_START:
        train_x, train_y = load_data()
    else:
        train_x, train_y = generate_initial_data(conn, n_samples=initial_samples)

    mll, model = initialize_model(train_x, train_y)

    pareto_mask = is_non_dominated(train_y)
    volume = hv_util.compute(train_y[pareto_mask])
    hvs.append(volume)
    save_hypervolume_to_file(hvs, 0)
    send_json_line(conn, {"type": "coverage", "value": float(volume)})

    for it in range(1, iterations + 1):
        t0 = time.time()
        fit_gpytorch_mll(mll)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]), seed=SEED)
        new_x = optimize_qnehvi(model, sampler)
        t_elapsed = time.time() - t0
        write_data_to_csv(exec_csv, ['Optimization', 'Execution_Time'],
                          [{'Optimization': it, 'Execution_Time': t_elapsed}])

        new_y = objective_function(conn, new_x[0])
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y.unsqueeze(0)])

        pareto_mask = is_non_dominated(train_y)
        volume = hv_util.compute(train_y[pareto_mask])
        hvs.append(volume)
        save_xy(train_x, train_y, it)
        save_hypervolume_to_file(hvs, it)
        send_json_line(conn, {"type": "coverage", "value": float(volume)})
        mll, model = initialize_model(train_x, train_y)

    send_json_line(conn, {"type": "optimization_finished"})
    return hvs, train_x, train_y

# -------------------- boot --------------------
def main():
    global N_INITIAL, N_ITERATIONS, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, SEED
    global PROBLEM_DIM, NUM_OBJS, ref_point, problem_bounds
    global WARM_START, CSV_PATH_PARAMETERS, CSV_PATH_OBJECTIVES
    global USER_ID, CONDITION_ID, GROUP_ID
    global parameter_names, objective_names, parameters_info, objectives_info

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    print('Server starts, waiting for connection...', flush=True)
    conn, addr = s.accept()
    print('Connected by', addr, flush=True)

    reader = ndjson_reader(conn)
    init_msg = None
    for msg in reader:
        if msg.get("type") == "init":
            init_msg = msg
            break
    if init_msg is None:
        raise RuntimeError("Did not receive init message.")

    cfg = init_msg.get("config", {}) or {}
    N_INITIAL      = int(cfg.get("numSamplingIterations", N_INITIAL))
    N_ITERATIONS   = int(cfg.get("numOptimizationIterations", N_ITERATIONS))
    BATCH_SIZE     = int(cfg.get("batchSize", BATCH_SIZE))
    NUM_RESTARTS   = int(cfg.get("numRestarts", NUM_RESTARTS))
    RAW_SAMPLES    = int(cfg.get("rawSamples", RAW_SAMPLES))
    MC_SAMPLES     = int(cfg.get("mcSamples", MC_SAMPLES))
    SEED           = int(cfg.get("seed", SEED))
    PROBLEM_DIM    = int(cfg.get("nParameters"))
    NUM_OBJS       = int(cfg.get("nObjectives"))
    WARM_START     = bool(cfg.get("warmStart", False))

    CSV_PATH_PARAMETERS = str(cfg.get("initialParametersDataPath") or "")
    CSV_PATH_OBJECTIVES = str(cfg.get("initialObjectivesDataPath") or "")

    user = init_msg.get("user", {}) or {}
    USER_ID      = str(user.get("userId", "user"))
    CONDITION_ID = str(user.get("conditionId", "cond"))
    GROUP_ID     = str(user.get("groupId", "grp"))

    parameters = init_msg.get("parameters", []) or []
    objectives = init_msg.get("objectives", []) or []

    parameter_names = [p.get("key") for p in parameters]
    objective_names = [o.get("key") for o in objectives]

    if len(parameter_names) != PROBLEM_DIM:
        raise ValueError(f"parameter_names len {len(parameter_names)} != nParameters {PROBLEM_DIM}")
    if len(objective_names) != NUM_OBJS:
        raise ValueError(f"objective_names len {len(objective_names)} != nObjectives {NUM_OBJS}")

    parameters_info = [parse_param_init(p.get("init")) for p in parameters]
    objectives_info = [parse_obj_init(o.get("init")) for o in objectives]

    ref_point = torch.full((NUM_OBJS,), -1.0, dtype=torch.double)
    problem_bounds = torch.stack(
        [torch.zeros(PROBLEM_DIM, dtype=torch.double),
         torch.ones(PROBLEM_DIM, dtype=torch.double)],
        dim=0
    )

    print("Init OK:", dict(
        BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES,
        N_ITERATIONS=N_ITERATIONS, MC_SAMPLES=MC_SAMPLES,
        N_INITIAL=N_INITIAL, SEED=SEED, PROBLEM_DIM=PROBLEM_DIM, NUM_OBJS=NUM_OBJS
    ), flush=True)

    try:
        mobo_execute(conn, SEED, N_ITERATIONS, N_INITIAL)
    finally:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        conn.close()
        s.close()

if __name__ == "__main__":
    main()

