from matplotlib import pyplot as plt
import socket
import pickle
import pandas as pd
import time
import csv
import os
import numpy as np
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

# -------------------- placeholders (overwritten by Unity init) --------------------
N_INITIAL = 5
N_ITERATIONS = 10
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 1024
MC_SAMPLES = 512
SEED = 3
PROBLEM_DIM = None    # build tensors only AFTER init arrives
NUM_OBJS = None

# built after init:
ref_point = None
problem_bounds = None

# paths/state
CURRENT_DIR = ""
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

# device
tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
device = torch.device("cpu")

# -------------------- TCP server --------------------
HOST = ''
PORT = 56001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print('Server starts, waiting for connection...', flush=True)
conn, addr = s.accept()
print('Connected by', addr, flush=True)

# wait for Unity init payload
data = conn.recv(1024)

# -------------------- parse init --------------------
init_data = data.decode("utf-8").split('_')

# [0]: numeric params
param_list = init_data[0].split(',')
(BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, N_ITERATIONS,
 MC_SAMPLES, N_INITIAL, SEED, PROBLEM_DIM, NUM_OBJS) = map(int, param_list)

# [1]: warm start
warm = init_data[1].split(',')
WARM_START = warm[0].lower() in ("true", "1", "yes")
CSV_PATH_PARAMETERS = str(warm[1])
CSV_PATH_OBJECTIVES = str(warm[2])

print("Init:", BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, N_ITERATIONS, MC_SAMPLES,
      N_INITIAL, SEED, PROBLEM_DIM, NUM_OBJS, flush=True)

# [2]: parameter bounds "low,high/low,high/..."
parameter_raw = init_data[2].split('/')
parameters_info = [list(map(float, r.split(','))) for r in parameter_raw] if init_data[2] else []

# [3]: objective bounds "low,high,minimizeFlag/..."
objective_raw = init_data[3].split('/')
objectives_info = [list(map(float, r.split(','))) for r in objective_raw] if init_data[3] else []

# [4]: parameter names
parameter_names = init_data[4].split(',') if init_data[4] else []

# [5]: objective names
objective_names = init_data[5].split(',') if init_data[5] else []

# [6]: study info
USER_ID, CONDITION_ID, GROUP_ID = init_data[6].split(',')

# sanity checks and build tensors with real dims
if len(parameters_info) != PROBLEM_DIM:
    raise ValueError(f"parameters_info len {len(parameters_info)} != PROBLEM_DIM {PROBLEM_DIM}")
if len(objectives_info) != NUM_OBJS:
    raise ValueError(f"objectives_info len {len(objectives_info)} != NUM_OBJS {NUM_OBJS}")

ref_point = torch.full((NUM_OBJS,), -1.0, dtype=torch.double)  # model Y in [-1,1], max
problem_bounds = torch.stack(
    [torch.zeros(PROBLEM_DIM, dtype=torch.double),
     torch.ones (PROBLEM_DIM, dtype=torch.double)],
    dim=0
)

# -------------------- helpers --------------------
def get_unique_folder(parent, folder_name):
    base_path = os.path.join(parent, folder_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path); return base_path
    k = 1
    while True:
        p = os.path.join(parent, f"{folder_name}_{k}")
        if not os.path.exists(p):
            os.makedirs(p); return p
        k += 1

def create_csv_file(csv_file_path, fieldnames):
    try:
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        write_header = not os.path.exists(csv_file_path)
        with open(csv_file_path, 'a+', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            if write_header: w.writeheader()
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
    return np.round(lo + val01 * (hi - lo), 2)

def denormalize_to_original_obj(v_m1p1, lo, hi, smaller_is_better):
    if smaller_is_better == 1: v_m1p1 *= -1
    return np.round(lo + (v_m1p1 + 1) * 0.5 * (hi - lo), 2)

# -------------------- objective evaluation --------------------
def objective_function(x_tensor):
    x = x_tensor.cpu().numpy()  # in [0,1]
    x_denorm = [denormalize_to_original_param(x[i], parameters_info[i][0], parameters_info[i][1])
                for i in range(PROBLEM_DIM)]
    msg = "parameters," + ",".join(str(round(v,3)) for v in x_denorm)
    print("Send Data:", msg, flush=True)
    conn.sendall(msg.encode("utf-8"))

    data = conn.recv(1024)
    rec = list(map(float, data.decode("utf-8").split(','))) if data else []
    if len(rec) != NUM_OBJS:
        print("received objective number not consistent", flush=True)

    # map to [-1,1], higher is better
    fs = []
    for i in range(NUM_OBJS):
        lo, hi, minflag = objectives_info[i]
        f = (rec[i] - lo) / (hi - lo) * 2 - 1
        if int(minflag) == 1: f *= -1
        fs.append(max(-1.0, min(1.0, f)))
    print("Objective normalized:", fs, flush=True)
    return torch.tensor(fs, dtype=torch.double)

# -------------------- data IO --------------------
def generate_initial_data(n_samples):
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
        y = objective_function(x)
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
    if X_baseline.dim() == 3: X_baseline = X_baseline[0]
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
    return candidates.detach()  # in [0,1]

# -------------------- logging --------------------
def save_xy(x_sample, y_sample, iteration):
    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    print("Project Path for Observations:", PROJECT_PATH, flush=True)

    # recompute Pareto mask for all rows
    pareto_mask = is_non_dominated(y_sample).tolist()

    x_np = x_sample.clone().cpu().numpy()
    y_np = y_sample.clone().cpu().numpy()

    # denormalize last row
    for j in range(PROBLEM_DIM):
        x_np[-1][j] = denormalize_to_original_param(x_np[-1][j], parameters_info[j][0], parameters_info[j][1])
    for j in range(NUM_OBJS):
        y_np[-1][j] = denormalize_to_original_obj(y_np[-1][j], objectives_info[j][0], objectives_info[j][1], objectives_info[j][2])

    if os.path.exists(obs_csv):
        df = pd.read_csv(obs_csv, delimiter=';')
    else:
        cols = ['UserID','ConditionID','GroupID','Timestamp','Iteration','Phase','IsPareto'] + objective_names + parameter_names
        df = pd.DataFrame(columns=cols)

    new_row = pd.DataFrame([[
        USER_ID,
        CONDITION_ID,
        GROUP_ID,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        iteration + N_INITIAL,
        'optimization',
        'FALSE',
        *y_np[-1], *x_np[-1]
    ]], columns=df.columns)

    df = pd.concat([df, new_row], ignore_index=True)

    # update IsPareto for all rows
    flags = ['TRUE' if b else 'FALSE' for b in pareto_mask]
    if len(flags) == len(df):
        df['IsPareto'] = flags

    df.to_csv(obs_csv, sep=';', index=False)

def save_hypervolume_to_file(hvs, iteration):
    hv_csv = os.path.join(PROJECT_PATH, "HypervolumePerEvaluation.csv")
    print("Project Path for Hypervolumes:", PROJECT_PATH, flush=True)
    os.makedirs(os.path.dirname(hv_csv), exist_ok=True)
    write_header = not os.path.exists(hv_csv) or os.path.getsize(hv_csv) == 0
    with open(hv_csv, 'a', newline='') as f:
        w = csv.writer(f, delimiter=';')
        if write_header: w.writerow(["Hypervolume", "Run"])
        w.writerow([hvs[-1], iteration])

# -------------------- main loop --------------------
def mobo_execute(seed, iterations, initial_samples):
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
        print("Loading warm start data...", flush=True)
        train_x, train_y = load_data()
    else:
        print("Generating initial training data...", flush=True)
        train_x, train_y = generate_initial_data(n_samples=initial_samples)

    mll, model = initialize_model(train_x, train_y)

    pareto_mask = is_non_dominated(train_y)
    volume = hv_util.compute(train_y[pareto_mask])
    hvs.append(volume)
    save_hypervolume_to_file(hvs, 0)

    for it in range(1, iterations + 1):
        print(f"---- MOBO Iteration {it}", flush=True)
        t0 = time.time()

        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]), seed=SEED)
        new_x = optimize_qnehvi(model, sampler)

        t_elapsed = time.time() - t0
        write_data_to_csv(exec_csv, ['Optimization', 'Execution_Time'], [{'Optimization': it, 'Execution_Time': t_elapsed}])
        print(f"Iter {it} time: {t_elapsed:.2f}s", flush=True)

        new_y = objective_function(new_x[0])

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y.unsqueeze(0)])

        pareto_mask = is_non_dominated(train_y)
        volume = hv_util.compute(train_y[pareto_mask])
        hvs.append(volume)
        save_xy(train_x, train_y, it)
        save_hypervolume_to_file(hvs, it)

        mll, model = initialize_model(train_x, train_y)

    print("Send Data: optimization_finished,", flush=True)
    conn.sendall(b'optimization_finished,')
    return hvs, train_x, train_y

# -------------------- run --------------------
hvs_qnehvi, train_x_qnehvi, train_obj_qnehvi = mobo_execute(SEED, N_ITERATIONS, N_INITIAL)
