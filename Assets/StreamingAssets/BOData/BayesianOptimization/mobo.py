# ==============================================================================
# HUMAN-IN-THE-LOOP  MULTI-OBJECTIVE  BAYESIAN  OPTIMISATION  (context-aware)
# Now supports *variable-length* context vectors, length communicated at start-up
# Complete script: observations CSV, run-time CSV, hyper-volume CSV
# ==============================================================================

from import_all import *           # torch, botorch, gpytorch, numpy, etc.
from matplotlib import pyplot as plt
import socket, pickle, pandas as pd, csv, os, time, numpy as np
from botorch.models import SingleTaskGP
from botorch.utils import draw_sobol_samples, is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.fixed_feature import FixedFeatureAcquisitionFunction
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import settings as gpy_settings

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL PLACE-HOLDERS (set after socket header)
# ──────────────────────────────────────────────────────────────────────────────
tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}

PROBLEM_DIM  = None
CONTEXT_DIM  = None
NUM_OBJS     = None

BATCH_SIZE   = None
NUM_RESTARTS = None
RAW_SAMPLES  = None
N_ITERATIONS = None
MC_SAMPLES   = None
N_INITIAL    = None
SEED         = None

parameter_names = []
objective_names = []
context_names   = []

device = tkwargs["device"]

# ──────────────────────────────────────────────────────────────────────────────
# CSV / FOLDER LOGGING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def get_unique_folder(parent: str, name: str) -> str:
    base = os.path.join(parent, name)
    if not os.path.exists(base):
        os.makedirs(base);  return base
    i = 1
    while True:
        trial = f"{base}_{i}"
        if not os.path.exists(trial):
            os.makedirs(trial);  return trial
        i += 1

def init_observation_csv(path, obj_names, ctx_names, param_names):
    header = (['UserID','ConditionID','GroupID','Timestamp',
               'Iteration','Phase','IsPareto']
              + obj_names + ctx_names + param_names)
    if not os.path.exists(path):
        with open(path,'w',newline='') as f:
            csv.writer(f,delimiter=';').writerow(header)

def append_observation(path, iteration, phase,
                       obj_vec, ctx_vec, x_vec, pareto=False):
    record = np.concatenate(([USER_ID,CONDITION_ID,GROUP_ID,
                              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),
                              iteration, phase, 'TRUE' if pareto else 'FALSE'],
                             obj_vec.cpu(), ctx_vec.cpu(), x_vec.cpu()))
    with open(path,'a',newline='') as f:
        csv.writer(f,delimiter=';').writerow(record)

def log_execution_time(path, iteration, seconds):
    if not os.path.exists(path):
        with open(path,'w',newline='') as f:
            csv.writer(f,delimiter=';').writerow(['Iteration','Execution_Time'])
    with open(path,'a',newline='') as f:
        csv.writer(f,delimiter=';').writerow([iteration,round(seconds,3)])

def log_hypervolume(path, iteration, value):
    if not os.path.exists(path):
        with open(path,'w',newline='') as f:
            csv.writer(f,delimiter=';').writerow(['Iteration','Hypervolume'])
    with open(path,'a',newline='') as f:
        csv.writer(f,delimiter=';').writerow([iteration,value])

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: context → tensor (runtime CONTEXT_DIM)
# ──────────────────────────────────────────────────────────────────────────────
def get_context_vector(lst):
    if len(lst)!=CONTEXT_DIM:
        raise ValueError(f"Expected {CONTEXT_DIM} context values, got {len(lst)}")
    return torch.tensor(lst, dtype=torch.float64, device=device)

# ──────────────────────────────────────────────────────────────────────────────
#  SOCKET INITIALISATION  (receive config from Unity)
# ──────────────────────────────────────────────────────────────────────────────
HOST, PORT = '', 56001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT)); sock.listen(1)
print('Waiting for Unity…', flush=True)
conn, addr = sock.accept(); print('Connected:', addr, flush=True)

raw_init = conn.recv(4096).decode('utf-8')
segments = raw_init.split('_')

hyper_block = segments[0]          # header counts + hyper-pars
warm_block  = segments[1]          # warmStart + file names (unused here)
range_block = segments[2]          # param/obj ranges
names_block = segments[3]          # paramNames
obj_names_block = segments[4]      # objNames
ctx_names_block = segments[5]      # ctxNames
study_block  = segments[6]         # IDs

# --- unpack hyper-block
hyper_vals = list(map(int, hyper_block.split(',')))
(BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES,
 N_ITERATIONS, MC_SAMPLES, N_INITIAL, SEED,
 PROBLEM_DIM, NUM_OBJS, CONTEXT_DIM) = hyper_vals

# --- info ranges
range_parts = range_block.split('/')
parameters_info = [list(map(float, r.split(','))) for r in range_parts[:PROBLEM_DIM]]
objectives_info = [list(map(float, r.split(','))) for r in range_parts[PROBLEM_DIM:]]

# --- names
parameter_names = names_block.split(',')
objective_names = obj_names_block.split(',')
context_names   = ctx_names_block.split(',')

USER_ID, CONDITION_ID, GROUP_ID = study_block.split(',')[:3]

# Derived globals
ref_point = torch.tensor([-1.]*NUM_OBJS, device=device)
bounds = torch.zeros(2, PROBLEM_DIM+CONTEXT_DIM, **tkwargs); bounds[1] = 1.

# ──────────────────────────────────────────────────────────────────────────────
# NORMALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def denorm_param(v, lo, hi): return round(lo + v*(hi-lo), 2)
def scale_obj(v, lo, hi, invert):
    norm = (v-lo)/(hi-lo)*2 - 1
    return max(-1, min(1, -norm if invert==1 else norm))

# ──────────────────────────────────────────────────────────────────────────────
# SOCKET-BACKED OBJECTIVE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def objective_function(x_tensor):
    x_np = x_tensor.cpu().numpy()
    x_dn = [denorm_param(x_np[i], parameters_info[i][0], parameters_info[i][1])
            for i in range(PROBLEM_DIM)]
    conn.sendall(("parameters," + ",".join(f"{v:.3f}" for v in x_dn)).encode())

    resp = conn.recv(4096)
    nums = list(map(float, resp.decode().split(',')))
    obj_raw, ctx_raw = nums[:NUM_OBJS], nums[NUM_OBJS:]
    ctx_vec = get_context_vector(ctx_raw)

    obj_norm = torch.tensor([scale_obj(obj_raw[i], *objectives_info[i]) for i in range(NUM_OBJS)],
                            dtype=torch.float64, device=device)
    return obj_norm, ctx_vec

# ──────────────────────────────────────────────────────────────────────────────
# INITIAL SOBOL DATA
# ──────────────────────────────────────────────────────────────────────────────
def initial_data(q):
    xs, ys, cs = [], [], []
    sobol = draw_sobol_samples(bounds=bounds[:,:PROBLEM_DIM], n=1, q=q,
                               seed=torch.randint(1_000_000,(1,)).item()).squeeze(0)
    for x in sobol:
        y,c = objective_function(x)
        xs.append(x); ys.append(y); cs.append(c)
    X = torch.cat([torch.stack(xs), torch.stack(cs)], dim=1)
    Y = torch.stack(ys)
    return X, Y, torch.stack(cs)

# ──────────────────────────────────────────────────────────────────────────────
# GP INITIALISER
# ──────────────────────────────────────────────────────────────────────────────
def init_gp(X,Y):
    with gpy_settings.fast_computations():
        model = SingleTaskGP(X,Y)
    return ExactMarginalLogLikelihood(model.likelihood, model), model

# =============================================================================
# MAIN LOOP
# =============================================================================
def run_mobo():
    proj = get_unique_folder(os.path.join(os.getcwd(),"LogData"), USER_ID)
    obs_csv = os.path.join(proj,"Observations.csv")
    time_csv= os.path.join(proj,"RunTimes.csv")
    hv_csv  = os.path.join(proj,"HyperVolumes.csv")
    init_observation_csv(obs_csv, objective_names, context_names, parameter_names)

    X, Y, C_last = initial_data(N_INITIAL)
    for i,(y,c,x_d) in enumerate(zip(Y, C_last, X[:,:PROBLEM_DIM])):
        append_observation(obs_csv, i+1, 'sampling', y, c, x_d)

    hv = Hypervolume(ref_point=ref_point)
    mll, model = init_gp(X,Y)
    hv_val = hv.compute(Y[is_non_dominated(Y)])
    log_hypervolume(hv_csv, 0, hv_val)

    for it in range(1, N_ITERATIONS+1):
        t0 = time.time()
        fit_gpytorch_mll(mll)
        sampler = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]), seed=SEED)

        ctx_ids = list(range(PROBLEM_DIM, PROBLEM_DIM+CONTEXT_DIM))
        acq_base = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            sampler=sampler,
            partitioning=NondominatedPartitioning(ref_point=ref_point, Y=Y))
        acq_ctx = FixedFeatureAcquisitionFunction(acq_base,
                                                  d=PROBLEM_DIM+CONTEXT_DIM,
                                                  indices=ctx_ids,
                                                  values=C_last.unsqueeze(0))

        X_new_d, _ = optimize_acqf(acq_ctx,
                                   bounds=bounds[:,:PROBLEM_DIM],
                                   q=BATCH_SIZE,
                                   num_restarts=NUM_RESTARTS,
                                   raw_samples=RAW_SAMPLES,
                                   options={"batch_limit":5,"maxiter":200},
                                   sequential=True)

        Y_new, C_last = objective_function(X_new_d[0])
        X_new = torch.cat([X_new_d[0], C_last]).unsqueeze(0)

        X = torch.cat([X, X_new]);  Y = torch.cat([Y, Y_new.unsqueeze(0)])
        append_observation(obs_csv, N_INITIAL+it, 'optimisation',
                           Y_new, C_last, X_new_d[0])

        hv_val = hv.compute(Y[is_non_dominated(Y)])
        log_hypervolume(hv_csv, it, hv_val)
        log_execution_time(time_csv, it, time.time()-t0)

        mll, model = init_gp(X,Y)
        print(f"Iter {it}/{N_ITERATIONS}  HV={hv_val:.4f}", flush=True)

    conn.sendall(b'optimization_finished,')
    print("Optimisation finished.", flush=True)

if __name__ == "__main__":
    run_mobo()
