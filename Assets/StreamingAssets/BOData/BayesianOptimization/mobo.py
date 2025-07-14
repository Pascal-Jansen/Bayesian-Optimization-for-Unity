# ==============================================================================
#  HITL-MOBO with hierarchical GP prior
#  ─ Traits (Big-5 + desired pro-activity) are fixed covariates, not a context
# ==============================================================================

from import_all import *                                    # torch, botorch …
import socket, os, csv, time, pickle, numpy as np
from botorch.models import LinearMixedGP, ModelListGP
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.utils.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions import (
    NondominatedPartitioning,
)
from botorch.utils import draw_sobol_samples, is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
PRIOR_FILE = "global_prior.pt"      # produced by train_prior.py
LOG_DIR    = "LogData"

tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
device = tkwargs["device"]

# placeholders filled after header
PROBLEM_DIM = TRAIT_DIM = NUM_OBJS = None
BATCH_SIZE = NUM_RESTARTS = RAW_SAMPLES = N_ITERATIONS = None
MC_SAMPLES = N_INITIAL = SEED = None
parameter_names = objective_names = trait_names = []
parameters_info = objectives_info = None

TRAIT_VEC = None          # 1 × TRAIT_DIM tensor, fixed for this participant

# ──────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────────────────────
def mkdir(p): os.makedirs(p, exist_ok=True); return p
def init_csv(path, objs, traits, pars):
    with open(path,'w',newline='') as f:
        csv.writer(f,delimiter=';').writerow(
            ['UID','CID','GID','time','iter','phase','pareto']
            + objs + traits + pars)
def row_csv(path, row):
    with open(path,'a',newline='') as f:
        csv.writer(f,delimiter=';').writerow(row)

# ──────────────────────────────────────────────────────────────────────────────
# Socket handshake   (Unity already connected on port 56001)
# ──────────────────────────────────────────────────────────────────────────────
HOST, PORT = '', 56001
srv = socket.socket()
srv.bind((HOST, PORT)); srv.listen(1)
conn, _ = srv.accept()

seg = conn.recv(4096).decode().split('_')

h = list(map(int, seg[0].split(',')))
(BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES,
 N_ITERATIONS, MC_SAMPLES, N_INITIAL, SEED,
 PROBLEM_DIM, NUM_OBJS, TRAIT_DIM) = h

parameters_info = [list(map(float, r.split(',')))
                   for r in seg[2].split('/')[:PROBLEM_DIM]]
objectives_info = [list(map(float, r.split(',')))
                   for r in seg[2].split('/')[PROBLEM_DIM:]]

parameter_names  = seg[3].split(',')
objective_names  = seg[4].split(',')
trait_names      = seg[5].split(',')
USER_ID, CONDITION_ID, GROUP_ID = seg[6].split(',')[:3]

bounds_x  = torch.zeros(2, PROBLEM_DIM, **tkwargs); bounds_x[1] = 1.
ref_point = torch.tensor([-1.]*NUM_OBJS, **tkwargs)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def scale_obj(v, lo, hi, inv):
    n = (v-lo)/(hi-lo)*2 - 1
    return max(-1, min(1, -n if inv else n))

# ──────────────────────────────────────────────────────────────────────────────
# Objective function (sync Unity call)
# ──────────────────────────────────────────────────────────────────────────────
def objective_fn(x_tensor):
    x_np = x_tensor.cpu().numpy()
    x_dn = [parameters_info[i][0] + x_np[i]*(parameters_info[i][1]-parameters_info[i][0])
            for i in range(PROBLEM_DIM)]
    conn.sendall(("parameters," + ','.join(f"{v:.3f}" for v in x_dn)).encode())

    vals = list(map(float, conn.recv(2048).decode().split(',')))
    obj_raw, trait_raw = vals[:NUM_OBJS], vals[NUM_OBJS:]

    global TRAIT_VEC
    if TRAIT_VEC is None:                      # first call
        TRAIT_VEC = torch.tensor(trait_raw, **tkwargs)

    obj_norm = torch.tensor(
        [scale_obj(obj_raw[i], *objectives_info[i]) for i in range(NUM_OBJS)],
        **tkwargs)
    return obj_norm

# ──────────────────────────────────────────────────────────────────────────────
# Initial Sobol data
# ──────────────────────────────────────────────────────────────────────────────
def initial_data(n):
    xs, ys = [], []
    sob = draw_sobol_samples(bounds=bounds_x, n=1, q=n,
                             seed=torch.randint(1_000_000,(1,)).item()).squeeze(0)
    for x in sob:
        ys.append(objective_fn(x)); xs.append(x)
    X = torch.cat([torch.stack(xs), TRAIT_VEC.repeat(n,1)], dim=1)
    Y = torch.stack(ys)
    return X, Y

# ──────────────────────────────────────────────────────────────────────────────
# GP builder (loads global hyper-parameters)
# ──────────────────────────────────────────────────────────────────────────────
def build_model(X, Y):
    prior_state = pickle.load(open(PRIOR_FILE,'rb'))
    models=[]
    for m in range(NUM_OBJS):
        mdl = LinearMixedGP(X, Y[:,m].unsqueeze(-1),
                            random_effects=None).to(device)
        mdl.load_state_dict({k:v for k,v in prior_state[m].items()
                             if k in mdl.state_dict()}, strict=False)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(mdl.likelihood, mdl))
        models.append(mdl)
    return ModelListGP(*models)

# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
def run():
    proj = mkdir(os.path.join(LOG_DIR, USER_ID))
    obs_csv = os.path.join(proj, "obs.csv")
    hv_csv  = os.path.join(proj, "hv.csv")
    time_csv= os.path.join(proj, "time.csv")
    init_csv(obs_csv, objective_names, trait_names, parameter_names)

    X, Y = initial_data(N_INITIAL)
    for i in range(N_INITIAL):
        row_csv(obs_csv, np.concatenate((
            [USER_ID, CONDITION_ID, GROUP_ID, time.strftime("%F %T"),
             i+1, "sample", "F"],
            Y[i].cpu(), TRAIT_VEC.cpu(), X[i, :PROBLEM_DIM].cpu())) )

    model = build_model(X, Y)
    hv = Hypervolume(ref_point=ref_point)
    row_csv(hv_csv, [0, hv.compute(Y[is_non_dominated(Y)])])

    sampler = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]), seed=SEED)
    trait_idx = list(range(PROBLEM_DIM, PROBLEM_DIM + TRAIT_DIM))

    for it in range(1, N_ITERATIONS+1):
        t0 = time.time()

        acq_base = qExpectedHypervolumeImprovement(
            model=model,
            sampler=sampler,
            ref_point=ref_point.tolist(),
            partitioning=NondominatedPartitioning(ref_point, Y))

        acq = FixedFeatureAcquisitionFunction(
            acq_base,
            d=PROBLEM_DIM + TRAIT_DIM,
            indices=trait_idx,
            values=TRAIT_VEC.unsqueeze(0))

        cand, _ = optimize_acqf(acq, bounds=bounds_x, q=1,
                                num_restarts=NUM_RESTARTS,
                                raw_samples=RAW_SAMPLES)

        y_new = objective_fn(cand[0])
        x_new = torch.cat([cand[0], TRAIT_VEC]).unsqueeze(0)
        X = torch.cat([X, x_new]);  Y = torch.cat([Y, y_new.unsqueeze(0)])

        row_csv(obs_csv, np.concatenate((
            [USER_ID, CONDITION_ID, GROUP_ID, time.strftime("%F %T"),
             N_INITIAL+it, "optim", "F"],
            y_new.cpu(), TRAIT_VEC.cpu(), cand[0].cpu())) )

        hv_val = hv.compute(Y[is_non_dominated(Y)])
        row_csv(hv_csv, [it, hv_val])
        row_csv(time_csv, [it, f"{time.time()-t0:.3f}"])

        model = build_model(X, Y)
        print(f"Iter {it}/{N_ITERATIONS}  HV={hv_val:.4f}", flush=True)

    conn.sendall(b"optimization_finished,")
    print("Finished.")

if __name__ == "__main__":
    run()
