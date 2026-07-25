# meta_train.py — build Meta-TAF population models from completed BOforUnity runs.
#
# Offline tool (run by the researcher, never by participants). It converts each finished
# run's ObservationsPerEvaluation.csv into one "source artifact" pair understood by the
# Meta-TAF backend / openbo:
#
#     <out>/trajectories/<name>.json   x_values (n,d) in [0,1]^d, y_values (n,M) in
#                                      [-1,1] maximization, pareto_front (P,M)
#     <out>/gp_states/<name>.json      per-objective GP hyperparameters + the study
#                                      frame + provenance
#
# Everything is normalized through bo_normalize (the exact transform the live backends
# apply), and the study frame is stamped into every artifact so the runtime can refuse
# sources whose bounds or minimize flags do not match the live study -- a mismatch there
# would silently transfer a rescaled or inverted response surface.
#
# Usage:
#   python meta_train.py --frame frame.json --out ../MetaSources/MyStudy runDir1 runDir2 ...
#
# frame.json mirrors the Unity configuration:
#   {
#     "parameters": [{"key": "speed", "low": 0.0, "high": 10.0}, ...],
#     "objectives": [{"key": "comfort", "low": 0.0, "high": 100.0, "minimize": 0}, ...]
#   }
#
# Requires the full stack (openbo + botorch); see docs/meta-taf-student-guide.md.

import argparse
import datetime
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import bo_normalize
import meta_fingerprint

# GP mean vs data sanity threshold (normalized objective units, range is [-1, 1]).
FIT_RESIDUAL_WARN = 0.3


def _import_stack():
    try:
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood

        from openbo.optimizers.mobo_botorch import pareto_front
        from openbo.optimizers.mobo_taf import _load_mo_source_surrogates
    except ImportError as e:
        raise SystemExit(
            f"meta_train.py needs torch/botorch/gpytorch and the openbo fork installed ({e}).\n"
            "Install openbo with:\n"
            "    python -m pip install \"open-bo @ git+https://github.com/M-Colley/openbo@main\"\n"
            "or, from a local clone:  python -m pip install -e path/to/openbo"
        )
    return (torch, fit_gpytorch_mll, SingleTaskGP, Standardize, MaternKernel,
            ScaleKernel, ExactMarginalLogLikelihood, pareto_front, _load_mo_source_surrogates)


def load_frame(frame_path):
    with open(frame_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("parameters") or []
    objs = payload.get("objectives") or []
    if not params or not objs:
        raise SystemExit("frame.json must define non-empty 'parameters' and 'objectives'.")
    parameter_names = [str(p["key"]) for p in params]
    parameters_info = [(float(p["low"]), float(p["high"])) for p in params]
    objective_names = [str(o["key"]) for o in objs]
    objectives_info = [(float(o["low"]), float(o["high"]), int(o.get("minimize", 0))) for o in objs]
    if len(objective_names) < 2:
        raise SystemExit("Meta-TAF sources are multi-objective: define at least 2 objectives.")
    frame = meta_fingerprint.canonical_frame(
        parameter_names, parameters_info, objective_names, objectives_info
    )
    return frame, parameter_names, parameters_info, objective_names, objectives_info


def sanitize_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_.")
    return cleaned or "source"


def derive_name(run_path, index):
    parts = [p for p in os.path.normpath(os.path.abspath(run_path)).split(os.sep) if p]
    tail = "_".join(parts[-3:]) if len(parts) >= 3 else "_".join(parts)
    return sanitize_name(f"{index:02d}_{tail}")


def read_run_csv(run_path):
    csv_path = run_path
    if os.path.isdir(run_path):
        csv_path = os.path.join(run_path, "ObservationsPerEvaluation.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No ObservationsPerEvaluation.csv under: {run_path}")
    return pd.read_csv(csv_path, delimiter=";")


def extract_normalized_xy(df, parameter_names, parameters_info, objective_names, objectives_info):
    """Columns by NAME (never position), raw units -> the canonical frame."""
    if "Context" in df.columns:
        raise ValueError(
            "This run contains a 'Context' column (contextual optimization). Contextual "
            "runs cannot become Meta-TAF sources: the context dimension has no home in a "
            "context-free source model."
        )
    missing = [c for c in parameter_names + objective_names if c not in df.columns]
    if missing:
        raise ValueError(f"Run CSV is missing required column(s): {missing}")

    x_raw = df[parameter_names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    y_raw = df[objective_names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if x_raw.shape[0] < 3:
        raise ValueError(f"Run has only {x_raw.shape[0]} rows; need at least 3 to fit a GP.")
    if not (np.all(np.isfinite(x_raw)) and np.all(np.isfinite(y_raw))):
        raise ValueError("Run CSV contains NaN/Inf values.")

    x_unit = np.zeros_like(x_raw)
    for j, (lo, hi) in enumerate(parameters_info):
        x_unit[:, j] = bo_normalize.normalize_param_column(x_raw[:, j], lo, hi)
    y_norm = np.zeros_like(y_raw)
    for j, (lo, hi, minflag) in enumerate(objectives_info):
        # ObservationsPerEvaluation.csv stores raw units by construction.
        y_norm[:, j] = bo_normalize.normalize_obj_column(y_raw[:, j], lo, hi, minflag, fmt="raw")
    return x_unit, y_norm


def fit_per_objective_hyperparameters(x_unit, y_norm, stack):
    """Fit one GP per objective and export hyperparameters the loader can replay.

    The kernel structure (ScaleKernel(Matern52) + Standardize outcome transform) is
    exactly what openbo's loader reconstructs, so exporting these values and replaying
    them is a faithful round trip by construction.
    """
    (torch, fit_gpytorch_mll, SingleTaskGP, Standardize, MaternKernel,
     ScaleKernel, ExactMarginalLogLikelihood, _, _) = stack

    x_t = torch.tensor(x_unit, dtype=torch.double)
    entries = []
    d = x_unit.shape[1]
    for j in range(y_norm.shape[1]):
        y_t = torch.tensor(y_norm[:, j:j + 1], dtype=torch.double)
        gp = SingleTaskGP(
            x_t, y_t,
            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        entries.append({
            "kernel_type": "matern52",
            "lengthscale": [float(v) for v in
                            gp.covar_module.base_kernel.lengthscale.detach().reshape(-1)],
            "variance": float(gp.covar_module.outputscale.detach()),
            "noise": float(gp.likelihood.noise.detach().reshape(-1)[0]),
            "standardize_targets": True,
            "optimize_noise": True,
        })
    return entries


def write_artifact(out_dir, name, x_unit, y_norm, gp_entries, frame, run_path, stack):
    (_, _, _, _, _, _, _, pareto_front, load_sources) = stack
    traj_dir = os.path.join(out_dir, "trajectories")
    gp_dir = os.path.join(out_dir, "gp_states")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(gp_dir, exist_ok=True)

    trajectory = {
        "x_values": x_unit.tolist(),
        "y_values": y_norm.tolist(),
        "pareto_front": pareto_front(y_norm).tolist(),
    }
    gp_payload = {
        "gp_state": {"objectives": gp_entries},
        "frame": frame,
        "provenance": {
            "schema_version": 1,
            "source_type": "human",
            "y_calibration": "measured",
            "generator": "Assets/StreamingAssets/BOData/BayesianOptimization/meta_train.py",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_path": os.path.abspath(str(run_path)),
            "n_observations": int(x_unit.shape[0]),
        },
    }
    with open(os.path.join(traj_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(trajectory, f)
    with open(os.path.join(gp_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(gp_payload, f, indent=1)

    # Self-check: reload through the SAME loader the runtime uses and compare the
    # replayed posterior mean against the data it was fitted on. A large residual means
    # the artifact does not reproduce the run it claims to represent.
    sources = load_sources(out_dir, expected_m=y_norm.shape[1], expected_d=x_unit.shape[1])
    match = [s for s in sources if s.name == name]
    if not match:
        raise RuntimeError(f"Self-check failed: artifact '{name}' did not load back.")
    mu = match[0].posterior_mean(x_unit)
    residual = float(np.max(np.abs(mu - y_norm)))
    gp_payload["provenance"]["fit_residual"] = residual
    with open(os.path.join(gp_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(gp_payload, f, indent=1)
    return residual


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert completed BOforUnity runs into Meta-TAF population models."
    )
    parser.add_argument("runs", nargs="+",
                        help="Run folders (containing ObservationsPerEvaluation.csv) or CSV paths.")
    parser.add_argument("--frame", required=True,
                        help="frame.json describing the study's parameters and objectives.")
    parser.add_argument("--out", required=True,
                        help="Output MetaSources directory (gp_states/ + trajectories/ are created).")
    parser.add_argument("--names", nargs="*", default=None,
                        help="Optional explicit artifact names (one per run).")
    args = parser.parse_args(argv)

    if args.names is not None and len(args.names) != len(args.runs):
        raise SystemExit(f"--names got {len(args.names)} names for {len(args.runs)} runs.")

    stack = _import_stack()
    frame, parameter_names, parameters_info, objective_names, objectives_info = load_frame(args.frame)
    print(f"Frame digest: {meta_fingerprint.frame_digest(frame)} "
          f"(d={frame['d']}, M={frame['M']})")

    written = []
    for i, run in enumerate(args.runs):
        name = sanitize_name(args.names[i]) if args.names else derive_name(run, i)
        try:
            df = read_run_csv(run)
            x_unit, y_norm = extract_normalized_xy(
                df, parameter_names, parameters_info, objective_names, objectives_info
            )
            gp_entries = fit_per_objective_hyperparameters(x_unit, y_norm, stack)
            residual = write_artifact(args.out, name, x_unit, y_norm, gp_entries, frame, run, stack)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"[SKIP] {run}: {e}")
            continue
        note = ""
        if residual > FIT_RESIDUAL_WARN:
            note = (f"  <-- WARNING: fit residual {residual:.3f} > {FIT_RESIDUAL_WARN}; "
                    "this model reproduces its own run poorly (very noisy data?)")
        print(f"[OK]   {name}: n={x_unit.shape[0]}, fit residual {residual:.4f}{note}")
        written.append(name)

    if not written:
        raise SystemExit("No artifacts were written.")
    print(f"\nWrote {len(written)} population model(s) to {os.path.abspath(args.out)}")
    print("Point the BoForUnityManager 'Meta Source Dir' at this folder (or copy it into "
          "Assets/StreamingAssets/BOData/MetaSources/).")


if __name__ == "__main__":
    main()
