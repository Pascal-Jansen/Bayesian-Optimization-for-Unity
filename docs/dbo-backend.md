# DBO backend (Dynamic Bayesian Optimization)

Single-objective Bayesian optimization for a cost that **drifts while the study
runs** — participant adaptation, learning effects, fatigue, habituation. Where
the BoTorch backend assumes the same design yields the same response all
session, DBO multiplies the GP covariance by a temporal decay factor
`alpha^|t-t'|` and fits the decay rate `alpha` from the data by marginal
likelihood, so the model infers how fast the participant is changing instead of
assuming they are not.

Method: Kim & Sergi, *Dynamic Bayesian optimization for non-stationary systems*,
Comput. Methods Biomech. Biomed. Eng., 2025 (doi 10.1080/10255842.2025.2595150)
and *Validation of Dynamic Bayesian Optimization for a Non-Stationary
Human-in-the-Loop Optimization Problem*, IEEE RA-L 11(5), 2026
(doi 10.1109/LRA.2026.3665072). The Python implementation is the BSD-3-Clause
[dbo-torch](https://github.com/M-Colley/dbo-torch) package, vendored under
`Assets/StreamingAssets/BOData/BayesianOptimization/dbo_torch/`.

## Selecting the backend

In the `BoForUnityManager` inspector set **Backend → DBO**. DBO is
single-objective: configure exactly one objective (the inspector warns
otherwise, and the backend refuses to start). Contextual optimization (LCE-M)
is not available with DBO, same as with CABOP and MetaTAF.

The launcher runs `dbo.py`, which drives `dbo_runtime.py` — the same protocol,
CSV outputs and log layout as `bo.py`, so existing analysis scripts keep
working.

## Settings (inspector, DBO section)

| Setting | Default | Meaning |
|---|---|---|
| DBO Spatial Kernel | Rbf | Covariance over design parameters. Rbf matches the reference DBO implementation. |
| DBO Alpha Parameterization | Decay | How alpha is fitted. Decay reproduces the reference; Direct behaves better under fast drift. |
| DBO Initial Alpha | 0.99 | Starting decay rate before fitting. |
| DBO Exploration Ratio | 0.1 | Re-search with inflated variance when the acquisition collapses onto an already-certain point. 0 disables. |
| DBO Acquisition Time Offset | 0 | 0 scores candidates at the current time (reference behaviour); 1 scores at the time they will actually be evaluated. |
| DBO Validation Every | 0 (off) | Every N iterations apply the model's best estimate instead of an exploratory point, making optimisers comparable across conditions. |
| DBO Validation Confidence | 0.01 | Tail mass of the validation upper confidence bound (0.01 ≈ mean + 2.33 sd). |
| DBO Validation Visited Only | on | Restrict validation candidates to already-tried designs (reference behaviour). |
| DBO Stationary Baseline | off | Pin alpha = 1 — plain stationary BO with everything else identical. The ablation/control condition. |

For a DBO-vs-BO study, run the same protocol twice with only **Stationary
Baseline** toggled: the Sobol sampling points are identical for matching
config, so conditions differ only in whether the model may discount the past.

## How time works

The backend's time coordinate is the global evaluation index: sampling
iterations occupy t = 1…N, optimization continues at N+1 with no reset.
Validation iterations only begin after sampling, since there is no model to
validate before that, but they are scheduled on the global index.

**Warm start** works, with one modelling caveat: imported rows carry no
timestamps, so they are replayed as t = 1…k and the live run continues at k+1.
The decay kernel then treats last week's session as if it ended moments before
this one began. If the participant plausibly changed between sessions, that is
an assumption you are making implicitly; stationary BO has no equivalent
exposure because it does not care when anything happened.

## Outputs

Standard files unchanged: `ObservationsPerEvaluation.csv`,
`BestObjectivePerEvaluation.csv`, `HypervolumePerEvaluation.csv` (legacy
mirror), `ExecutionTimes.csv`. One DBO-specific addition:

```
DboDiagnosticsPerEvaluation.csv
  Iteration;Phase;IsValidation;Alpha;PredictedCost;PredictedSd;
  ObservedCost;ObservedObjective;SuggestSeconds
```

`Alpha` (blank during sampling, also printed to the Unity console live) is the
first thing to look at after a run: **near 1.0 throughout means the objective
did not measurably drift and the BoTorch backend would have done the same job.**
`PredictedCost` vs `ObservedCost` on validation rows measures model accuracy
independently of exploration luck — the quantity that separates DBO from BO as
a session progresses.

`coverage` keeps its usual meaning (best observed objective so far, normalized
frame) for comparability with all other backends; under drift, prefer the
diagnostics file for interpretation, since an early "best" may no longer be
attainable.

## Verifying the backend

```bash
python tests/dbo_protocol_check.py
```

launches the real backend and drives it with a mock Unity client speaking the
wire protocol byte for byte, including messages split across TCP writes. ~15s,
needs torch/botorch installed. Deliberately not part of `unittest discover`.

## Updating the vendored dbo_torch package

The backend imports a vendored snapshot rather than a pip install, because the
bundled Python runtime may lack git and `dbo-torch` is not on PyPI; the backend
puts its own folder first on `sys.path`, so the snapshot always wins. To
refresh it after an upstream change, copy from a dbo-torch checkout:

```
dbo-torch/src/dbo_torch/{__init__,kernels,model,optimizer,mo_optimizer}.py
dbo-torch/LICENSE
  -> Assets/StreamingAssets/BOData/BayesianOptimization/dbo_torch/
```

Do **not** copy `unity_bridge.py` — it implements a different, incompatible
socket protocol. Re-run `tests/dbo_protocol_check.py` after refreshing.
