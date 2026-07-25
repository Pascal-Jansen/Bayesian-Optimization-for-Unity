# Meta-BO in BOforUnity — Student Guide (MetaTAF backend)

*BOforUnity v1.6.0. For questions, start with section 7 (troubleshooting) and section 9
(what to cite).*

## 1. What this does, in plain language

Ordinary Bayesian optimization starts every participant **from zero**: the first
iterations are spent sampling the design space just to get oriented, which costs precious
human trials. But if ten people already went through your study, their data collectively
says a lot about *where good designs live*.

The **MetaTAF** backend uses that knowledge. Before your study, you convert completed runs
into **population models** — one small Gaussian-process surrogate per prior participant.
During a new participant's session, the optimizer blends:

* the **current user's own model** (exactly the multi-objective `qLogNEHVI` optimizer the
  BoTorch backend uses), and
* each **population model's opinion** of how much a candidate design would improve on what
  that prior participant achieved (a hypervolume-improvement score).

Population models that *agree* with the current user's observed data keep their influence;
models that disagree are automatically down-weighted (that is the "TAF" part — Transfer
Acquisition Function). On top of that, a **decay schedule** shifts control to the current
user as their own data accumulates, and with **zero** population models the backend is
plain multi-objective BO. So enabling MetaTAF is always safe — it can only add
information, and it lets go of that information when it stops matching the person in
front of you.

This is the multi-objective (Pareto/hypervolume) counterpart of the meta-BO approach that
Liao et al. (CHI 2024) showed cuts calibration to a handful of trials in wrist-input
studies. The optimizer itself lives in the [openbo](https://github.com/M-Colley/openbo)
Python package; BOforUnity talks to it through the same socket protocol as every other
backend.

## 2. Requirements

* BOforUnity set up and working normally (any backend) — see the main README first.
* **At least 2 objectives** (this backend is multi-objective only).
* The **openbo** package installed for the same Python that BOforUnity launches:

```bash
python -m pip install "open-bo @ git+https://github.com/M-Colley/openbo@main"
```

  or, if you have a local clone of the fork:

```bash
python -m pip install -e path/to/openbo
```

  openbo's own dependencies (botorch, torch, ...) are already satisfied by BOforUnity's
  `requirements.txt` versions. If the backend starts without openbo it exits immediately
  with the exact install command above — nothing hangs.

* Not compatible with **Warm Start** or **Contextual Optimization** (both are rejected
  with a clear error; see section 8).

## 3. Workflow overview

```
Step 1  Collect prior runs        Step 2  Build population models     Step 3  Run new users
------------------------------    --------------------------------    -----------------------
Run participants with the         python meta_train.py                Backend = MetaTAF in the
plain BoTorch backend             --frame frame.json                  BoForUnityManager
(mobo.py, same objectives         --out .../MetaSources               inspector. New users now
and parameters!)                  LogData/.../run ...                 start from the population.
```

### Step 1 — Collect prior runs

Run a pilot cohort with the **BoTorch** backend and *exactly the study configuration you
will use later*: same parameter names and bounds, same objective names, bounds, and
minimize flags. Each completed run leaves an `ObservationsPerEvaluation.csv` under
`Assets/StreamingAssets/BOData/LogData/<user>/<condition>/run*/`.

More iterations per pilot participant = better population models. As a rule of thumb,
aim for at least ~15 evaluations per run; `meta_train.py` refuses runs with fewer than 3.

### Step 2 — Build population models with `meta_train.py`

First describe your study in a small `frame.json` (copy the values from your
BoForUnityManager configuration):

```json
{
  "parameters": [
    {"key": "speed", "low": 0.0, "high": 10.0},
    {"key": "gap",   "low": 1.0, "high": 5.0}
  ],
  "objectives": [
    {"key": "comfort",  "low": 0.0, "high": 100.0, "minimize": 0},
    {"key": "duration", "low": 0.0, "high": 60.0,  "minimize": 1}
  ]
}
```

Then convert the pilot runs (any machine with the full Python stack + openbo):

```bash
cd Assets/StreamingAssets/BOData/BayesianOptimization
python meta_train.py --frame frame.json --out ../MetaSources ^
    ../LogData/p01/main/run ../LogData/p02/main/run ../LogData/p03/main/run
```

For every run this fits one GP per objective, normalizes everything into the optimizer's
internal space, stamps the frame into the artifact, and **self-checks** that the written
artifact reproduces its own run (the `fit residual` it prints; values ⪅ 0.15 are typical,
a warning appears above 0.3). Output:

```
MetaSources/
  gp_states/00_..._run.json      hyperparameters + frame + provenance
  trajectories/00_..._run.json   normalized observations + Pareto front
```

### Step 3 — Run new participants

1. In the `BoForUnityManager` inspector, set **Backend = MetaTAF**.
2. Leave **Meta Source Dir** at `MetaSources` (or point it at your folder; relative paths
   resolve against `StreamingAssets/BOData/`).
3. Press play. The backend log lists which population models were accepted:
   `Meta-TAF: using 3 population model(s): [...]`.

That's it — the participant experience is identical to a normal run.

## 4. Inspector settings (defaults are sensible)

| Setting | Default | Meaning |
|---|---|---|
| Meta Source Dir | `MetaSources` | Folder holding `gp_states/` + `trajectories/`. |
| Meta Weight Mode | `TafR` | How population models are weighted. `TafR`: by Pareto-ranking agreement with the current user's observations (recommended — this is the negative-transfer guard). `TafM`: by meta-feature similarity. |
| Meta Rho | `1.0` | Bandwidth of the weighting kernel. Smaller = stricter (disagreeing sources are dropped sooner). |
| Meta Target Weight | `1.0` | Weight of the current user's own model in the blend. |
| Meta Warmup Iters | `1` | The first *k* optimization suggestions follow the population models alone (the user's own model has too little data to say anything yet). Keep small — on a 10-iteration budget, 1 is a good default. |
| Meta Decay Start Iter | `2` | Iteration after which population influence starts to fade (d1 in Liao et al.'s decay). |
| Meta Decay Rate | `0.3` | How fast it fades per iteration (d2). `0` = never fade. With the defaults (2, 0.3), population influence is gone after iteration 5 and the run finishes fully personalized. |

The remaining hyperparameters (restarts, raw samples, MC samples, seed, iteration counts)
are the shared ones described in README 8.10/8.11 and apply unchanged.

## 5. What gets logged

Everything a normal multi-objective run logs (`ObservationsPerEvaluation.csv` with
`IsPareto`, `HypervolumePerEvaluation.csv`, `ExecutionTimes.csv`), plus:

* **`MetaWeightsPerEvaluation.csv`** — one row per optimization iteration:
  `Iteration; TargetWeight; DecayFactor; <one column per population model>`.
  Read it as "who was steering": `TargetWeight = 0.0` marks warmup iterations driven by
  the population alone; the per-source columns show TAF weights after decay. If one source
  dominates every participant, your population may be too homogeneous — if weights differ
  a lot between participants, TAF is doing its job selecting matching predecessors.
* **`MetaSourcesUsed/`** — an exact copy of the population models this run used. This is
  your provenance record: even if the shared `MetaSources` folder changes later, every
  run archives what it actually saw.

## 6. Study-design guidance (read before running a real experiment)

* **Freeze the population.** If Meta-BO is a condition in your experiment, build the
  population models from a *pilot* cohort, freeze the folder, and give every analyzed
  participant the identical set. If you instead keep adding each finished participant as
  a new source, participant N's treatment depends on participants 1..N-1 — an ordering
  confound that breaks independence assumptions in your analysis.
* **Compare against a no-transfer control.** The honest baseline for "Meta-BO helped" is
  the same study with the BoTorch backend (or MetaTAF with an empty source folder, which
  is the same optimizer). Liao et al. (CHI 2024) is the template for this comparison.
* **Population size:** their study used 14 population models; simulations showed benefits
  from as few as a handful, with earlier convergence as the population grows.
* **Same frame, always.** Sources are only accepted when parameter/objective names,
  bounds, and minimize flags match the live study exactly. Changing an objective's range
  or direction mid-study invalidates your population models — regenerate them.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `The Meta-TAF backend needs the 'openbo' package` | Install openbo **for the Python BOforUnity uses** (README 8.5 shows which one that is): `python -m pip install "open-bo @ git+https://github.com/M-Colley/openbo@main"` |
| `source '<name>' was built for a different study frame; skipping: ...` | The artifact was generated for different names/bounds/minimize flags. The message lists the exact field. Regenerate with a matching `frame.json`. |
| `source '<name>' carries no frame block; skipping` | The artifact predates frame stamping or was hand-built. Regenerate with `meta_train.py` (or set the env var `BO_META_ALLOW_UNFRAMED=1` if you are absolutely sure). |
| `Meta-TAF: no valid population models found` | Check Meta Source Dir path and that `gp_states/` + `trajectories/` contain paired `.json` files. The run still works (plain MOBO). |
| Backend log stops right after `using N population model(s)` | Stale PyTorch JIT lock from a previously killed run. Current builds isolate this per-process; if you ever see it, delete `%LOCALAPPDATA%\torch_extensions` and restart. |
| Suggestions feel slow | Per-iteration optimization cost grows with the number of population models (measured: ~5 s with 0 sources to ~17 s with 14 sources at study-quality settings, machine-dependent). Cap the population folder to the most relevant sources if needed. |
| `MetaTAF does not support Warm Start` / contextual error | By design — see section 8. |

## 8. Current limitations

* **Multi-objective only** (≥ 2 objectives). Single-objective Meta-BO is available in the
  openbo package (`bo_taf`) but not wired into Unity.
* **No Warm Start.** Population models are the transfer mechanism; mixing both would
  double-count prior data.
* **No Contextual Optimization (LCE-M).** Combining per-context embeddings with
  population-model transfer is a genuinely open design question (which context's data may
  enter a source? does transfer happen within or across contexts?) — deliberately not
  shipped half-baked. The architecture keeps the seam open: sources are context-free
  surfaces over the design space, and the runtime already validates context configuration
  separately, so a future version can add e.g. per-context source pools without breaking
  today's artifacts.
* Population models are treated as **fixed** during a run (their GPs are not re-fitted).

## 9. What to cite

If you publish results obtained with this backend, cite the method lineage and the
implementation:

* **TAF (the transfer mechanism):** M. Wistuba, N. Schilling, L. Schmidt-Thieme.
  *Scalable Gaussian process-based transfer surrogates for hyperparameter optimization.*
  Machine Learning 107(1), 2018.
* **Meta-BO for HCI calibration (the approach this backend operationalizes):** Y.-C. Liao,
  R. Desai, A. M. Pierce, K. E. Taylor, H. Benko, T. R. Jonker, A. Gupta. *A Meta-Bayesian
  Approach for Rapid Online Parametric Optimization for Wrist-based Interactions.*
  CHI 2024. https://doi.org/10.1145/3613904.3642071
* **openbo (the optimizer implementation):** https://github.com/M-Colley/openbo (MIT,
  © Yi-Chi Liao; fork of https://github.com/yichiliao/openbo with correctness fixes and
  the multi-objective TAF-EHVI optimizers).
* **BOforUnity itself:** see README section 12.

Method summary for your paper's notation: the acquisition blends the current user's
`qLogNEHVI` with per-source hypervolume-improvement terms computed from each population
model's posterior mean against that model's own Pareto front, combined in log space as a
weighted average; source weights come from Pareto-ranking agreement (TAF-R) or
meta-feature similarity (TAF-M), multiplied by the Liao-et-al. decay γ(t) with
hyperparameters (d1, d2) = (Meta Decay Start Iter, Meta Decay Rate).
