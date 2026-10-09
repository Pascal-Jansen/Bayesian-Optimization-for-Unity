# Meta-BO in BOforUnity — Student Guide (MetaTAF backend)

*Written for BOforUnity v1.8. For questions, start with section 7 (troubleshooting) and
section 9 (what to cite).*

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
user as their own data accumulates, so the run finishes on the participant's own model.
Weighting and decay limit — but do not remove — the risk of a population that does not
fit the person in front of you: the first transfer iterations can still be pulled in a
wrong direction. And if no population model can be loaded at all, the run refuses to
start (see **Meta Require Sources**) instead of silently running plain multi-objective BO.

This is the multi-objective (Pareto/hypervolume) counterpart of the meta-BO approach that
Liao et al. (CHI 2024) showed cuts calibration to a handful of trials in wrist-input
studies. The optimizer itself lives in the [openbo](https://github.com/M-Colley/openbo)
Python package; BOforUnity talks to it through the same socket protocol as every other
backend.

## 2. Requirements

* BOforUnity set up and working normally (any backend) — see the main README first.
* **At least 2 objectives** (this backend is multi-objective only).
* The **openbo** package installed for the same Python that BOforUnity launches. That is
  BOforUnity's private environment (README 8.5), not your system Python: the Unity Console
  logs its interpreter at startup as `Optimizer Python: …`; use that path in place of
  `python` in the commands below (`"<that path>" -m pip install …`). The backend's error
  messages print the complete command with that interpreter.

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

  Already installed openbo earlier? The backend requires the **2026-08 TAF-R rework**
  (objective-wise ranking weights, see section 4) and refuses older installs at startup
  with the exact upgrade command:

```bash
python -m pip install --force-reinstall --no-deps "open-bo @ git+https://github.com/M-Colley/openbo@main"
```

  (`--force-reinstall` because openbo's version number did not change, so a plain
  `--upgrade` reports "already satisfied" and does nothing; `--no-deps` keeps your
  pinned torch/botorch stack untouched.)

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

Then convert the pilot runs (any machine with the full Python stack + openbo; on the Unity
machine that is the `Optimizer Python` interpreter from section 2, so use its path in place
of `python`):

```bash
cd Assets/StreamingAssets/BOData/BayesianOptimization
python meta_train.py --frame frame.json --out ../MetaSources ^
    --source-type human --y-calibration measured ^
    ../LogData/p01/main/run ../LogData/p02/main/run ../LogData/p03/main/run
```

`--source-type` and `--y-calibration` are **required provenance stamps** written into
every artifact (and copied into each run's `MetaSourcesUsed/` audit trail): who produced
the objective values (`human`, `llm-persona`, `synthetic`) and whether they were
`measured` from real participants/systems or `generated` by a model. Label them honestly
— a false "human/measured" stamp poisons the audit trail of every study using the source.

Every objective in `frame.json` needs its `minimize` flag (`0` or `1`); the tool refuses
a frame without it instead of assuming "maximize".

For every run this fits one GP per objective, normalizes everything into the optimizer's
internal space, stamps the frame into the artifact, and **self-checks** — before anything
is written — that the artifact reproduces its own run (the `fit residual` it prints;
values ⪅ 0.15 are typical, a warning appears above 0.3). A run that fails the check, or
whose parameter values lie outside the bounds in `frame.json` (i.e. it was recorded with
different bounds), is skipped with the reason and leaves no artifact behind. Output:

```
MetaSources/
  gp_states/p01_main_run.json      hyperparameters + frame + provenance
  trajectories/p01_main_run.json   normalized observations + Pareto front
  population.json                  the population manifest (see below)
```

* **Names** come from the last three folders of the run path
  (`<participant>_<condition>_<run>`), never from the order of the command line, so
  rebuilding the folder with one more participant adds exactly one source. Two runs with
  the same path tail (e.g. from two studies) get a short path hash appended
  (`p01_main_run-1a2b3c4d`). Explicit names: `--name p01 --name p02` (one per run, in
  order; `--name p01 run1 --name p02 run2` works too) or `--names p01,p02`.
* **Existing artifacts are kept**: a run already in `--out` is reported as `[KEEP]` and not
  rebuilt; add `--force` to rebuild it (this changes the population). It keeps the name it
  has there, also an index-prefixed `00_p01_main_run` from an older `meta_train.py` (which
  numbered sources by command-line position), and also when it was built from another path
  (on another machine, or before the project folder moved: the same path tail with
  identical observations), so rebuilding a folder with one more participant adds that
  participant only. If the same run is in the folder under *two* names (or you give it
  another one with `--name`), the tool warns: delete one pair, or that participant counts
  twice in every later run.
* **`--dry-run`** checks the frame, every run and the names, and prints what would be
  written — without fitting or writing anything (no torch/openbo needed).
* **`population.json`** is rewritten after every build: the name and SHA-256 hashes of every
  source pair in the folder plus the frame digest (line endings do not count, so a git
  checkout that converts them, e.g. `core.autocrlf` on Windows, is not a change). When it
  is present, the backend refuses to start if the population it loads differs from it in
  any way (a source added, removed, replaced, or no longer loadable) — this enforces
  "freeze the population" (section 6).
  After changing the folder by hand *before* the study, rewrite it with
  `python meta_train.py --frame frame.json --out ../MetaSources --manifest-only`.
* Copy the folder as a whole. Copies through exFAT/FAT32 drives or network shares can add
  macOS `._<name>.json` metadata files; the backend ignores them (listed as skipped).

### Step 3 — Run new participants

1. In the `BoForUnityManager` inspector, set **Backend = MetaTAF**.
2. Leave **Meta Source Dir** at `MetaSources` (or point it at your folder; relative paths
   resolve against `StreamingAssets/BOData/`).
3. Press play. The backend log lists the population models the optimizer actually loaded:
   `Meta-TAF: using 3 population model(s): [...]`. A source that matches the study frame
   but cannot be used is named with the reason and left out. If **none** is loaded, the
   run aborts before the first trial with the per-source reasons (see **Meta Require
   Sources** below) instead of silently continuing as plain multi-objective BO. If the
   folder has a `population.json` and the loaded population differs from it, the run also
   aborts before the first trial and lists the differences.
4. Use a **Random Seed ≥ 0** (the same in every condition). MetaTAF refuses negative seeds
   at startup: openbo cannot use them, and mapping them to another value would also change
   the initial Sobol design, which must equal the BoTorch condition's.

That's it — the participant experience is identical to a normal run.

## 4. Inspector settings (defaults are sensible)

| Setting | Default | Meaning |
|---|---|---|
| Meta Source Dir | `MetaSources` | Folder holding `gp_states/` + `trajectories/`. |
| Meta Require Sources | On | Abort at startup when no population model survives validation, instead of silently running plain qLogNEHVI — which would turn a MetaTAF condition into a no-transfer control. Disable only if a source-less run is genuinely intended. |
| Meta Weight Mode | `TafR` | How population models are weighted. `TafR`: by objective-wise pairwise ranking agreement with the current user's observations — every pair of observations is scored once per objective (higher / lower / tied), and a source's weight shrinks with its share of mismatched rankings (recommended — this is the negative-transfer guard). `TafM`: by meta-feature similarity. `TafRPareto`: the former Pareto-dominance variant of `TafR`, kept **only as an ablation** — pairs near the Pareto front are typically mutually non-dominated, which dominance must discard as "incomparable", starving the similarity estimate exactly where a tuned study spends its iterations. |
| Meta Rho | `1.0` | Bandwidth of the weighting kernel. Smaller = stricter (disagreeing sources are dropped sooner). |
| Meta Target Weight | `1.0` | Weight of the current user's own model in the blend. |
| Meta Warmup Iters | `1` | The first *k* optimization suggestions follow the population models alone (the user's own model has too little data to say anything yet). Keep small — on a 10-iteration budget, 1 is a good default. |
| Meta Decay Start Iter | `2` | Iteration after which population influence starts to fade (d1 in Liao et al.'s decay). |
| Meta Decay Rate | `0.3` | How fast it fades per iteration (d2). With the defaults (2, 0.3), population influence is gone after iteration 5 and the run finishes fully personalized. `0` = never fade — an ablation setting, not a study setting (see section 6). |

A note on **ties** under `TafR`: a tie in the participant's rated values is a ranking
claim of its own, so a source that asserts a strict order there scores a mismatch (over
the fixed denominator of all pairs × objectives). With quantized ratings (e.g. Likert
scales) target ties are common while GP posterior means essentially never tie — so every
source's absolute weight shrinks by roughly the tie fraction, uniformly. The *ordering*
between sources is unaffected, but if you tighten `Meta Rho` below its default, remember
this inflation: sources near the cutoff get dropped sooner on tie-heavy data.

The remaining hyperparameters (restarts, raw samples, MC samples, seed, iteration counts)
are the shared ones described in README 8.10/8.11 and apply unchanged.

## 5. What gets logged

Everything a normal multi-objective run logs (`ObservationsPerEvaluation.csv` with
`IsPareto`, `HypervolumePerEvaluation.csv`, `ExecutionTimes.csv`), plus:

* **`MetaWeightsPerEvaluation.csv`** — one row per optimization iteration:
  `Iteration; OptimizationStep; TargetWeight; DecayFactor; <one column per population model>`.
  `Iteration` is the global evaluation index — the same as the design's row in
  `ObservationsPerEvaluation.csv` and `HypervolumePerEvaluation.csv`, so the files join on
  it (with 5 sampling iterations, the first weights row is `Iteration` 6);
  `OptimizationStep` counts the optimization iterations only (1, 2, …). Files written
  before this change logged the optimization step as `Iteration`: add the number of
  sampling iterations before joining them.
  Read it as "who was steering": `TargetWeight = 0.0` marks warmup iterations driven by
  the population alone; the per-source columns show TAF weights after decay. If one source
  dominates every participant, your population may be too homogeneous — if weights differ
  a lot between participants, TAF is doing its job selecting matching predecessors.
* **`HypervolumePerEvaluation.csv`** — the hypervolume openbo computes after every
  evaluation, against the reference point −1.1 in every objective of the normalized
  [−1, 1] space (the `ReferencePoint` column), the same as the BoTorch backend. Values are
  not comparable with logs written with the former reference point −1.
* **`MetaSourcesUsed/`** — an exact copy of the population models this run actually
  loaded. Candidates that matched the study frame but could not be used by the optimizer
  (unreadable trajectory, malformed hyperparameters, a Pareto front that never beats the
  reference point) are removed from this folder and listed with the reason in
  `MetaRunState.json`. Even if the shared `MetaSources` folder changes later, every run
  archives what it actually used.
* **`MetaRunState.json`** — the run's provenance, written before the first trial: library
  versions (torch, botorch, gpytorch, open-bo, numpy, ...), how openbo is installed (pip's
  `direct_url.json`: git commit or editable path) plus SHA-256 hashes of the openbo modules
  in use, the exact optimizer configuration (`MOTAFConfig`, including the settings the
  backend pins instead of relying on openbo defaults, and the hypervolume reference
  point), the study frame and its digest, the seed and the Unity init configuration, the
  loaded / dropped / frame-rejected sources with their provenance stamps, sources with
  identical trajectories, and the population manifest it was checked against. It is
  rewritten after every evaluation: `progress` holds the evaluations completed, the last
  `Iteration`, the latest hypervolume and source weights; at the end `finished` is `true`
  and `finish_reason` says why — `completed`, `stop_requested` (Unity ended the study,
  e.g. perfect ratings; the reason is in `finish_detail`), `error`, or `startup_abort`
  (then `abort_reason` says why). A file that still says `"finished": false` belongs to a
  backend that was killed mid-run.

## 6. Study-design guidance (read before running a real experiment)

* **Freeze the population.** If Meta-BO is a condition in your experiment, build the
  population models from a *pilot* cohort, freeze the folder, and give every analyzed
  participant the identical set. If you instead keep adding each finished participant as
  a new source, participant N's treatment depends on participants 1..N-1 — an ordering
  confound that breaks independence assumptions in your analysis. The `population.json`
  that `meta_train.py` writes enforces this: once it is in the folder, a run whose loaded
  population differs from it does not start. Each run's `MetaRunState.json` records the
  manifest's hash, so you can show that every participant got the same population.
* **Compare against a no-transfer control.** The honest baseline for "Meta-BO helped" is
  the same study with the BoTorch backend. For the same Seed both backends draw the
  identical initial Sobol design and use the same target model and acquisition
  (SingleTaskGP, qLogNEHVI with reference point −1.1, same MC-sampler seed), so they differ
  in the transfer terms plus one implementation detail: the acquisition optimizer. MetaTAF
  (openbo) uses BoTorch's `optimize_acqf` defaults (all restarts in one L-BFGS-B batch, up
  to 2000 iterations, a per-iteration RNG seed); `mobo.py` optimizes the restarts in
  batches of 5 with at most 200 iterations. Suggestions after the sampling phase are
  therefore methodologically equivalent, not bit-identical. (A
  source-less MetaTAF run needs **Meta Require Sources** switched off.) Liao et al. (CHI
  2024) is the template for this comparison.
* **Leave the decay on.** `Meta Decay Rate = 0` ("never fade") is an ablation setting,
  not a study setting: without decay the population keeps a constant-size say in the
  acquisition while the participant's own improvement signal shrinks as their model
  converges, so late iterations stay population-driven and the run never finishes
  personalizing. In simulation benchmarks, no-decay TAF finished **below plain
  multi-objective BO** at the final checkpoint — the decay γ(t) is exactly what Liao et
  al. added to hand control back. This applies to every weight mode (TafR, TafM,
  TafRPareto) alike; keep the (d1, d2) defaults unless you are explicitly ablating the
  decay mechanism itself.
* **Population size:** their study used 14 population models; simulations showed benefits
  from as few as a handful, with earlier convergence as the population grows.
* **Same frame, always.** Sources are only accepted when parameter/objective names,
  bounds, and minimize flags match the live study exactly. Changing an objective's range
  or direction mid-study invalidates your population models — regenerate them.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `The Meta-TAF backend needs the 'openbo' package` | Install openbo **for the Python BOforUnity uses** (README 8.5 shows which one that is): `python -m pip install "open-bo @ git+https://github.com/M-Colley/openbo@main"` |
| `The installed 'openbo' predates the TAF-R rework` | Your openbo is from before 2026-08, when the `taf_r` mode changed from Pareto-dominance to objective-wise ranking agreement; running it would silently compute a different similarity than configured. Run the printed command: `python -m pip install --force-reinstall --no-deps "open-bo @ git+https://github.com/M-Colley/openbo@main"` (plain `--upgrade` does nothing here — openbo's version number did not change). |
| `source '<name>' skipped: built for a different study frame` | The artifact was generated for different names/bounds/minimize flags. The lines below it list the exact field. Regenerate with a matching `frame.json`. |
| `source '<name>' skipped: no frame block` | The artifact predates frame stamping or was hand-built. Regenerate with `meta_train.py` (or set the env var `BO_META_ALLOW_UNFRAMED=1` if you are absolutely sure). |
| `source '._<name>' skipped: macOS AppleDouble metadata file` | Harmless: metadata files that macOS adds when a folder is copied via an exFAT/FAT32 drive or a network share. Delete them (`dot_clean` on macOS) to silence the message. |
| `source '<name>' skipped: could not be copied` / `gp_states unreadable` / `is not a JSON object` | The file is locked by another program, is a cloud-storage placeholder that is not downloaded, or is not a valid artifact. Make the folder available offline / close the program, or regenerate the source. |
| `source '<name>' skipped: its trajectory is not the one its gp_state was built with` | The pair was only half replaced (a `meta_train.py` rebuild that could not replace the gp_state, or an interrupted sync). Rebuild that run with `meta_train.py --force`. |
| `sources [...] have identical trajectories` | The same run is in the folder under two names (e.g. `00_p01_main_run` and `p01_main_run`); it counts twice. Delete all but one and rewrite `population.json` (`--manifest-only`). |
| `the population loaded from '...' differs from its frozen population manifest` | The source folder changed since `population.json` was written (a source added, removed, replaced, or no longer loadable — each difference is listed). During a study: restore the frozen folder. Before the study: rewrite the manifest with `meta_train.py --frame frame.json --out <folder> --manifest-only`. |
| `Seed must be >= 0 for the Meta-TAF backend` | Set a non-negative **Random Seed** in the inspector — the same in every condition, so MetaTAF and the BoTorch control start from the same initial design. |
| `meta_train.py` prints `[KEEP] <name>: already in --out` | That run is already a source in the folder; it is not rebuilt (that would change the population). Add `--force` to rebuild it on purpose. |
| `meta_train.py`: `'--names' placed before the run paths takes them as names` | Put `--names a b` after the run paths, or use `--names a,b` / `--name a --name b`. |
| `source '<name>' passed frame validation but openbo did not load it` | The artifact matches the study frame but the optimizer cannot use it: an unreadable (e.g. half-synced) trajectory file, malformed hyperparameters, or a Pareto front that never beats the reference point. The message carries openbo's reason; regenerate the source with `meta_train.py`, which refuses such artifacts up front. The run continues with the remaining sources (or aborts if none is left). |
| `Meta-TAF: no valid population model found ... requires sources (metaRequireSources)` | The run aborts on purpose, before the first trial (`MetaRunState.json` in the run folder records why). Check the Meta Source Dir path, that `gp_states/` + `trajectories/` contain paired `.json` files, and the listed per-source rejection reasons (frame mismatches as well as sources openbo could not load); regenerate sources against the current frame. Only if a source-less (plain MOBO) run is genuinely intended, switch off **Meta Require Sources** in the inspector. |
| `Parameter/objective key(s) [...] collide with the fixed columns of ObservationsPerEvaluation.csv` | Rename that parameter/objective key: `UserID`, `ConditionID`, `GroupID`, `Timestamp`, `Iteration`, `Phase` and `IsPareto` are columns of the observation log. |
| `meta_train.py` prints `Parameter column '<name>': ... outside the frame bounds` | That run was recorded with different parameter bounds than `frame.json` describes. Use the frame it was recorded with, or leave the run out — rescaling it would silently distort the population model. |
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
weighted average; source weights are multiplied by the Liao-et-al. decay γ(t) with
hyperparameters (d1, d2) = (Meta Decay Start Iter, Meta Decay Rate). Under TAF-R each
pair of observations (i, j) is labeled r_ijm ∈ {+1, −1, 0} per objective m — once on the
participant's observed values, once on the source's posterior mean at the same designs
(ties detected with a 10⁻¹² tolerance) — and the source's distance is its label mismatch
count over the fixed denominator M·C(n, 2), mapped to a weight by an Epanechnikov kernel
with bandwidth ρ (Meta Rho); a source asserting no strict order anywhere is zero-weighted.
TAF-M instead measures meta-feature similarity (dimension + per-objective moments).
`TafRPareto` is the pre-2026-08 dominance-agreement variant of TAF-R, retained as an
ablation.
