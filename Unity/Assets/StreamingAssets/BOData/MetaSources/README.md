# MetaSources — population models for the Meta-TAF backend

This folder holds the "population models" (source artifacts) that the **MetaTAF**
optimizer backend transfers from. Layout:

```
MetaSources/
  gp_states/       one <name>.json per source (GP hyperparameters + study frame)
  trajectories/    one <name>.json per source (normalized observations + Pareto front)
  population.json  population manifest: names, SHA-256 hashes, frame digest
```

You do not write these files by hand. Generate them from completed runs with:

```
python Assets/StreamingAssets/BOData/BayesianOptimization/meta_train.py ^
    --frame frame.json --out Assets/StreamingAssets/BOData/MetaSources ^
    --source-type human --y-calibration measured ^
    path/to/LogData/<user>/<condition>/run ...
```

`--source-type` (`human`, `llm-persona`, `synthetic`) and `--y-calibration` (`measured`,
`generated`) are required provenance stamps written into every artifact — label them
honestly. `meta_train.py` needs the full stack plus openbo: on the Unity machine, run it with
the interpreter the Unity Console logs as `Optimizer Python: …` instead of `python`.

Each source is named after its run's path (`<participant>_<condition>_<run>`, e.g.
`p01_main_run`), so rebuilding the folder with more runs adds only the new ones; runs that
are already here are kept, under the name they have here (also an index-prefixed
`00_..._run` name from an older `meta_train.py`), unless you pass `--force`. `--dry-run`
shows what would be written without writing anything.

`population.json` freezes the population: it is rewritten by every `meta_train.py` build,
and while it is present the backend refuses to start when the sources it loads differ from
it (added, removed, replaced or unloadable sources; converted line endings, e.g. from a git
checkout, do not count). After editing this folder by hand
before a study, rewrite it with `meta_train.py --frame frame.json --out <this folder>
--manifest-only`; do not change the folder during a study.

Sources whose study frame (parameter/objective names, bounds, minimize flags) does not
match the live study are skipped at runtime with a field-by-field explanation — that is
intentional, not a bug. Sources that match the frame but cannot be loaded by the optimizer
are dropped with the reason as well (so are macOS `._*.json` metadata files from copies via
exFAT drives or network shares); each run records what it actually used in its
`MetaSourcesUsed/` folder and `MetaRunState.json`. See `docs/meta-taf-student-guide.md`
for the full workflow.
