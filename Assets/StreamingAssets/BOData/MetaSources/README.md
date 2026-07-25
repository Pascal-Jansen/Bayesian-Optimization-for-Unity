# MetaSources — population models for the Meta-TAF backend

This folder holds the "population models" (source artifacts) that the **MetaTAF**
optimizer backend transfers from. Layout:

```
MetaSources/
  gp_states/      one <name>.json per source (GP hyperparameters + study frame)
  trajectories/   one <name>.json per source (normalized observations + Pareto front)
```

You do not write these files by hand. Generate them from completed runs with:

```
python Assets/StreamingAssets/BOData/BayesianOptimization/meta_train.py ^
    --frame frame.json --out Assets/StreamingAssets/BOData/MetaSources ^
    path/to/LogData/<user>/<condition>/run ...
```

Sources whose study frame (parameter/objective names, bounds, minimize flags) does not
match the live study are skipped at runtime with a field-by-field explanation — that is
intentional, not a bug. See `docs/meta-taf-student-guide.md` for the full workflow.
