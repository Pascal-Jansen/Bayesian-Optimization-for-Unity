# MetaTAF source models

MetaTAF transfers information from a fixed population of source models. Each population contains GP hyperparameters, normalized observations, a Pareto front, and a population manifest:

```text
study-sources/
  gp_states/<name>.json
  trajectories/<name>.json
  population.json
```

Generate source models from completed runs with the canonical tool in the common repository. Use the Python environment that runs the optimizer. The optional OpenBO dependency must use the audited revision specified in Backend/requirements.txt.

```text
python Backend/meta_train.py --frame frame.json --out study-sources --source-type human --y-calibration measured path/to/completed/run
```

The frame file records every parameter and objective, including bounds and objective direction. Source type and calibration describe how objective values were obtained. Set source type to human, llm-persona, or synthetic. Set calibration to measured or generated. These values are recorded with each source.

Source names derive from their original run paths. Existing sources are retained unless --force is given. Use --dry-run to inspect a planned build. Configure metaSourceDir with the population directory in the engine wrapper. An absolute path can identify an external study directory.

The population manifest records source names, content hashes, and the frame digest. Freeze this directory before collecting study data. The runtime rejects a population whose loaded sources differ from its manifest. Regenerate the manifest with --manifest-only after intentional changes made before the study.

Source frames must match the current study. The runtime reports incompatible or unloadable sources. Each run records its actual source population in MetaSourcesUsed and MetaRunState.json. Preserve these records with the study data and analysis scripts.
