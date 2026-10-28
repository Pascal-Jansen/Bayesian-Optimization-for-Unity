# Bayesian Optimization for Unreal Engine

This distribution provides a native Unreal Engine 5 runtime plugin and a separate Python optimization backend. The component exposes study settings and events to Blueprint and C++. The backend implements single objective BO, multi objective BO, CABOP, DynamicBO, and MetaTAF. Parameters and objectives retain their original units across the connection. The Python backend performs normalization and objective direction changes.

## Installation

Use Unreal Engine 5.8 with its supported C++ compiler. Copy `Plugins/BayesianOptimization` into the `Plugins` folder of a C++ Unreal project. Regenerate project files and build the project. Enable Bayesian Optimization in the Plugins window. The plugin uses Unreal's Core, CoreUObject, Engine, Sockets, and Json modules. It requires no additional Unreal plugin.

Keep the remaining distribution in a separate writable directory. Install Python 3.13 and create a virtual environment. Run these commands from this directory with that interpreter:

```text
python -m venv .venv
```

On Windows, install dependencies with `.venv\Scripts\python.exe -m pip install -r requirements.txt`. On macOS and Linux, use `.venv/bin/python -m pip install -r requirements.txt`. Dependency installation can require several gigabytes. The pinned requirements apply to the copied Python backend. Optional MetaTAF and image embedding dependencies are described in `requirements.txt`.

Start the backend before starting the Unreal study. Use the virtual environment's interpreter in place of `python`:

```text
python launch_backend.py --backend botorch --objectives 1 --log-root runs
```

Wait for `Server starts, waiting for connection...`. The component connects to loopback with default port `56001`. Each backend process accepts one study. Start a new process for each study. For concurrent studies, pass a distinct `--port` to each launcher and set the matching Server Port on each component. The launcher verifies the backend and objective count before running an experiment.

## Analytic demonstration

Add a `BoAnalyticDemoActor` to any level. Keep its default settings. Start the single objective backend with the command above and enter Play mode. The actor evaluates `cost = (x - 0.3)^2` for each suggested design. It performs three sampling trials and five optimization trials. The theoretical minimum is zero at `x = 0.3`. Inspect the Optimizer component's state and generated CSV files to check the run.

The actor's `Start On Begin Play` setting controls automatic startup. Its Optimizer component exposes the same study settings as other actors. The analytic objective provides a deterministic integration example. Human studies supply measurements through the Blueprint workflow below.

## Blueprint study workflow

1. Add a Bayesian Optimizer component to an actor that remains alive throughout the study.
2. Add parameter entries to Settings. Set unique keys and finite bounds. Each parameter requires `Low < High`.
3. Add objective entries. Set their measurement bounds and minimization flags. Each objective requires `Low < High`.
4. Bind On Design Suggested, On Error, and On Completed before calling Start Study.
5. In On Design Suggested, read the Values map and apply every design parameter. Store Iteration alongside participant telemetry. Begin the participant's trial after applying the design.
6. Complete the trial and calculate its objective values. Create a String to Double map containing every configured objective key. Call Submit Objectives once. The component waits for the next design.
7. Present completion feedback when On Completed fires. Present Last Error when On Error fires. Stop Study ends a session intentionally. Disconnect releases local resources immediately.

Keys are case sensitive on the Python side. The component rejects duplicate keys, case variants, overlapping parameter and objective keys, and reserved CSV column names. Objective values outside their bounds are refused. The current trial remains active so instrumentation can correct the measurement. The component snapshots settings at startup. Changes to Settings during a run apply to the next run.

The socket worker handles connecting, partial writes, and newline framing outside the game thread. Component events run on the game thread during ticks. Keep the component enabled and ticking while a study is active. Idle Timeout Seconds includes participant response time and model computation. Match the Python `BO_SOCKET_TIMEOUT_SEC` environment variable to the longest permitted trial. Stop Study sends an ordered stop message and closes the socket. EndPlay and component destruction stop and join the worker.

## Backend selection

| Component Backend | Launcher backend | Objective count | Method |
| --- | --- | --- | --- |
| BoTorch | `botorch` | 1 | Single objective acquisition |
| BoTorch | `botorch` | 2 or more | Multi objective acquisition |
| CABOP | `cabop` | 1 | Cost aware acquisition |
| CABOP | `cabop` | 2 or more | Scalarized multi objective CABOP |
| DynamicBO | `dbo` | 1 | Temporal Gaussian process |
| MetaTAF | `meta-taf` | 2 or more | Transfer from source tasks |

For example, two objectives require `python launch_backend.py --backend botorch --objectives 2`. The component supports one design per trial. CABOP's multi objective mode uses scalarization. The BoTorch multi objective mode uses its existing Pareto based backend. On Metric reports `tempCoverage` for sampling progress. The meaning of `coverage` depends on the backend, including best objective or hypervolume. Last Suggestion contains the full message JSON for additional metadata.

## Advanced configuration

Advanced Init Json accepts three optional root fields: `config`, `context`, and `cabopGroupCosts`. Typed component settings own the core configuration, including dimensions, budgets, backend, seed, and warm start. Attempts to override these fields through Advanced Init Json fail validation. Backend specific fields within `config` are passed to the existing Python runtime. Keep their types and names consistent with that runtime.

For a stationary DynamicBO comparison, use:

```json
{"config":{"dboStationaryBaseline":true,"dboValidationEvery":5,"explorationRatio":0.1}}
```

For CABOP, set Group, Tolerance, and Prefabricated Values in each parameter entry. Tolerance is a fraction of the parameter range between zero and one. Objective Weight controls scalarization. Group costs can be supplied explicitly:

```json
{"config":{"cabopUseCostAwareAcquisition":true,"cabopUpdateRule":"actual"},"cabopGroupCosts":[{"group":"default","cost":{"unchanged":1,"swapped":10,"acquired":100},"actualCost":{"unchanged":1,"swapped":10,"acquired":100}}]}
```

For contextual BoTorch studies, supply the same context object accepted by `Backend/context_support.py`. An enabled context is refused for CABOP, DynamicBO, and MetaTAF. For MetaTAF, configure `metaSourceDir` in `config` and install the optional OpenBO dependency before launch. `MetaSources/README.md` describes the source data format. The launcher sets `BO_INIT_ROOT` to this distribution's InitData directory and `BO_LOG_ROOT` to the requested writable directory. Use absolute paths for external data and source libraries.

Warm start requires parameter and objective CSV paths. Select the objective format explicitly when the source format is known. The available formats are `raw`, `normalized_max`, `normalized_native`, and `auto`. MetaTAF uses its source library and refuses this warm start mode.

## Data and reproducibility

The backend writes semicolon separated CSV files under `<log root>/<user>/<condition>/run`, with numbered folders for subsequent runs. Observation Iteration identifies the exact design delivered to Unreal. Record that value with any additional application telemetry. Preserve study settings, backend selection, dependency versions, and seed with each experiment. Participant identifiers should be pseudonymous. The connection and CSV files remain local.

The backend source and vendored licenses are included in Backend. The root MIT license covers the original project and plugin additions. CABOP retains its MIT notice. The DynamicBO implementation retains its BSD 3 Clause notice and provenance document. Installed Python packages retain their own licenses. Commercial redistribution requires retaining applicable notices.

## Maintenance and validation

The common repository maintains numerical source in `Backend/`. Run `python tools/sync_backends.py` to generate both engine copies. `python Unreal/tools/refresh_backend.py` remains a compatibility command for the Unreal copy. Synchronization includes source, licenses, example data, and requirements. Stale managed files are removed after containment checks. Canonical text hashes keep Windows and Linux manifests consistent.

Check source drift with `python Unreal/tools/refresh_backend.py --check`. Run portable checks with `python -m unittest discover -s Unreal/tests -v`. Set `BO_RUN_INTEGRATION_TESTS=1` to run actual BO and MOBO studies through the launcher. These tests use the fixed optimizer port and require it to be available. Standalone copies use `python -m unittest discover -s tests -v` from their distribution directory.

Build the plugin with Unreal's `RunUAT BuildPlugin` command for the target platform. Run `Automation RunTests BayesianOptimization` in Unreal's automation console to execute native protocol validation. The source compiled with Unreal Engine 5.8 on Windows for Editor Development, Win64 Development, and Win64 Shipping. Native protocol automation passed with zero warnings and zero errors. Seven portable tests passed, including actual BO and MOBO runs with three evaluations each.

Additionally, the native analytic actor completed eight evaluations against the real Python backend in an Unreal 5.8 game session with NullRHI. Its CSV contained all three sampling trials and five optimization trials. The backend exited successfully. The best observation was `x = 0.3048200638`, with `cost = 0.00002323301472`. The actor can also be started through the game console with `Summon /Script/BayesianOptimization.BoAnalyticDemoActor`. Linux and macOS engine builds and packaged application deployment require validation on their respective toolchains.

Unity's questionnaire UI, editor inspectors, prefabs, and demo scenes remain Unity assets. Unreal studies need corresponding UMG interfaces and level content. The native analytic actor and Blueprint component provide the Unreal integration. Packaged deployment requires distributing and starting the external Python environment alongside the application. The plugin source has no embedded Python runtime or automatic dependency installation.

The implementation follows Epic's [plugin structure](https://dev.epicgames.com/documentation/en-us/unreal-engine/plugins-in-unreal-engine), [socket API](https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Sockets/FSocket), and [component metadata](https://dev.epicgames.com/documentation/en-us/unreal-engine/gameplay-classes-in-unreal-engine). The installed Unreal 5.8 source was used to verify nonblocking connection and receive semantics.
