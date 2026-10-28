# iBO Protocol, version 1

How an integration (Unity, Unreal, web, Python, ...) runs a human-in-the-loop study with the iBO
server. Version 1 is the protocol that Bayesian Optimization for Unity has used up to 1.9, written
down; the Unity integration already speaks it. The machine-readable form is
[`ibo-protocol-v1.schema.json`](ibo-protocol-v1.schema.json).

## Roles and session

- The **server** is `Backend/ibo_server.py` (it picks the optimizer backend from the init message) or one
  backend script (`bo.py`, `mobo.py`, `cabop_bo.py`, `cabop_mobo.py`, `meta_mobo.py`, `dbo.py`).
  It listens on **127.0.0.1**, port **56001** or `BO_PORT`, accepts **one** client and exits when the study ends.
- The **client** is the integration. It describes the study once, then shows each proposed design to a
  person and reports the measured objective values.
- The server prints `Server starts, waiting for connection...` on stdout right before it accepts.
  Launchers wait for that line; clients may also retry the connection for a few seconds.
- The server binds to loopback and has no authentication. Never expose it to a network.

## Framing

- TCP, UTF-8, one JSON object per line (NDJSON) ending in `\n`. `\r\n` and blank lines are tolerated.
- Every message is an object with a string `type`. Receivers **ignore unknown types**, so later
  versions can add messages. (The Unity and Unreal integrations still reject unknown server messages: the study stops.)
- Numbers are finite JSON numbers; `NaN` and `Infinity` are not allowed.
- A line may hold up to 64 MiB (`BO_MAX_RECV_BUF_BYTES`), enough for large context embeddings.

## Sequence

```
client                                         server
  | --- init -------------------------------->  |
  | <------------------------- tempCoverage ---  |  sampling phase, 0..1 (optional)
  | <--------------------------- parameters ---  |  design to show
  |       the person experiences the design      |
  | --- objectives -------------------------->  |
  | <----------------------------- coverage ---  |  optimization progress (optional)
  |              ... parameters / objectives ... |
  | <---------------- optimization_finished ---  |
  |                                  connection closed by the server
```

The client may send `stop` at any time instead of the next message: the server logs the reason and
ends; every completed evaluation stays logged. A `stop` in place of the init refuses the study.

## Client to server

### `init`

| Field | Type | Meaning |
|---|---|---|
| `type` | `"init"` | |
| `protocolVersion` | integer, optional | `1`. Absent means 1. |
| `config` | object | Study and optimizer settings (below) |
| `parameters` | array | Design parameters, in the order they are logged |
| `objectives` | array | Objectives, in the order they are logged |
| `user` | object | `userId`, `conditionId`, `groupId` (strings, default `"-1"`). Each study is logged in `<userId>/<conditionId>/run/` (`run_1`, `run_2`, ... for later runs). |
| `context` | object or null, optional | Contextual optimization (LCE-M GP), manual section 8.13 |
| `cabopGroupCosts` | array, optional | CABOP cost model per parameter group, manual section 8.7 |

**Parameter**: `key` (unique string), `init: {"low", "high"}` with `low < high`, in the parameter's own
units. CABOP also reads `group` (string), `tolerance` (fraction of the range, 0..1) and
`prefabValues` (numbers); `optSeqOrder` (integer) is informational.

**Objective**: `key` (unique, distinct from parameter keys), `init: {"low", "high", "minimize"}` with
`low <= high` and `minimize` 0 (larger is better) or 1 (smaller is better). CABOP's scalarized
multi-objective mode also reads `weight`.

Older clients may send `init` as a string, `"low,high"` or `"low,high,minimize"`; servers accept both.

**`config`** (unknown keys are ignored; the defaults are the Unity integration's):

| Key | Default | Applies to | Meaning |
|---|---|---|---|
| `optimizerBackend` | `"botorch"` | all | `botorch`, `cabop`, `meta-taf` or `dbo` |
| `nParameters`, `nObjectives` | required | all | Must equal the array lengths |
| `numSamplingIterations` | 2·(nParameters+1) | all | Initial space-filling designs |
| `numOptimizationIterations` | 10 | all | Model-based designs after sampling |
| `seed` | 3 | all | Reproducibility |
| `batchSize` | 1 | botorch | Only 1 is supported in the human-in-the-loop loop |
| `numRestarts`, `rawSamples`, `mcSamples` | 10, 1024, 128 | botorch, meta-taf | Acquisition optimization |
| `warmStart`, `initialParametersDataPath`, `initialObjectivesDataPath`, `warmStartObjectiveFormat` | `false` | botorch, cabop, dbo | Start from earlier data (manual section 8.10) |
| `cabopObjectiveMode`, `cabopUseCostAwareAcquisition`, `cabopUpdateRule`, `cabopEnableCostBudget`, `cabopMaxCumulativeCost` | | cabop | Manual section 8.7 |
| `metaSourceDir`, `metaRequireSources`, `metaWeightMode`, `metaRho`, `metaTargetWeight`, `metaWarmupIters`, `metaDecayStartIter`, `metaDecayRate` | | meta-taf | Manual section 8.14 |
| `dboSpatialKernel`, `dboAlphaParameterization`, `dboInitialAlpha`, `dboAcquisitionTimeOffset`, `dboValidationEvery`, `dboValidationConfidence`, `dboValidationVisitedOnly`, `dboStationaryBaseline`, `explorationRatio` | | dbo | Manual section 8.15 |

`ibo_server.py` runs `bo.py` for `botorch` with one objective and `mobo.py` with more, `cabop_runtime`
for `cabop` (mode from `cabopObjectiveMode`), `meta_mobo_runtime` for `meta-taf` and `dbo_runtime` for
`dbo`. Each backend validates the rest and refuses an inconsistent study before the first design.

### `objectives`

`{"type": "objectives", "values": {"<objective key>": number, ...}}`, exactly once after each
`parameters` message, with one value per objective in the objective's own units (within its bounds).
Several measurements of one objective (questionnaire items) are averaged by the client first.

### `stop`

`{"type": "stop", "reason": "<text, optional>"}`

## Server to client

| Message | Fields | Meaning |
|---|---|---|
| `parameters` | `values: {"<parameter key>": number}`, `iteration` (integer, optional) | The next design, in the parameters' own units within their bounds. `iteration` is the row's `Iteration` in `ObservationsPerEvaluation.csv`. |
| `tempCoverage` | `value` (0..1) | Progress of the sampling phase |
| `coverage` | `value` (number) | Hypervolume of the observed designs (several objectives) or best objective value (one) |
| `optimization_finished` | | No further designs; the server closes the connection |

## Errors and timeouts

Version 1 has no error message. A server that cannot run the study prints the reason to its console
and closes the connection; clients should show the server's console output (the launchers keep it).
The server waits up to `BO_ACCEPT_TIMEOUT_SEC` (300 s) for the connection and up to
`BO_SOCKET_TIMEOUT_SEC` (3600 s) for each message.

## Server environment

| Variable | Default | Meaning |
|---|---|---|
| `BO_PORT` | 56001 | Listening port (the Unity integration always uses 56001) |
| `BO_LOG_ROOT` | `./LogData` | Root of the run folders `<userId>/<conditionId>/run/` (manual section 8.12) |
| `BO_INIT_ROOT` | `./InitData` | Warm-start files and context data |
| `BO_ACCEPT_TIMEOUT_SEC`, `BO_SOCKET_TIMEOUT_SEC`, `BO_MAX_RECV_BUF_BYTES` | 300, 3600, 64 MiB | Limits above |
| `BO_MODEL_LIBRARY` | `botorch` | `botorch`, `botorch_rs` (botorch-rust) or `auto`; a study factor, never changed within a study (`Backend/bo_library.py`) |
| `BO_TORCH_THREADS` | | Torch thread count |

## Implementations

| Side | Where |
|---|---|
| Server | `Backend/ibo_server.py` and the backends next to it; `tools/sync_backends.py` copies them into the engines |
| Unity client | `Unity/Assets/iBO/Scripts/SocketNetwork.cs`, started by `PythonStarter.cs` |
| Unreal client | `Unreal/Plugins/BayesianOptimization/Source/BayesianOptimization/Private/BoProtocol.cpp` and `BoTransport.cpp`, started by `Unreal/launch_backend.py` |
| Python client | `Python/interactive_bo/client.py`, launcher `Python/interactive_bo/server.py` |

## Planned for version 2

Not part of version 1; listed so integrations can prepare:

- an `error` message, so clients can show why a study was refused;
- a `hello` handshake announcing the server's version and capabilities;
- client-driven commands (`suggest`, `observe`) for pairwise preferences and designer steering;
- a WebSocket transport for browsers.
