# Interactive Bayesian Optimization (IBO): roadmap

Bayesian Optimization for Unity becomes **Interactive Bayesian Optimization (IBO)**: one optimizer
core that any engine or experiment tool can use to run human-in-the-loop studies, with Unity as one
integration among several.

## Decisions taken (change them here if needed)

| Topic | Decision |
|---|---|
| Name | Interactive Bayesian Optimization, short **IBO** |
| Package names | `interactive-bo` on PyPI and npm (`ibo` is taken on both); Python import `interactive_bo` |
| Repository | One repository for core, protocol and all integrations |
| Protocol | Version 1 = today's wire protocol, written down; new features go into version 2 |
| Existing Unity users | A migration guide (no compatibility layer) |
| Order | Core and protocol, then Unity on the core with the rename, then Unreal, then web, then new HITL methods |

## Target layout

```
core/                 Python: optimizer backends, IBO server, Python client   (pip install interactive-bo)
protocol/             IBO protocol specification and JSON schema
integrations/unity/   Unity package (today's Assets/BOforUnity + QuestionnaireToolkit) and its host project
integrations/unreal/  Unreal Engine plugin (C++, Blueprint nodes)
integrations/web/     JavaScript/TypeScript client (online studies, jsPsych)
docs/                 Manual, guides
```

## Phases

### 1. Core and protocol: done

- `ibo_server.py`: one server entry point that picks the backend from the init message; the backends
  run unchanged on the connection it hands over (`bo_protocol.adopt_connection`). `BO_PORT` selects the port.
- [`protocol/`](../../protocol/README.md): protocol v1 specification and JSON schema, validated against the
  messages of Unity, the backends and the Python client.
- [`core/`](../../core/README.md): `interactive_bo` with the Python client (standard library only), the
  server launcher and `python -m interactive_bo serve|demo`.
- Tests: stub tests for the server and hand-over, client and launcher tests, end-to-end studies with the
  real backends (multi- and single-objective BoTorch, DBO), all in CI.

The backends still live in `Assets/StreamingAssets/BOData/BayesianOptimization`, so Unity keeps working unchanged.

### 2. Unity on the core, and the rename

- Move the backends into `core/interactive_bo/backends/`; ship them to Unity as a build step instead of keeping
  the only copy in StreamingAssets.
- PythonStarter starts `ibo_server.py` instead of choosing a script.
- Move the Unity project to `integrations/unity/`; rename namespaces and assemblies `BOforUnity` → `IBO.Unity`
  (scenes and prefabs keep working: Unity links scripts by GUID).
- Let SocketNetwork ignore unknown server messages (protocol v1 rule).
- Rename the repository (GitHub redirects old links), the Zenodo record title ("formerly Bayesian Optimization
  for Unity"), README, citation and the Asset Store submission kit in `AssetStore/`.
- Migration guide for existing projects.

### 3. Unreal Engine plugin

C++ client of protocol v1 (TCP, NDJSON), server launcher, Blueprint nodes (start study, next design, report),
an example map. Needs an Unreal Engine installation to build and test.

### 4. Web client

Browsers cannot open TCP sockets: a WebSocket transport (protocol v2) or a small bridge, a TypeScript client,
and a jsPsych plugin for online studies.

### 5. Human-in-the-loop methods

| Feedback | Optimizer |
|---|---|
| Ratings and questionnaires (explicit) | Single- and multi-objective BO, cost-aware (CABOP), drifting objectives (DBO), contextual (LCE-M), transfer (MetaTAF): available |
| Behavior, telemetry, physiology (implicit) | Same as above: available |
| Pairwise choice ("A or B?") | Preferential BO (BoTorch `PairwiseGP`): new |
| Direct manipulation (slider along a proposed line) | Sequential line search (Koyama et al., SIGGRAPH 2017): new |
| Designer steering (lock, propose, veto) | User-steered BO: new |
| Comfort and safety limits | Constrained / safe BO: new |

The new methods need client-driven commands (`suggest`, `observe`) from protocol v2.
