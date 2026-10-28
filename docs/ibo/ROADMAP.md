# Interactive Bayesian Optimization (IBO): roadmap

Bayesian Optimization for Unity becomes **Interactive Bayesian Optimization (IBO)**: one optimizer
backend that any engine or experiment tool can use to run human-in-the-loop studies, with Unity,
Unreal and Python as integrations.

## Decisions

| Topic | Decision |
|---|---|
| Name | Interactive Bayesian Optimization, short **IBO** |
| Repository | This repository, with the layout of Mark Colley's Bayesian-Optimization-for-Game-Engines |
| Package names | `interactive-bo` on PyPI and npm (`ibo` is taken on both); Python import `interactive_bo` |
| Protocol | Version 1 = the existing wire protocol, written down; new features go into version 2 |
| QuestionnaireToolkit | An optional Unity add-on; the Unity core works without it |
| Renamed | Repository, documentation, citation, Unity package folder and menus, Unreal plugin display name, Python package |
| Kept | C# namespaces (`BayesianOptimization`), class names (`BoForUnityManager`), assembly names and the Unreal module, so authored projects need no second migration |
| Existing Unity users | A migration guide (no compatibility layer) |

## Layout

```
Backend/        canonical Python optimizer and IBO server (ibo_server.py); copied into the engines
Unity/          Unity project: the IBO package, samples, tests, and the QuestionnaireToolkit add-on
Unreal/         Unreal plugin (C++, Blueprint component), launcher and backend copy
Python/         Python client and server launcher (pip install interactive-bo)
protocol/       IBO protocol specification and JSON schema
tools/          backend synchronization (sync_backends.py) and release packaging
tests/          Python tests of the backend
docs/           architecture, validation, guides
```

## Phases

1. **Core and protocol: done.** `ibo_server.py` picks the backend from the init message; protocol v1
   specification and schema; Python client and launcher; end-to-end studies in CI.
2. **Merge and rename: done in the repository.**
   - Mark's multi-engine layout (shared `Backend/` with synchronized engine copies, Unity core without
     QuestionnaireToolkit, URP/HDRP support, Unreal plugin, release packaging) with Phase 1 on top
     (`Backend/ibo_server.py`, `Python/`); QuestionnaireToolkit add-on with the questionnaire examples;
     IBO naming (README, citation, `Assets/IBO`, `Tools > IBO` menus, Unreal plugin name, artifact names);
     [migration guide](../migration-to-ibo.md).
   - Open: GitHub repository rename and Zenodo record title; a Play-mode run of the questionnaire examples
     with Python on Windows and macOS; the three open license items of the add-on
     (`Unity/Assets/IBO-QuestionnaireToolkit/Third-Party Notices.txt`).
3. **Unreal: largely done** by Mark's plugin (built and tested with Unreal Engine 5.8 on Windows).
   Open: macOS and Linux validation; optionally start `ibo_server.py` from the launcher.
4. **Web client.** Browsers cannot open TCP sockets: a WebSocket transport (protocol v2) or a small
   bridge, a TypeScript client, and a jsPsych plugin for online studies.
5. **Human-in-the-loop methods.**

| Feedback | Optimizer |
|---|---|
| Ratings and questionnaires (explicit) | Single- and multi-objective BO, cost-aware (CABOP), drifting objectives (DBO), contextual (LCE-M), transfer (MetaTAF): available |
| Behavior, telemetry, physiology (implicit) | Same as above: available |
| Pairwise choice ("A or B?") | Preferential BO (BoTorch `PairwiseGP`): new |
| Direct manipulation (slider along a proposed line) | Sequential line search (Koyama et al., SIGGRAPH 2017): new |
| Designer steering (lock, propose, veto) | User-steered BO: new |
| Comfort and safety limits | Constrained / safe BO: new |

The new methods need client-driven commands (`suggest`, `observe`) from protocol v2. Also open for v2:
clients ignoring unknown server messages (the Unity integration still rejects them), and an `error` message.
