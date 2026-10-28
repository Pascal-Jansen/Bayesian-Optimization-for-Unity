<h1 align="center">iBO · Interactive Bayesian Optimization</h1>

<p align="center">
  <b>Human-in-the-loop Bayesian optimization for Unity, Unreal Engine and Python.</b><br>
  People's ratings and behavior steer the design of interactive systems, one evaluation at a time.
</p>

<p align="center">
  <a href="Unity/README.md"><img alt="Unity 6000.6" src="https://img.shields.io/badge/Unity-6000.6-000000?logo=unity&amp;logoColor=white"></a>
  <a href="Unreal/README.md"><img alt="Unreal Engine 5.8" src="https://img.shields.io/badge/Unreal%20Engine-5.8-0E1128?logo=unrealengine&amp;logoColor=white"></a>
  <a href="Python/README.md"><img alt="Python 3.13" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://botorch.org"><img alt="BoTorch 0.18" src="https://img.shields.io/badge/BoTorch-0.18-EE4C2C?logo=pytorch&amp;logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue"></a>
  <a href="https://doi.org/10.5281/zenodo.19786494"><img alt="DOI 10.5281/zenodo.19786494" src="https://zenodo.org/badge/DOI/10.5281/zenodo.19786494.svg"></a>
  <a href="https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/actions/workflows/python-checks.yml"><img alt="Python checks" src="https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/actions/workflows/python-checks.yml/badge.svg"></a>
</p>

<p align="center">
  <b><a href="https://pascal-jansen.github.io">Pascal Jansen</a></b>, Ulm University ·
  <b><a href="https://m-colley.github.io">Mark Colley</a></b>, University College London
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/ibo-overview-dark.svg">
  <img alt="iBO overview: Unity, Unreal Engine, Python and other clients exchange designs and objectives with the iBO backend over the iBO protocol; the backend runs BoTorch or botorch-rust with BO, MOBO, CABOP, DBO, contextual LCE-M and MetaTAF" src="images/ibo-overview-light.svg">
</picture>

## What iBO does

You declare **design parameters** (what the system may change, such as sizes, colors or timings) and **objectives**
(what a person's ratings or behavior score, such as usability or task time). iBO's Bayesian optimizer then proposes the
next design to show after every evaluation. It finds good designs, or the **Pareto front** of best trade-offs between
several objectives, with far fewer trials than manual tuning, A/B tests or grid search.

One Python optimizer serves every integration over the same [iBO protocol](protocol/README.md). iBO continues
*Bayesian Optimization for Unity*; if you used it, see the [migration guide](docs/migration-to-ibo.md).

## At a glance

| Part | What you get | Folder | Start here |
|---|---|---|---|
| **Unity** | C# package configured in the Inspector, analytic and rendering samples; optional QuestionnaireToolkit add-on with questionnaire studies and a Fitts's law study | `Unity/` | [Unity guide](Unity/README.md), [manual](docs/unity-manual.md) |
| **Unreal Engine** | C++ plugin with a Blueprint component, analytic example actor, Python launcher | `Unreal/` | [Unreal guide](Unreal/README.md) |
| **Python** | `interactive_bo`: client and server launcher for scripts, PsychoPy or lab hardware | `Python/` | [Python guide](Python/README.md) |
| **Backend** | The optimizer: BO, MOBO, CABOP, DBO, contextual LCE-M, MetaTAF on BoTorch or botorch-rust; `ibo_server.py` for the Python client and new integrations | `Backend/` | [Backend](Backend/README.md), [architecture](docs/architecture.md) |
| **Protocol** | Message specification and JSON schema for new integrations | `protocol/` | [Protocol](protocol/README.md) |

**Status:** release candidate 2.0.0-rc.1.

- Windows: the Unity and Unreal integrations iBO is built from were validated with Unity 6000.6 and Unreal Engine 5.8.
- macOS: the backend, the Python client and the Unity package (compiled and tested in Unity 6000.4) are tested for 2.0.
- Linux: the backend and Python client tests run in CI.
- Not yet validated: the renamed package in Unity 6000.6 and Unreal Engine 5.8, the engine integrations on macOS and
  Linux, and standalone player builds ([validation record](docs/validation.md)).

## Features

- Single- and multi-objective Bayesian optimization on [BoTorch](https://botorch.org);
  [botorch-rust](https://github.com/M-Colley/botorch-rust), a faster Rust implementation of BoTorch installed with the
  requirements, can run the models instead.
- Cost-aware optimization ([CABOP](https://dl.acm.org/doi/full/10.1145/3772318.3791024)) when design changes have different costs.
- Dynamic BO ([DBO](https://doi.org/10.1109/LRA.2026.3665072)) for objectives that drift during a session (learning, fatigue).
- Contextual optimization with a latent context embedding multi-task GP (LCE-M) to reuse data across users, devices or environments.
- Transfer from prior participants (MetaTAF).
- Explicit feedback (questionnaires; QuestionnaireToolkit add-on for Unity) and implicit feedback (telemetry, behavior).
- CSV logging per participant and condition, warm starts, reproducible seeds, concurrent studies on separate ports.

## Quick start

The optimizer runs on **Python 3.13** with the pinned packages of [`Backend/requirements.txt`](Backend/requirements.txt)
(several GB with PyTorch). Unity installs them by itself on the first Play. For Unreal and Python, create an environment
once (on Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`):

```bash
python3.13 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r Backend/requirements.txt
```

**Unity**: open `Unity/` in Unity 6000.6 (tested; the package also compiles and passes its EditMode tests on 6000.4),
open `Assets/iBO/Samples/Analytic/BO-analytic-demo.unity` and press Play. The questionnaire examples
(`BO-example-scene`, the Fitts's law study and others) are in `Assets/iBO-QuestionnaireToolkit/Samples/Scenes`.

**Unreal**: copy `Unreal/Plugins/BayesianOptimization` into the `Plugins` folder of a C++ project, start the backend
with `.venv/bin/python Unreal/launch_backend.py --backend botorch --objectives 1`, add a `BoAnalyticDemoActor` to a
level and press Play ([Unreal guide](Unreal/README.md)).

**Python**: install the client (it needs only Python 3.9 or newer) and run a study with a simulated participant:

```bash
python -m pip install ./Python
```

```bash
ibo demo --python .venv/bin/python
```

![The Unity integration in action](images/BOforUnity.gif)

## Use iBO in your own project

| Integration | What you need | Guide |
|---|---|---|
| Unity | `Assets/iBO` and `Assets/StreamingAssets/BOData` from `Unity/`; for questionnaires also `Assets/iBO-QuestionnaireToolkit` | [Unity guide](Unity/README.md#use-it-in-your-own-project) |
| Unreal Engine | The plugin `Plugins/BayesianOptimization`, plus the rest of `Unreal/` (launcher, backend copy) in a writable folder | [Unreal guide](Unreal/README.md) |
| Any other program | The Python client, or your own client of the iBO protocol | [Python guide](Python/README.md), [protocol](protocol/README.md) |

From 2.0 on, every [release](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/releases) carries
`iBO-Unity-<version>.zip` and `iBO-Unreal-<version>.zip`. The QuestionnaireToolkit add-on and the Python package come
from this repository (the package is not on PyPI).

## Choosing an optimizer

| Your study | Backend (`optimizerBackend`) | Details |
|---|---|---|
| One objective, or the Pareto front of several (default) | BoTorch (`botorch`): BO for one objective, MOBO for several | [Manual 8.7](docs/unity-manual.md#87-optimizer-backend-and-cabop-settings) |
| Changing a design costs differently depending on what changes (e.g. swapping hardware) | CABOP (`cabop`) | [Manual 8.7](docs/unity-manual.md#87-optimizer-backend-and-cabop-settings) |
| One objective that drifts during a session (learning, fatigue) | DBO (`dbo`) | [Manual 8.15](docs/unity-manual.md#815-dynamic-bo-dbo-optimizing-a-drifting-objective), [DBO guide](docs/dbo-backend.md) |
| Several objectives, with data of earlier participants to start from | MetaTAF (`meta-taf`) | [Manual 8.14](docs/unity-manual.md#814-meta-bo-metataf-transfer-from-prior-participants), [MetaTAF guide](docs/meta-taf-student-guide.md) |
| Users, devices or environments that differ but share structure | Contextual optimization (LCE-M) with BoTorch | [Manual 8.13](docs/unity-manual.md#813-contextual-optimization-and-context-embeddings-lce-m-gp) |

## Data and logs

Every run writes semicolon-separated CSV files (observations, Pareto flags or best value, hypervolume, timings) and
its provenance to `<log root>/<user ID>/<condition ID>/run/`
([manual 8.12](docs/unity-manual.md#812-output-files-and-metrics)). Everything stays on the local machine; use
pseudonymous participant IDs.

| Integration | Log root |
|---|---|
| Unity | `Assets/StreamingAssets/BOData/LogData` (below `persistentDataPath` when StreamingAssets is read-only) |
| Unreal Engine | `--log-root` of `launch_backend.py` (default `~/BayesianOptimizationRuns`) |
| Python | `log_root` or `--log-root` (default `./LogData`) |

## Repository layout

| Folder | Content |
|---|---|
| `Backend/` | The optimizer: backends, iBO server (`ibo_server.py`), requirements, licenses, example data |
| `Unity/`, `Unreal/`, `Python/` | Integrations; `Unity/` and `Unreal/` carry generated copies of `Backend/` |
| `protocol/` | iBO protocol specification and JSON schema |
| `tools/` | `sync_backends.py` (engine copies of `Backend/`) and release packaging |
| `tests/` | Python tests of the backend and the release tools |
| `docs/` | Manual, migration guide, architecture, validation, backend guides, [roadmap](docs/ibo/ROADMAP.md) |
| `images/` | Figures of the README and the manual |
| `.github/workflows/` | CI checks and the release workflow |

## Contributing

Report bugs and ideas in the [issues](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/issues); pull
requests are welcome. Change the optimizer only in `Backend/`, then regenerate and verify the engine copies and run the
tests. They need Python 3.13 with numpy, pandas, pytest and jsonschema; with the full stack of `Backend/requirements.txt`
the tests that run real optimizers are included.

```bash
python tools/sync_backends.py
```

```bash
python tools/sync_backends.py --check
```

```bash
python -m pytest -q tests
```

```bash
python -m unittest discover Python/tests
```

## Documentation

- [Unity guide](Unity/README.md) and [Unity manual](docs/unity-manual.md): concepts, every setting, logging, troubleshooting
- [Unreal guide](Unreal/README.md), [Python guide](Python/README.md)
- [Migration guide](docs/migration-to-ibo.md) from Bayesian Optimization for Unity 1.x
- [iBO protocol](protocol/README.md) for new integrations
- [Architecture](docs/architecture.md), [validation record](docs/validation.md), [render pipelines](Unity/Assets/iBO/Documentation/render-pipeline-support.md)
- Backend guides: [Dynamic BO](docs/dbo-backend.md), [MetaTAF](docs/meta-taf-student-guide.md)
- [Roadmap](docs/ibo/ROADMAP.md), [changelog](CHANGELOG.md)

## Publications

Scientific publications built on iBO (as Bayesian Optimization for Unity):

**2026**

[BlurDriving: Investigating How Personalized Blur Techniques Impact Drivers' Performance in Virtual Reality](https://dl.acm.org/doi/10.1145/3831646). In *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*, Vol. 10. **IMWUT**. ACM.

[Comparing Preferences Between Japan and Germany for External Communication of Automated Vehicles Using Bayesian Optimization](https://dl.acm.org/doi/10.1145/3831988). In *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*, Vol. 10. **IMWUT**. ACM.

[Multi-Session User Experience Assessments of Computationally Optimized Automated Vehicle Functionality Visualizations](https://dl.acm.org/doi/10.1145/3828157.3828785). In *Proceedings of the 18th International Conference on Automotive User Interfaces and Interactive Vehicular Applications*. **AutomotiveUI '26**. ACM.

[MoTUI: Personalization of In-Vehicle Tactile Interfaces for People With Vision Impairments and the Blind](https://pascal-jansen.github.io/data/publications/MoTUI.pdf). In *Proceedings of the 39th Annual ACM Symposium on User Interface Software and Technology*. **UIST '26**. ACM.

[ProVoice: Designing proactive functionality for in-vehicle conversational assistants using multi-objective Bayesian optimization to enhance driver experience](https://dl.acm.org/doi/full/10.1145/3772318.3791877). In *Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems*. **CHI '26**. ACM.

**2025**

[OptiCarVis: Improving automated vehicle functionality visualizations using Bayesian optimization to enhance user experience](https://dl.acm.org/doi/full/10.1145/3706598.3713514). In *Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems*. **CHI '25**. ACM.
  **Best Paper Honorable Mention (top 5%)**

[Improving external communication of automated vehicles using Bayesian optimization](https://dl.acm.org/doi/full/10.1145/3706598.3714187). In *Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems*. **CHI '25**. ACM.

[Fly Away: Evaluating the impact of motion fidelity on optimized user interface design via Bayesian optimization in automated urban air mobility simulations](https://dl.acm.org/doi/full/10.1145/3706598.3713288). In *Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems*. **CHI '25**. ACM.

## Citation

If you use iBO in research, please cite it:

```bibtex
@software{jansen_interactive_bayesian_optimization,
  author    = {Pascal Jansen and Mark Colley},
  title     = {Interactive Bayesian Optimization (iBO), formerly Bayesian Optimization for Unity},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19786494},
  url       = {https://doi.org/10.5281/zenodo.19786494}
}
```

## License

MIT, see [LICENSE](LICENSE). Vendored components keep their licenses; see [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).
The QuestionnaireToolkit add-on has open license items (see its
[Third-Party Notices](Unity/Assets/iBO-QuestionnaireToolkit/Third-Party%20Notices.txt)) and is not part of the release
packages. The multi-engine layout and the Unreal integration come from
[Bayesian-Optimization-for-Game-Engines](https://github.com/M-Colley/Bayesian-Optimization-for-Game-Engines).
Parts of the code were written with AI coding assistants (OpenAI Codex, Anthropic Claude).
Unity, Unreal Engine, Python and PyTorch are trademarks of their respective owners; their logos only show
which tools iBO works with.
