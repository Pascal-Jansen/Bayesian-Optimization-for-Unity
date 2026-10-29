<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/ibo-logo-dark.svg">
    <img alt="iBO · Interactive Bayesian Optimization" src="images/ibo-logo-light.svg" width="460">
  </picture>
</h1>

<p align="center">
  <b>Human-in-the-loop Bayesian optimization for any interactive system.</b><br>
  Ready for Unity, Unreal Engine and Python; anything else connects through an open protocol.
</p>

<p align="center">
  <a href="Unity/README.md"><img alt="Unity 6000.6" src="https://img.shields.io/badge/Unity-6000.6-000000?logo=unity&amp;logoColor=white"></a>
  <a href="Unreal/README.md"><img alt="Unreal Engine 5.8" src="https://img.shields.io/badge/Unreal%20Engine-5.8-0E1128?logo=unrealengine&amp;logoColor=white"></a>
  <a href="Python/README.md"><img alt="Python 3.13" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="protocol/README.md"><img alt="iBO protocol: JSON over TCP" src="https://img.shields.io/badge/iBO%20protocol-JSON%20over%20TCP-6E7781"></a>
  <a href="https://botorch.org"><img alt="BoTorch 0.18" src="https://img.shields.io/badge/BoTorch-0.18-EE4C2C?logo=pytorch&amp;logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue"></a>
  <a href="https://doi.org/10.5281/zenodo.16849778"><img alt="DOI 10.5281/zenodo.16849778" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16849778.svg"></a>
  <a href="https://github.com/Pascal-Jansen/Interactive-Bayesian-Optimization/actions/workflows/python-checks.yml"><img alt="Python checks" src="https://github.com/Pascal-Jansen/Interactive-Bayesian-Optimization/actions/workflows/python-checks.yml/badge.svg"></a>
</p>

<p align="center">
  <b><a href="https://pascal-jansen.github.io">Pascal Jansen</a></b><sup>1</sup>&nbsp;&nbsp;·&nbsp;&nbsp;<b><a href="https://m-colley.github.io">Mark Colley</a></b><sup>2</sup><br>
  <sup>1</sup>Ulm University&nbsp;&nbsp;·&nbsp;&nbsp;<sup>2</sup>University College London
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/ibo-roles-dark.gif">
  <img alt="Animation of the roles in an iBO study: the designer sets up what may change and what counts as good; the interactive system shows each design; the participant uses it and rates it while their behavior is measured; iBO learns from every response and proposes the next design; at the end the designer gets the best designs and the logs." src="images/ibo-roles-light.gif">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/ibo-overview-dark.svg">
  <img alt="iBO overview: any front end (Unity, Unreal Engine, Python or your own tool) exchanges designs and objectives with the iBO backend over the iBO protocol; the backend offers BO, MOBO, CABOP, DBO, contextual LCE-M and MetaTAF" src="images/ibo-overview-light.svg">
</picture>

## What iBO does

You declare **design parameters** (what the system may change, such as sizes, colors or timings) and **objectives**
(what a person's ratings or behavior score, such as usability or task time). iBO's Bayesian optimizer then proposes the
next design to show after every evaluation. It finds good designs, or the **Pareto front** of best trade-offs between
several objectives, with far fewer trials than manual tuning, A/B tests or grid search.

iBO is not tied to an engine: the optimizer runs as its own process, and whatever shows the designs talks to it over
a small open protocol. iBO continues *Bayesian Optimization for Unity*: its last version, 1.9.1, stays
available in the [`BOforUnity`](https://github.com/Pascal-Jansen/Interactive-Bayesian-Optimization/tree/BOforUnity)
branch, and the [migration guide](docs/migration-to-ibo.md) explains the move to iBO.

## How iBO works

Every study runs the loop shown at the top: iBO proposes a design, the participant uses it, and their rating and
behavior go back to iBO, which updates its estimate of what works. The first few designs are spread out to get an
overview (**sampling**); after that, iBO picks designs where its estimate is high or still uncertain
(**optimization**). All animations in this README are recorded iBO runs with simulated participants.

### One parameter, one rating

One design parameter, for example text size, and one rating. The dashed curve is what the participant would rate,
which iBO never sees; the blue line is iBO's estimate and the band its uncertainty. After three spread-out designs,
iBO found the peak with its 6th rating, then checked two regions it was still unsure about and spent the last four
ratings on the peak.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/ibo-one-parameter-dark.gif">
  <img alt="Animation of an iBO run with one design parameter: after three spread-out designs, iBO's estimate and its uncertainty guide each next design, and the ratings settle at the best design within 12 ratings." src="images/ibo-one-parameter-light.gif">
</picture>

### Two goals that conflict

Comfort and speed pull in different directions, so there is no single best design but a set of best trade-offs, the
Pareto front: designs that no other design beats on both goals. The dotted line links each design iBO shows (left) to
its result (right). With multi-objective BO, 12 of the 20 designs are best trade-offs, all close to the best possible
ones; which compromise to use is then your decision.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/ibo-trade-offs-dark.gif">
  <img alt="Animation of a multi-objective iBO run: designs on the left are linked to their comfort and speed results on the right; the best trade-offs line up along the hidden best possible curve." src="images/ibo-trade-offs-light.gif">
</picture>

### Preferences that change during a session

People adapt, learn and tire, so the best design moves during a session (the gold band). Both optimizers start with
the same spread-out designs. Dynamic BO (DBO) trusts recent ratings more and follows the moving best design: 15 of its
20 model-guided designs land in the gold band (within 0.1 of the best design at that moment), against 7 for standard
BO, which weighs old and new ratings alike.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/ibo-drift-dark.gif">
  <img alt="Animation comparing Dynamic BO and standard BO while the best design moves during a session: DBO's designs follow the moving best design, standard BO's designs scatter." src="images/ibo-drift-light.gif">
</picture>

## One optimizer, any front end

Unity, Unreal Engine and Python come with ready-made integrations. Anything else that can open a local TCP connection
and write JSON can run a study too: a VR or driving simulator, an experiment program, a web service, a robot or
haptics controller. Every front end gets the same methods and the same CSV logs.

| Front end | What you use | Start here |
|---|---|---|
| <img src="images/front-end-unity.svg" width="20" height="20" align="center" alt="">&nbsp;**Unity** | C# package: copy `Assets/iBO` and `Assets/StreamingAssets/BOData` into your project and set it up in the Inspector; samples; optional QuestionnaireToolkit add-on with questionnaire studies and a Fitts's law study | [Unity guide](Unity/README.md), [manual](docs/unity-manual.md) |
| <img src="images/front-end-unreal.svg" width="20" height="20" align="center" alt="">&nbsp;**Unreal Engine** | C++ plugin with a Blueprint component and an analytic example actor; the rest of `Unreal/` (launcher, backend copy) stays in a writable folder | [Unreal guide](Unreal/README.md) |
| <img src="images/front-end-python.svg" width="20" height="20" align="center" alt="">&nbsp;**Python programs**: scripts, PsychoPy, lab hardware | `interactive_bo`: client and server launcher | [Python guide](Python/README.md) |
| <img src="images/front-end-protocol.svg" width="20" height="20" align="center" alt="">&nbsp;**Your own tool**: Godot, C++, Java, Node.js, MATLAB, R, ... | Start the server (`ibo serve`) and speak the iBO protocol in any language with TCP sockets and JSON; a JSON schema is included | [Protocol](protocol/README.md) |

This is a complete study on the wire, recorded from iBO (`→` to iBO, `←` from iBO; each message is one line, the
first is wrapped here, numbers are shortened):

```text
→ {"type": "init", "parameters": [{"key": "size", "init": {"low": 0.2, "high": 2.0}}],
   "objectives": [{"key": "usability", "init": {"low": 0, "high": 100, "minimize": 0}}],
   "config": {"nParameters": 1, "nObjectives": 1, "numSamplingIterations": 2, "numOptimizationIterations": 1}}
← {"type": "parameters", "values": {"size": 0.32}, "iteration": 1}
→ {"type": "objectives", "values": {"usability": 75.9}}
← {"type": "tempCoverage", "value": 0.5}
← {"type": "parameters", "values": {"size": 1.26}, "iteration": 2}
→ {"type": "objectives", "values": {"usability": 99.0}}
← {"type": "tempCoverage", "value": 1.0}
← {"type": "coverage", "value": 0.98}
← {"type": "parameters", "values": {"size": 1.60}, "iteration": 3}
→ {"type": "objectives", "values": {"usability": 90.0}}
← {"type": "coverage", "value": 0.98}
← {"type": "optimization_finished"}
```

`tempCoverage` (share of the sampling phase done) and `coverage` (best value so far, on iBO's internal −1 to 1 scale)
report progress and can be ignored. The optimizer behind every front end is in `Backend/`
([Backend](Backend/README.md), [architecture](docs/architecture.md)).

Every iBO [release](https://github.com/Pascal-Jansen/Interactive-Bayesian-Optimization/releases) carries
`iBO-Unity-<version>.zip` and `iBO-Unreal-<version>.zip`. The QuestionnaireToolkit add-on and the Python package come
from this repository (the package is not on PyPI).

## Features

### What you can optimize

- **Design parameters.** Any number of numeric parameters, each a range in its own units: a size in pixels, a color
  value from 0 to 1, a delay in seconds. iBO proposes values within these ranges and normalizes them internally.
  Parameters are continuous; for whole numbers or a fixed set of options, round or map the value in your front end.
  CABOP can also snap proposals to a list of values that already exist, such as prototypes that were already built.
- **Objectives.** One objective, or several that compete, such as usability against task time. Each objective has its
  own range and direction (larger or smaller is better). With one objective, iBO searches for the best design; with
  several, it searches for the **Pareto front**, the designs where no objective can improve without another getting
  worse.
- **Budget.** A study starts with **sampling** (spread-out designs for an overview; by default 2 × (number of
  parameters + 1)) and continues with **optimization** iterations, in which the model picks each design. You set both
  counts. A warm start from earlier data skips the sampling.

### Feedback from people

- **Ratings** (explicit feedback): questionnaire answers, such as a usability score or a single preference rating.
  Several items that belong to one objective are averaged. In Unity, the optional QuestionnaireToolkit add-on shows
  questionnaires inside the scene, on desktop and in VR, and can mark a participant's previous answer as a subtle hint.
- **Behavior** (implicit feedback): anything your system measures, such as task time, errors, accuracy or sensor and
  physiological signals. It is sent as an objective value, like a rating.
- Ratings and behavior can be combined in one study, for example a rating to maximize and a task time to minimize.

### Optimization methods

All methods run in the same backend and are selected with one setting. [Choosing an optimizer](#choosing-an-optimizer)
shows which one fits your study.

- **BO** (one objective): a Gaussian process model with BoTorch's qLogNoisyExpectedImprovement acquisition function.
- **MOBO** (two or more objectives): a Gaussian process per objective with qLogNoisyExpectedHypervolumeImprovement;
  [moocore](https://github.com/multi-objective/moocore) computes the Pareto front and its hypervolume.
- **CABOP** (cost-aware): for studies where changes cost different amounts, such as swapping a physical part versus
  moving a slider. It weighs the expected improvement of a design against the cost of the change, groups parameters by
  cost, can stop at a cost budget and snaps to existing prototypes. Works with one objective or several combined into
  one score. Based on [Langerak et al., CHI 2026](https://dl.acm.org/doi/full/10.1145/3772318.3791024).
- **DBO** (drifting preferences): for one objective that changes during a session because participants learn, adapt
  or tire. It estimates from the data how quickly older responses lose their relevance; one switch turns it into
  standard BO for a control condition. Based on
  [Kim and Sergi, IEEE RA-L 2026](https://doi.org/10.1109/LRA.2026.3665072).
- **MetaTAF** (transfer from earlier participants): for two or more objectives. Each new participant starts from
  models of earlier participants' runs, so good designs appear after fewer evaluations. Needs the `openbo` package.
- **Contextual optimization** (LCE-M, with BO and MOBO): one model for several users, devices or environments that
  differ but share structure. Each context is described by an embedding that iBO learns from the data, by your own
  feature vector, or by an image that an open_clip vision transformer (ViT-G/14 by default) turns into features.
- **Model library:** BO, MOBO and MetaTAF run on BoTorch or on [botorch-rust](https://github.com/M-Colley/botorch-rust),
  BoTorch's API implemented in Rust and several times faster per suggestion. DBO always uses botorch-rust.

### Running a study

- **Participants and conditions:** every study carries a user, condition and group ID. Each participant and condition
  gets its own log folder, and the IDs are on every logged row, for within- and between-subject designs.
- **Warm start:** continue from earlier data, such as a pilot study or a previous session, given as CSV files of
  parameters and objectives.
- **Stopping early:** a front end can stop a study at any time, and every completed evaluation stays logged. Unity can
  also stop when a participant gives a perfect rating.
- **Several studies at once:** each study runs in its own backend process on its own port.
- **Reproducibility:** a fixed random seed; BO and MOBO runs also record the model library, package versions and seed.
- **Unity study flow:** the next design starts with a button, from your own code or automatically after a delay. An
  optional final round shows the participant the best design found, without sending it to the optimizer. Works with
  the built-in render pipeline, URP and HDRP.
- **Python setup in Unity:** the first Play installs the pinned packages into a private environment (Python 3.13 must
  be installed). Labs without internet access can provide the packages as local wheel files.

### Data

- Semicolon-separated CSV files per participant and condition: every design with its parameter values, objective
  values, phase (sampling or optimization) and whether it is on the Pareto front or the best so far; the hypervolume or
  best value after each evaluation; computation times; and the backend's console output. DBO and MetaTAF add their own
  diagnostics. See [Data and logs](#data-and-logs).
- All data stays on the local machine.

### Limits

- **One design at a time:** every proposal waits for the person's response; there are no parallel batches.
- **No constraints between parameters:** each parameter has its own range, and combinations cannot be excluded.
- **Made for studies with people:** tens of evaluations per participant, each costing a person's time. For thousands of
  cheap, automatic evaluations, a general optimization library is the better fit.
- **Python backend:** it needs Python 3.13 with the pinned packages (several GB with PyTorch) and runs next to your
  application; the engine integrations do not bundle it.

## Status

Version 2.0.0. There was no iBO 1.x: the numbering continues that of Bayesian Optimization for Unity (1.0 to
1.9.1).

- Windows: [Bayesian-Optimization-for-Game-Engines](https://github.com/M-Colley/Bayesian-Optimization-for-Game-Engines),
  where iBO's layout and Unreal plugin come from, was validated before the renaming with Unity 6000.6 and Unreal
  Engine 5.8.
- macOS: the backend and Python client tests pass, real BO, MOBO and DBO studies ran through the Python
  client, and the Unity package compiles and passes its EditMode tests in Unity 6000.4; no Play-mode run with Python yet.
- Linux: the backend and Python client tests run in CI.
- Not yet validated: the renamed package in Unity 6000.6 and Unreal Engine 5.8, the engine integrations on macOS and
  Linux, and standalone player builds ([validation record](docs/validation.md)).

## Quick start

Run the commands from the repository root. The optimizer needs **Python 3.13** with the pinned packages of
[`Backend/requirements.txt`](Backend/requirements.txt) (several GB with PyTorch).

**Unity**: install Python 3.13; Unity installs the packages into a private environment on the first Play. Open
`Unity/` in Unity 6000.6 (the project's version; the iBO package has so far been tested in 6000.4), open
`Assets/iBO/Samples/Analytic/BO-analytic-demo.unity` and press Play. The questionnaire examples (`BO-example-scene`,
the Fitts's law study and others) are in `Assets/iBO-QuestionnaireToolkit/Samples/Scenes`.

For **Unreal**, **Python** and **your own tool**, create an environment once (on Windows: `py -3.13 -m venv .venv`,
then `.venv\Scripts\...` instead of `.venv/bin/...`):

```bash
python3.13 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r Backend/requirements.txt ./Python
```

**Python**: run a complete study with a simulated participant:

```bash
.venv/bin/ibo demo
```

**Unreal**: copy `Unreal/Plugins/BayesianOptimization` into the `Plugins` folder of a C++ project and build it, start
the backend with `.venv/bin/python Unreal/launch_backend.py --backend botorch --objectives 1`, add a
`BoAnalyticDemoActor` to a level and press Play ([Unreal guide](Unreal/README.md)).

**Your own tool**: start a server with `.venv/bin/ibo serve`, connect to `127.0.0.1:56001` and exchange the messages
shown [above](#one-optimizer-any-front-end) ([protocol](protocol/README.md)).

## Choosing an optimizer

| Your study | Backend (`optimizerBackend`) | Details |
|---|---|---|
| One objective, or the Pareto front of several (default) | BoTorch (`botorch`): BO for one objective, MOBO for several | [Manual 8.7](docs/unity-manual.md#87-optimizer-backend-and-cabop-settings) |
| Changing a design costs differently depending on what changes (e.g. swapping hardware) | CABOP (`cabop`) | [Manual 8.7](docs/unity-manual.md#87-optimizer-backend-and-cabop-settings) |
| One objective that drifts during a session (learning, fatigue) | DBO (`dbo`) | [Manual 8.15](docs/unity-manual.md#815-dynamic-bo-dbo-optimizing-a-drifting-objective), [DBO guide](docs/dbo-backend.md) |
| Several objectives, with data of earlier participants to start from | MetaTAF (`meta-taf`; needs the `open-bo` package, see the guide) | [Manual 8.14](docs/unity-manual.md#814-meta-bo-metataf-transfer-from-prior-participants), [MetaTAF guide](docs/meta-taf-student-guide.md) |
| Users, devices or environments that differ but share structure | Contextual optimization (LCE-M): `botorch` with a `context` in the init message (Unity: **Enable Contextual Optimization** in the Inspector) | [Manual 8.13](docs/unity-manual.md#813-contextual-optimization-and-context-embeddings-lce-m-gp) |

## Data and logs

Every run writes semicolon-separated CSV files (observations, Pareto flags or best value, hypervolume, timings) to
`<log root>/<user ID>/<condition ID>/run/`
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
| `tools/` | `sync_backends.py` (engine copies of `Backend/`), release packaging, and `readme_figures/` (the README figures, drawn from recorded iBO runs) |
| `tests/` | Python tests of the backend and the release tools |
| `docs/` | Manual, migration guide, architecture, validation, backend guides, [roadmap](docs/ibo/ROADMAP.md) |
| `images/` | Figures of the README and the manual |
| `.github/workflows/` | CI checks and the release workflow |

## Contributing

Report bugs and ideas in the [issues](https://github.com/Pascal-Jansen/Interactive-Bayesian-Optimization/issues); pull
requests are welcome. Change the optimizer only in `Backend/`, then regenerate and verify the engine copies and run the
tests, as CI does. They need Python 3.13 with numpy, pandas, pytest and jsonschema. Tests that run real optimizers
need the full stack of `Backend/requirements.txt`: the backend tests use it when it is installed, the Python client's
when `IBO_TEST_PYTHON` names that interpreter, and the Unreal launcher's when `BO_RUN_INTEGRATION_TESTS=1` is set.

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
python -m pytest -q docs/standalone-bridge
```

```bash
python -m unittest discover -s Unreal/tests
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

**8 papers** at CHI, UIST, IMWUT and AutomotiveUI used iBO (as Bayesian Optimization for Unity) to optimize interfaces
with people in the loop.

| Year | Venue | Paper |
|---|---|---|
| 2026 | **[IMWUT](https://dl.acm.org/journal/imwut)** | [**BlurDriving**: Investigating How Personalized Blur Techniques Impact Drivers' Performance in Virtual Reality](https://dl.acm.org/doi/10.1145/3831646) |
| | **[IMWUT](https://dl.acm.org/journal/imwut)** | [Comparing Preferences Between Japan and Germany for External Communication of Automated Vehicles Using Bayesian Optimization](https://dl.acm.org/doi/10.1145/3831988) |
| | **[AutomotiveUI](https://www.auto-ui.org/26/)** | [Multi-Session User Experience Assessments of Computationally Optimized Automated Vehicle Functionality Visualizations](https://dl.acm.org/doi/10.1145/3828157.3828785) |
| | **[UIST](https://uist.acm.org/2026/)** | [**MoTUI**: Personalization of In-Vehicle Tactile Interfaces for People With Vision Impairments and the Blind](https://pascal-jansen.github.io/data/publications/MoTUI.pdf) |
| | **[CHI](https://chi2026.acm.org/)** | [**ProVoice**: Designing proactive functionality for in-vehicle conversational assistants using multi-objective Bayesian optimization to enhance driver experience](https://dl.acm.org/doi/full/10.1145/3772318.3791877) |
| 2025 | **[CHI](https://chi2025.acm.org/)** | [**OptiCarVis**: Improving automated vehicle functionality visualizations using Bayesian optimization to enhance user experience](https://dl.acm.org/doi/full/10.1145/3706598.3713514)<br>🏆 Best Paper Honorable Mention (top 5%) |
| | **[CHI](https://chi2025.acm.org/)** | [Improving external communication of automated vehicles using Bayesian optimization](https://dl.acm.org/doi/full/10.1145/3706598.3714187) |
| | **[CHI](https://chi2025.acm.org/)** | [**Fly Away**: Evaluating the impact of motion fidelity on optimized user interface design via Bayesian optimization in automated urban air mobility simulations](https://dl.acm.org/doi/full/10.1145/3706598.3713288) |

IMWUT: *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*, Vol. 10; all papers are
published by ACM. Used iBO in your research? Open an issue or a pull request to add your paper.

## Citation

If you use iBO in research, please cite it:

```bibtex
@software{jansen_interactive_bayesian_optimization,
  author    = {Pascal Jansen and Mark Colley},
  title     = {Interactive Bayesian Optimization (iBO), formerly Bayesian Optimization for Unity},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.16849778},
  url       = {https://doi.org/10.5281/zenodo.16849778}
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
