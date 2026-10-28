# Asset Store Listing Draft

Copy these texts into the Publisher Portal (draft package > Metadata). Edit them in your own words where needed:
guideline 3.1.a rejects purely AI-generated descriptions, and this draft was written with an AI assistant.

---

## Product name

Bayesian Optimization for Unity

(Third-party products may use "for Unity" to state compatibility; avoid wording that suggests Unity endorses the
asset.)

## Category

Tools > AI-ML Integration (alternative: Tools > Utilities)

## Price

Free (suggested: the same code is MIT-licensed on GitHub; see README.md in this folder, decision 6).

## Summary (one sentence)

Human-in-the-loop Bayesian optimization for user studies and adaptive designs: declare design parameters and objectives in the Inspector and let a BoTorch backend find the best designs, with Pareto fronts, questionnaires and CSV logging included.

## Description

> **Requirements and external dependencies.** BOforUnity runs its optimizer as a local **Python 3.13** process. Python is **not included**: install it from python.org before the first Play. On first use, BOforUnity downloads the Python packages it needs from the Python Package Index (PyTorch, BoTorch, GPyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib, moocore, loguru; several hundred MB, free of charge) into a private virtual environment; offline installation from wheel files is supported. Supported platforms: Unity Editor and desktop players on Windows, macOS and Linux. The example scenes use the legacy input module (Active Input Handling: "Input Manager (Old)" or "Both") and Built-in Render Pipeline materials.
>
> **Third-party software.** This asset includes third-party components under the MIT and BSD 3-Clause licenses (CABOP, dbo_torch, SimpleJSON, Unity Standalone File Browser, Reorderable List, among others). See `Third-Party Notices.txt` in the package for details. The Python packages installed on first use are licensed under their own terms (listed in the same file; moocore is LGPL-2.1-or-later).

**Bayesian Optimization for Unity** brings an end-to-end human-in-the-loop (HITL) optimization workflow into your scenes. Instead of tuning design parameters by hand, running A/B tests or grid searches, you let a Bayesian optimizer propose the next design after every participant response, so good designs are found with far fewer trials.

**How it works**
1. Declare **design parameters** (anything Unity can change: sizes, colors, speeds, timings) and **objectives** (questionnaire scores, task times, error rates, telemetry) on one Inspector component.
2. Press Play. The optimizer proposes a design, your scene shows it, and the participant's ratings or behavior are sent back.
3. After a few iterations you get the best design (single objective) or the **Pareto front** of best trade-offs (multiple objectives), logged to CSV.

**Features**
- Single- and multi-objective Bayesian optimization built on BoTorch (Gaussian-process surrogates, qLogNEHVI acquisition for MOBO, Pareto front and hypervolume via moocore).
- Cost-aware optimization (CABOP) when design changes have different costs.
- Dynamic BO (DBO) for objectives that drift during a session (learning, fatigue), with a stationary baseline for control conditions.
- Contextual optimization with a latent context embedding multi-task GP (LCE-M) to reuse data across users, devices or environments; embeddings learned, supplied, or computed from images.
- Meta-BO transfer from prior participants (MetaTAF).
- Bundled QuestionnaireToolkit for in-scene questionnaires (desktop and VR); question items map to objectives by name. Implicit telemetry objectives work too.
- Warm start from earlier runs, reproducible seeds, configurable sampling and optimization budgets, and validated configuration in the Inspector.
- Automatic CSV logging per participant and condition: parameters, objectives, metric traces and questionnaire answers.
- Example scenes: questionnaire-driven multi-objective optimization, single-objective color matching, a click task, and a complete Fitts's law study with HITL MOBO, Static and Random conditions.
- Offline documentation (Getting Started, full manual, backend guides) and full C# and Python source code.

**Research-proven.** BOforUnity has been used in studies published at CHI, UIST, IMWUT and AutomotiveUI. If you use it in research, please cite it (DOI 10.5281/zenodo.19786494).

## Technical details

- Unity 6000.4 or newer; tested with 6000.4.7f1 and 6000.5.3f1.
- Platforms: Unity Editor and Standalone players on Windows 10/11 x64, macOS 11+ (Apple silicon, Intel), Linux x86_64. Not for mobile, WebGL or consoles.
- Python 3.13 (64-bit) required, installed by the user; Python packages are installed automatically on first use (internet connection needed once, or a local wheel folder).
- Unity package dependencies: Newtonsoft Json (com.unity.nuget.newtonsoft-json), Unity UI (com.unity.ugui), TextMesh Pro essentials.
- Render pipelines: the package code is pipeline-independent (UI and primitives). Example materials target the Built-in Render Pipeline; convert them with the Render Pipeline Converter in URP/HDRP projects.
- Input: example scenes use the legacy Input Manager (set Active Input Handling to "Input Manager (Old)" or "Both").
- Communication: Unity and Python talk over a local TCP socket on 127.0.0.1 (default port 56001); no data leaves the machine apart from the one-time package download.
- Full source code: C# (namespaces `BOforUnity`, `QuestionnaireToolkit`) and Python backends in `Assets/StreamingAssets/BOData`.
- Documentation: `Assets/BOforUnity/Documentation` (HTML, offline) and https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity

## Keywords (up to 15)

bayesian optimization, human-in-the-loop, botorch, multi-objective, pareto, user study, hci, questionnaire, personalization, adaptive ui, gaussian process, optimization, research, experiment, machine learning

## AI disclosure (Publisher Portal "AI description" field, guideline 1.6)

Confirm and adjust before submitting:

> Parts of the C# and Python source code, tests and documentation were written with the help of AI coding assistants (Anthropic Claude via Claude Code, GitHub Copilot). The authors designed the optimization methods, the architecture and the study workflows, and reviewed, edited and tested all AI-assisted changes. No AI-generated art, audio or models are included.

## Media

Sizes from Unity's support pages; check the current templates in the Publisher Portal before uploading.

| Slot | Size | Suggested source in this repository |
|---|---|---|
| Icon | 160 x 160 px, no text | new artwork |
| Card image | 420 x 280 px | new artwork (a Pareto-front motif with a Unity scene) |
| Cover image | 3:2, e.g. 1950 x 1300 px | new artwork |
| Social media image | check the portal template | new artwork |
| Screenshots | high resolution, 16:9 recommended | `images/MOBO_Pareto_Front.png`, `images/HITL.png`, `images/System_Architecture.png`, `images/parameter_settings.png`, `images/objective_settings.png`, Play-mode captures of the four example scenes |
| Video | YouTube or Vimeo URL | https://www.youtube.com/watch?v=J1hrFuiGiRI (shows an older UI: consider recording a new one) |

## Links

- Documentation: https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity#readme
- Support: https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/issues (the portal may also ask for a support email)
- Publisher website: https://pascal-jansen.github.io
