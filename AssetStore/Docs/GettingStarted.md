# Getting Started with Bayesian Optimization for Unity

Bayesian Optimization for Unity (BOforUnity) runs human-in-the-loop Bayesian optimization inside your Unity scenes. You declare **design parameters** (what Unity may change) and **objectives** (what a participant's response or behavior scores), and a [BoTorch](https://botorch.org)-based Python backend proposes the next design to show after every evaluation. Single- and multi-objective studies are supported (Pareto front, hypervolume), as well as cost-aware (CABOP), drifting-objective (DBO), contextual and transfer (MetaTAF) optimization.

This page covers the Asset Store edition. The [Manual](Manual.html) documents every setting.

## 1. Requirements

| Item | Requirement |
|---|---|
| **Unity** | 6000.4 or newer, Built-in Render Pipeline for the example scenes (the package itself only uses UI and primitives) |
| **Platforms** | Unity Editor and desktop players on Windows 10/11 (x64), macOS 11+ (Apple silicon or Intel) and Linux (x86_64). Mobile, WebGL and consoles are not supported: the optimizer runs as a local Python process. |
| **Python** | Python **3.13** (64-bit), installed by you. It is **not** included in this package. |
| **Internet** | Needed once: on the first Play, BOforUnity downloads the Python packages it needs (PyTorch, BoTorch, GPyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib, moocore, loguru; several hundred MB) into a private virtual environment. Offline labs can provide the packages as wheel files instead (Manual, section 8.5). |
| **Unity packages** | Newtonsoft Json (`com.unity.nuget.newtonsoft-json`) and Unity UI (`com.unity.ugui`), installed with the package; TextMesh Pro Essential Resources (step 2 below). |
| **Input handling** | The example scenes use the legacy input module. Set **Project Settings > Player > Other Settings > Active Input Handling** to **Input Manager (Old)** or **Both**. |

## 2. Installation

1. Import the package from **Window > Package Manager > My Assets** (keep everything selected). After the import, the bundled QuestionnaireToolkit adds its tags (names starting with `QT`) under **Project Settings > Tags and Layers**; the questionnaire items need them.
2. Import the TextMesh Pro essentials when Unity asks for them, or via **Window > TextMeshPro > Import TMP Essential Resources**. Reopen the scene afterwards if texts were missing.
3. Install **Python 3.13** from [python.org/downloads](https://www.python.org/downloads/).
   - Windows: tick **Add python.exe to PATH** in the installer.
   - macOS: run the installer package; Homebrew or pyenv installations work as well.
   - Linux: install `python3.13` and `python3.13-venv` with your package manager.

   If you prefer a specific interpreter, virtual environment or conda environment, select it on the `BOforUnityManager` object under **Python Settings > Manually Installed Python** (Manual, section 8.5).
4. Recommended: keep the project in a folder whose path contains no spaces.

On the first Play, BOforUnity finds Python 3.13, creates its private environment under `<persistentDataPath>/BOData/python-venv` and installs the pinned packages from `Assets/StreamingAssets/BOData/Installation/requirements.txt`. The progress is shown on screen; it takes a few minutes once. Every later start only checks the installed versions (about a second). If no Python 3.13 is found, the screen says so and tells you what to install.

## 3. First Run (about 10 minutes)

1. Open `Assets/BOforUnity/Scenes/BO-example-scene.unity`.
2. Press **Play** and click **Next**. The first time, wait until the Python packages are installed and the message "The system has been started successfully!" appears.
3. Click **Next** to see the first design (one or two colored shapes).
4. Click **End Simulation**, answer the short questionnaire and press **Finish**. The optimizer records your ratings and proposes the next design.
5. Repeat a few iterations. The designs change between iterations as the optimizer learns.

Results are written to `Assets/StreamingAssets/BOData/LogData/<User ID>/<Condition ID>/` (in built players: `<persistentDataPath>/BOData/LogData/...`), including `ObservationsPerEvaluation.csv` and, for multi-objective studies, `HypervolumePerEvaluation.csv`.

## 4. Example Scenes

| Scene | Shows |
|---|---|
| `BO-example-scene` | Multi-objective optimization of a simple design with questionnaire feedback (QuestionnaireToolkit). |
| `BO-single-objective-demo` | Single-objective optimization: the optimizer proposes a color (RGB parameters), the participant rates its similarity on a one-item scale. |
| `BO-fitts-law-task` | A 2D Fitts's law pointing task with HITL MOBO, Static and Random conditions, explicit design parameters and telemetry objectives. |
| `BO-click-game` | A click task whose target size and eccentricity the optimizer adjusts, rated on a one-item scale. |

## 5. Using It in Your Own Scene

1. Drag `Assets/BOforUnity/Prefabs/BOforUnityManager.prefab` and `BOControlCanvas.prefab` into your scene.
2. On the `BOforUnityManager`, define your parameters (key, bounds) and objectives (key, bounds, **Smaller is Better**), the number of iterations and the Python settings.
3. Read the proposed parameter values in your scripts and report the objective values back; the [Manual](Manual.html) (sections 4 and 8.3-8.4) has the integration checklist and code examples. Questionnaire items whose **Header Name** equals an objective key are mapped automatically.

## 6. Folder Layout

| Folder | Content |
|---|---|
| `Assets/BOforUnity/Scripts`, `Editor` | Runtime and inspector code (namespace `BOforUnity`) |
| `Assets/BOforUnity/Prefabs`, `Scenes`, `Examples` | Prefabs, example scenes and their scripts |
| `Assets/BOforUnity/QuestionnaireToolkit` | The bundled QuestionnaireToolkit, adapted for BOforUnity (questionnaire UI, CSV logging) |
| `Assets/BOforUnity/Documentation` | This page, the Manual and the backend guides |
| `Assets/StreamingAssets/BOData` | Python optimizer backends, `requirements.txt`, example warm-start data. Keep this folder at this path: the backends are started from it. |

The Manual describes the GitHub edition of the same code. Where it mentions `Assets/QuestionnaireToolkit`, the folder is `Assets/BOforUnity/QuestionnaireToolkit` in this edition, and the Python installers mentioned in sections 3 and 8.5 are not included here (install Python 3.13 as in section 2 above).

## 7. Troubleshooting

- **"No Python 3.13 or newer was found ... includes no Python installer"**: install Python 3.13 (section 2) or select an interpreter under **Manually Installed Python**, then press Play again.
- **The package installation fails**: the screen shows the reason, the Console the last lines of pip's output, and `<persistentDataPath>/BOData/pip-install.log` the complete log. Check the internet connection, proxy and free disk space.
- **`InvalidOperationException` about `UnityEngine.Input`**: set Active Input Handling to **Input Manager (Old)** or **Both** (section 1).
- **Missing texts**: import the TextMesh Pro Essential Resources (section 2).
- More cases: Manual, section 9.

## 8. Support, Source and Citation

- Questions and bug reports: <https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/issues>
- Source code and changelog: <https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity>
- If you use BOforUnity in research, please cite it: Pascal Jansen and Mark Colley, *Bayesian Optimization for Unity*, Zenodo, [doi:10.5281/zenodo.19786494](https://doi.org/10.5281/zenodo.19786494).

Third-party components and their licenses are listed in `Assets/BOforUnity/Third-Party Notices.txt`.
