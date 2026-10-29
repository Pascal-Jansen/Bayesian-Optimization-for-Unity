# Migrating from Bayesian Optimization for Unity 1.x to iBO

Interactive Bayesian Optimization (iBO) is the successor of Bayesian Optimization for Unity 1.9, with a shared Python
backend and further integrations. There was no iBO 1.x: iBO's version numbers continue those of Bayesian Optimization
for Unity, so its first version is 2.0.0. Scenes and prefabs keep working: every script keeps its GUID, so Unity still finds the
components. Code that references the package needs the changes below. Version 1.9.1 stays available in the
[`BOforUnity`](https://github.com/Pascal-Jansen/Interactive-Bayesian-Optimization/tree/BOforUnity) branch and as the
release v1.9.1.

## What changed

| Area | Bayesian Optimization for Unity 1.x | iBO |
|---|---|---|
| Repository | Unity project at the repository root | Unity project in `Unity/`; open that folder in Unity Hub |
| Package folder | `Assets/BOforUnity` | `Assets/iBO` (same GUIDs) |
| Namespaces | `BOforUnity`, `BOforUnity.Scripts`, `BOforUnity.Examples`, `BOforUnity.Editor` | `BayesianOptimization`, `BayesianOptimization.Scripts`, `BayesianOptimization.Examples`, `BayesianOptimization.Editor` |
| Assemblies | `BOforUnity.Runtime`, `.Editor`, `.Examples` | Unchanged, plus `BOforUnity.Shared` (questionnaire contracts) |
| QuestionnaireToolkit | Bundled and required by the runtime | Optional add-on `Assets/iBO-QuestionnaireToolkit`; the runtime uses `IStudyQuestionnaire` and `IQuestionnaireOptimizationBridge` from `BayesianOptimization.Integration` |
| `LogDataFolderUtility`, `IQuestionnaireOptimizationBridge` | `QuestionnaireToolkit.Scripts` | `BayesianOptimization.Integration` (assembly `BOforUnity.Shared`) |
| Example scenes | `Assets/BOforUnity/Scenes`, prefabs in `Assets/BOforUnity/Prefabs` | `Assets/iBO-QuestionnaireToolkit/Samples/Scenes` and `/Prefabs`; new core samples in `Assets/iBO/Samples` |
| Python | Installers for Windows and macOS bundled | Not bundled: install Python 3.13; the packages are still installed automatically on the first Play |
| Optimizer source | `Assets/StreamingAssets/BOData/BayesianOptimization` | `Backend/`; `tools/sync_backends.py` writes the copies in `Unity/` and `Unreal/` |
| Menus | | `Tools > iBO > ...` (analytic demo, rendering tools) |
| Unity version | 6000.x | The project uses 6000.6; the iBO package has so far been compiled and tested on 6000.4 |

New in iBO: a configurable port per study (`BO_PORT`, for concurrent studies), URP and HDRP material tools, the Unreal
plugin, the Python client and `Backend/ibo_server.py`.

## Projects that imported the package

Unity matches files by GUID and updates such a file where it already is, so importing iBO over 1.x would leave the
updated files in `Assets/BOforUnity` and `Assets/QuestionnaireToolkit` next to new ones in `Assets/iBO`, and the
project would not compile. Remove the old version first:

1. Back up the project or commit it, then close Unity.
2. Delete `Assets/BOforUnity`, `Assets/QuestionnaireToolkit`, and in `Assets/StreamingAssets/BOData` the folders
   `BayesianOptimization` and `Installation` (this removes the old installers; move an offline `Installation/wheels`
   cache aside first). Keep `LogData` (your logs), `InitData` (warm-start files, context images) and `MetaSources`
   (MetaTAF population models).
3. Copy `Assets/iBO` and `Assets/StreamingAssets/BOData` from the iBO Unity project (`Unity/` in this repository, or
   the `iBO-Unity-<version>.zip` of a release) into the project, merging `BOData` with the folders you kept. If you use
   questionnaires, also copy `Assets/iBO-QuestionnaireToolkit` from this repository (the add-on is not part of the
   release packages).
4. Open the project. Scenes and prefabs keep their components: every script kept its GUID.
5. Replace `using BOforUnity...` with `using BayesianOptimization...` in your scripts. Code that used
   `QTQuestionnaireManager` from the optimization side can use `StudyQuestionnaires.Resolve()` and `IStudyQuestionnaire`.
6. Replace hard-coded paths that start with `Assets/BOforUnity/` or `Assets/QuestionnaireToolkit/`, including a
   questionnaire's **Results Save Path** (the iBO default is `Assets/StreamingAssets/BOData/LogData/`).
7. Install Python 3.13 if you relied on the bundled installer, or select an interpreter under
   **Python Settings > Manually Installed Python**.

Your logs stay where they were: `Assets/StreamingAssets/BOData/LogData` is not part of the package.

## Clones of this repository

- Open `Unity/` instead of the repository root in Unity Hub. The old `Library/`, `Logs/` and `UserSettings/` folders at the
  repository root can be deleted.
- Logs written by 1.x stay in `Assets/StreamingAssets/BOData/LogData` at the repository root (git ignores them). Move them to
  `Unity/Assets/StreamingAssets/BOData/LogData` if you want them next to new runs.
- Edit the optimizer in `Backend/`, never in the engine copies, and run `python tools/sync_backends.py` afterwards.
- The product name is now `iBO`, which moves Unity's `persistentDataPath` (from `.../DefaultCompany/BO-for-Unity`
  to `.../DefaultCompany/iBO`): the first Play installs the Python packages again into a new private environment, and
  `pip-install.log` and the backend console log `output.txt` are written there. The old folder can be deleted.
