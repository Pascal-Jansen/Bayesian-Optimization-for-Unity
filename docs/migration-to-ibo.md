# Migrating from Bayesian Optimization for Unity 1.x to iBO

Interactive Bayesian Optimization (iBO) 2.0 continues Bayesian Optimization for Unity 1.8 with a shared Python backend
and further integrations. Scenes and prefabs keep working: every script keeps its GUID, so Unity still finds the
components. Code that references the package needs the changes below.

## What changed

| Area | 1.x | iBO 2.0 |
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
| Unity version | 6000.x | Tested with 6000.6; the package compiles on 6000.4 |

New in 2.0: a configurable port per study (`BO_PORT`, for concurrent studies), the botorch-rust model library
(`BO_MODEL_LIBRARY`), URP and HDRP material tools, the Unreal plugin, the Python client and `Backend/ibo_server.py`.

## Projects that imported the package

1. Back up the project or commit it.
2. Import the iBO package. Unity moves the files from `Assets/BOforUnity` to `Assets/iBO` because the GUIDs match.
   If you use questionnaires, also import the QuestionnaireToolkit add-on; it replaces `Assets/QuestionnaireToolkit` the same way.
3. Delete what is left in `Assets/BOforUnity` and `Assets/QuestionnaireToolkit` after the import (files that iBO no
   longer ships), and `Assets/StreamingAssets/BOData/Installation/Windows`, `MacOs` and `Linux` (the old installers).
4. Replace `using BOforUnity...` with `using BayesianOptimization...` in your scripts. Code that used
   `QTQuestionnaireManager` from the optimization side can use `StudyQuestionnaires.Resolve()` and `IStudyQuestionnaire`.
5. Replace hard-coded paths that start with `Assets/BOforUnity/`.
6. Install Python 3.13 if you relied on the bundled installer, or select an interpreter under
   **Python Settings > Manually Installed Python**.

Your logs stay where they were: `Assets/StreamingAssets/BOData/LogData` is not part of the package.

## Clones of this repository

- Open `Unity/` instead of the repository root in Unity Hub. The old `Library/`, `Logs/` and `UserSettings/` folders at the
  repository root can be deleted.
- Logs written by 1.x stay in `Assets/StreamingAssets/BOData/LogData` at the repository root (git ignores them). Move them to
  `Unity/Assets/StreamingAssets/BOData/LogData` if you want them next to new runs.
- Edit the optimizer in `Backend/`, never in the engine copies, and run `python tools/sync_backends.py` afterwards.
