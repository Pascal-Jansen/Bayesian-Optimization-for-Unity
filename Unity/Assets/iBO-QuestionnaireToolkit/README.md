# iBO QuestionnaireToolkit add-on

Optional add-on to the iBO Unity package (`Assets/iBO`): in-scene questionnaires (desktop and VR) whose answers become
iBO objectives, and the questionnaire-based example studies. The iBO core works without it.

| Folder | Content |
|---|---|
| `QuestionnaireToolkit/` | QuestionnaireToolkit by Pascal Jansen, adapted for iBO: `QTQuestionnaireManager` implements `IStudyQuestionnaire` and finds the optimizer through `IQuestionnaireOptimizationBridge` (both in `BOforUnity.Shared`, namespace `BayesianOptimization.Integration`). Its own documentation is in `QuestionnaireToolkit/Documentation`, its demo scenes in `QuestionnaireToolkit/Demo Scenes`. |
| `Samples/Scenes/` | `BO-example-scene` (multi-objective, questionnaire ratings), `BO-single-objective-demo` (color matching, one rating), `BO-click-game` (click task, one rating), `BO-fitts-law-task` (Fitts's law study with HITL MOBO, Static and Random conditions) |
| `Samples/Prefabs/` | `BOforUnityManager` and `BOControlCanvas` as used by the scenes |
| `Samples/LoadingImages/` | Loading animation of the control canvas |

## Use

1. Import the iBO Unity package first, then this add-on.
2. Import the TextMesh Pro essentials when Unity asks for them.
3. Open a scene in `Samples/Scenes` and press Play. The first Play installs the Python packages (see the
   [Unity guide](../../README.md)).

On load, the toolkit adds its tags (names starting with `QT`) to **Project Settings > Tags and Layers**. Questionnaire
items whose **Header Name** equals an objective key report their answer as that objective. The answers are logged next
to the optimization logs in `Assets/StreamingAssets/BOData/LogData/<User ID>/<Condition ID>/`.

The add-on is not part of the release packages (`tools/build_packages.py` leaves it out) until the open license items
in [Third-Party Notices.txt](Third-Party%20Notices.txt) are resolved; use it from this repository.
