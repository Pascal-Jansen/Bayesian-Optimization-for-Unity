# iBO for Unity

This folder is the Unity package of iBO (Interactive Bayesian Optimization). The repository with the manual, the
other integrations and the releases: https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity

## Requirements

- Unity 6000.6 (the project uses 6000.6.7f1; the 2.0 package has so far been compiled and tested on 6000.4).
- The packages `com.unity.nuget.newtonsoft-json` 3.2.2 and `com.unity.ugui` 2.6.0. The tests also need
  `com.unity.test-framework` 1.8.0.
- Python 3.13. On the first Play, iBO installs the pinned packages of `requirements.txt` into a private virtual
  environment in `<persistentDataPath>/BOData/python-venv` (needs internet; with PyTorch this can take several GB).
  If Python is not found automatically, select it under **Python Settings > Manually Installed Python**
  ([manual 8.5](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/blob/main/docs/unity-manual.md#85-python-settings)).

## Try it

- **Core sample:** open `Assets/iBO/Samples/Analytic/BO-analytic-demo.unity` (or recreate it with **Tools > iBO >
  Create Analytic Demo**) and press Play. It minimizes `(x - 0.25)^2 + (y - 0.75)^2` over two parameters and shows the
  completed evaluations and the best cost.
- **Questionnaire studies:** the scenes of the QuestionnaireToolkit add-on in
  `Assets/iBO-QuestionnaireToolkit/Samples/Scenes`: a multi-objective questionnaire study, color matching, a click game
  and a Fitts's law study. Accept the TextMesh Pro essentials when Unity offers them, otherwise their text stays empty.

## Use it in your own project

Copy into your project's `Assets` folder (from this project or from the `iBO-Unity-<version>.zip` of a release):

- `Assets/iBO`: scripts, editor tools, samples and tests,
- `Assets/StreamingAssets/BOData` without `LogData`: the Python backend, its requirements and example data,
- `Assets/iBO-QuestionnaireToolkit` if you need questionnaires; the add-on is only available from the repository.

Then put a `BoForUnityManager` and its companion components on an object
([manual section 4](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/blob/main/docs/unity-manual.md#4-integration-checklist)), set the parameter keys and bounds and
the objective keys, bounds and directions, apply each suggested design, and report every objective before the next
iteration. The analytic sample is the smallest complete example. Projects that used Bayesian Optimization for Unity
1.x follow the [migration guide](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/blob/main/docs/migration-to-ibo.md).

## Notes

- Questionnaires connect through `IStudyQuestionnaire` and `IQuestionnaireOptimizationBridge` (assembly
  `BOforUnity.Shared`, namespace `BayesianOptimization.Integration`); the core contains no QuestionnaireToolkit code.
- The core compiles with the built-in render pipeline, URP and HDRP. **Tools > iBO > Rendering** creates Lit and Unlit
  sample materials for the active pipeline ([render pipeline support](render-pipeline-support.md)).
- **Optimizer TCP Port** (default 56001) reaches Python as `BO_PORT`; give concurrent studies different ports.
- Logs go to `Assets/StreamingAssets/BOData/LogData/<User ID>/<Condition ID>/run/`, or below `persistentDataPath`
  when StreamingAssets is read-only ([manual 8.12](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/blob/main/docs/unity-manual.md#812-output-files-and-metrics)).

See also the [validation record](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/blob/main/docs/validation.md), [render pipeline support](render-pipeline-support.md) and the
[publication assessment](publication-status.md).
