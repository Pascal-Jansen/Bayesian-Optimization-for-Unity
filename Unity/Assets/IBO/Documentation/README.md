# Interactive Bayesian Optimization (IBO) for Unity

Open this directory as a Unity project, or import the Unity asset package from the release. The core uses `BayesianOptimization` namespaces. Applications using `BOforUnity` namespaces must update their imports. Component GUIDs and assembly names are retained for authored project migration.

Before importing the asset package, install `com.unity.nuget.newtonsoft-json` version `3.2.2` and `com.unity.ugui` version `2.6.0` through Package Manager. The release was tested with Unity `6000.6.7f1`. The source project includes `com.unity.test-framework` version `1.8.0` for its tests. Tests are omitted from the standard asset export.

Install Python 3.13 before running optimization. Configure the manager's manually installed Python path when automatic interpreter detection does not find that environment. The integration installs pinned requirements into the configured environment. Internet access and sufficient disk space are required for initial setup. Bundled installers and native questionnaire dialog libraries are excluded.

Create the analytic example through `Tools > IBO > Create Analytic Demo`. Enter Play mode. The example minimizes `(x - 0.25)^2 + (y - 0.75)^2`. It configures two parameters and one objective. The display reports completed evaluations and the best observed cost. CSV logs retain actual parameter and objective values.

For a study, set parameter keys and bounds on `BoForUnityManager`. Add objective keys, bounds, and directions. Apply each suggested design before collecting measurements. Submit every objective value before advancing. The analytic example provides the minimal telemetry integration.

Questionnaire support uses `IStudyQuestionnaire` and `IQuestionnaireOptimizationBridge` from the shared contract assembly. The optional QuestionnaireToolkit add-on in `Assets/IBO-QuestionnaireToolkit` implements them and holds the questionnaire examples: `BO-example-scene`, the single-objective color demo, the click game and the Fitts's law study. The core package contains no QuestionnaireToolkit source or assets.

Select the render pipeline through Unity's project settings. Install URP or HDRP through Package Manager when required. The rendering preview command creates compatible Lit and Unlit materials for the active pipeline. Shader selection and blend state use public material validation. New preview assets are generated separately. The core compiles without either SRP package.

The manager exposes a loopback port with default `56001`. Python receives the same value through `BO_PORT`. Assign different ports to concurrent studies. Logs follow the configured study directory and fall back to a persistent writable location when StreamingAssets is read only.

See the [validation record](validation.md) and [publication assessment](publication-status.md).
