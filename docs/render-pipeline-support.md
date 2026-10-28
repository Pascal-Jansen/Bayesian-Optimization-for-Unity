# Render pipeline support

The optimization loop uses engine scripts and screen space UI. It works with the built in render pipeline, Universal Render Pipeline (URP), and High Definition Render Pipeline (HDRP). The rendering integration creates materials for the project's active pipeline. It has no compile dependency on either optional SRP package.

Configure the project with Unity's pipeline tools before creating the rendering examples. URP projects require a URP Asset and renderer. HDRP projects require the HDRP project settings and resources. Unity's [URP documentation](https://docs.unity.com/en-us/engine/6000.0/manual/render-pipelines/universal-render-pipeline/urp-reference) and [HDRP Wizard documentation](https://github.com/Unity-Technologies/Graphics/blob/master/Packages/com.unity.render-pipelines.high-definition/Documentation~/Render-Pipeline-Wizard.md) describe these settings.

## Example materials

Choose `Tools > iBO > Rendering > Generate Materials for Active Pipeline`. The command writes four materials below `Assets/iBOGenerated/Rendering/<pipeline>`. CubeMat and CylinderMat use the pipeline's Lit shader. CubeUnlit and CylinderUnlit use its Unlit shader. The built in Unlit examples use Unity's transparent sprite shader. All four templates support changes to their main color and alpha.

The integration uses `GraphicsSettings.currentRenderPipeline` to respect the active quality level's pipeline override. It then uses the selected shader's public `ShaderGUI.ValidateMaterial` implementation to configure blend states, shader keywords, queues, and passes. The command preserves the project pipeline settings. Repeating it updates its generated materials. Unity documents the [pipeline selection rule](https://docs.unity.com/en-us/engine/6000.7/script-reference/unityengine/rendering/graphicssettings/currentrenderpipeline) and [material validation API](https://docs.unity.com/en-us/engine/6000.0/script-reference/unityeditor/shadergui/validatematerial).

Choose `Create Material Preview Scene` from the same menu to generate a scene with a cube, cylinder, camera, and light. Its objects use Unlit materials so their displayed colors can be inspected without scene lighting. Generated scenes contain direct material references. Unity includes the required shader variants when those scenes are included in a build.

The package includes `Assets/iBO/Samples/Rendering/BuiltIn/BO-rendering-preview.unity`. Open this scene and choose `Prepare Open BO Demo Scene` to use it with the active pipeline. The command updates the package's rendering sample materials and the original CubeMat and CylinderMat references. Save the scene after conversion. Other scene materials remain unchanged.

## Analytic optimization sample

Choose `Tools > iBO > Create Analytic Demo` to create `Assets/iBO/Samples/Analytic/BO-analytic-demo.unity`. This sample uses an automatic quadratic objective and requires no questionnaire assets. Its optimization measurements have no dependency on rendering. The same sample can therefore exercise the Python connection in built in, URP, and HDRP projects.

## Validation

The isolated validation fixture is in `docs/rendering-validation` of the [iBO repository](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity). It creates a disposable project, installs the selected Unity SRP dependency through that project's manifest, and uses a GPU render target to inspect the example colors. The package itself never installs packages or changes the consumer project's pipeline.

Run `python docs/rendering-validation/validate.py --unity <Unity executable> --pipeline Universal` for URP. Select `HighDefinition` for HDRP or `BuiltIn` for a project without either package. Add `--both-packages` to check compilation with both optional packages installed. Add `--full-package` to compile the optimizer without QT, convert its shipped rendering sample, and verify the analytic objective and generated scene. The fixture records `render-validation.json`, a preview PNG, and the editor log in the temporary project. It uses Unity 6000.6.7f1 and SRP 17.6.0. HDRP receives an initial render request to initialize its resources before the image assertions.

Validation passed on Windows with an NVIDIA GeForce RTX 5080. Projects compiled with neither optional SRP package, each package separately, and both packages installed. The full optimizer compiled without QT in the URP project and the combined project with HDRP active. Both converted the shipped rendering scene and generated the analytic scene without missing scripts. The analytic cost was zero at `(0.25, 0.75)` and `0.625` at `(0, 0)`. GPU readback confirmed the Unlit colors and transparency. Lit readback passed the fallback color check. The recorded results are in `docs/rendering-validation/results`.

Pipeline lighting, exposure, and tone mapping affect displayed stimuli. Studies that compare conditions across pipelines need their own visual calibration.

Standalone player builds, other graphics backends, and XR rendering still need verification.
