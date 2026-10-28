// Imported only into the disposable projects created by validate.py.
using System;
using System.IO;
using BayesianOptimization.Editor.Rendering;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
#if BO_VALIDATION_FULL
using BayesianOptimization.Editor;
using BayesianOptimization.Examples;
#endif
#if BO_VALIDATION_URP
using UnityEngine.Rendering.Universal;
#endif
#if BO_VALIDATION_HDRP
using UnityEngine.Rendering.HighDefinition;
#endif

public static class BoRenderingValidation
{
    public static void Run()
    {
        try
        {
            PlayerSettings.colorSpace = ColorSpace.Linear;
            AssetDatabase.CreateFolder("Assets", "ValidationSettings");
            string pipelineName = Environment.GetEnvironmentVariable("BO_RENDER_PIPELINE") ?? "BuiltIn";
            RenderPipelineAsset pipeline = null;
#if BO_VALIDATION_URP
            if (pipelineName == "Universal")
            {
                var renderer = ScriptableObject.CreateInstance<UniversalRendererData>();
                AssetDatabase.CreateAsset(renderer, "Assets/ValidationSettings/URPRenderer.asset");
                pipeline = UniversalRenderPipelineAsset.Create(renderer);
            }
#endif
#if BO_VALIDATION_HDRP
            if (pipelineName == "HighDefinition")
            {
                pipeline = ScriptableObject.CreateInstance<HDRenderPipelineAsset>();
            }
#endif
            if (pipeline != null) AssetDatabase.CreateAsset(pipeline, "Assets/ValidationSettings/Pipeline.asset");
            GraphicsSettings.defaultRenderPipeline = pipeline;
            QualitySettings.renderPipeline = null;
            AssetDatabase.SaveAssets();
            if (BoDemoRendering.ActivePipeline.ToString() != pipelineName) throw new Exception("Incorrect pipeline detection.");
#if BO_VALIDATION_FULL
            var previewScene = UnityEditor.SceneManagement.EditorSceneManager.OpenScene("Assets/BOforUnity/Samples/Rendering/BuiltIn/BO-rendering-preview.unity");
            if (BoDemoRendering.PrepareScene(previewScene, "Assets/Validation/Materials") != 2)
                throw new Exception("Rendering sample conversion did not update both renderers.");
#else
            BoDemoRendering.CreatePreviewScene("Assets/Validation/BO-rendering-preview.unity");
#endif
            Material[] materials = BoDemoRendering.GenerateMaterials("Assets/Validation/Materials");
            if (GraphicsSettings.defaultRenderPipeline != pipeline || QualitySettings.renderPipeline != null)
                throw new Exception("Example generation changed project render pipeline settings.");
            foreach (Material material in materials)
            {
                if (!material.shader.isSupported || material.shader.name.Contains("Error")) throw new Exception("Unsupported demo shader: " + material.shader.name);
                if (material.renderQueue < 3000) throw new Exception("Transparent demo material has an opaque queue: " + material.name);
            }
            Camera camera = Camera.main;
            camera.aspect = 1f;
#if BO_VALIDATION_HDRP
            if (pipelineName == "HighDefinition")
            {
                var cameraData = camera.gameObject.AddComponent<HDAdditionalCameraData>();
                cameraData.clearColorMode = HDAdditionalCameraData.ClearColorMode.Color;
                cameraData.backgroundColorHDR = camera.backgroundColor;
                var volume = new GameObject("Validation Exposure", typeof(Volume)).GetComponent<Volume>();
                volume.isGlobal = true;
                volume.priority = 1000;
                volume.sharedProfile = ScriptableObject.CreateInstance<VolumeProfile>();
                Exposure exposure = volume.sharedProfile.Add<Exposure>();
                exposure.mode.Override(ExposureMode.Fixed);
                exposure.fixedExposure.Override(0f);
            }
#endif
            var target = new RenderTexture(128, 128, 24, RenderTextureFormat.ARGB32);
            target.Create();
            camera.targetTexture = target;
            if (pipeline == null) camera.Render();
            else
            {
                RenderPipeline.SubmitRenderRequest(camera, new RenderPipeline.StandardRequest { destination = target });
                RenderPipeline.SubmitRenderRequest(camera, new RenderPipeline.StandardRequest { destination = target });
            }
            RenderTexture previous = RenderTexture.active;
            RenderTexture.active = target;
            var pixels = new Texture2D(128, 128, TextureFormat.RGBA32, false);
            pixels.ReadPixels(new Rect(0, 0, 128, 128), 0, 0);
            pixels.Apply();
            RenderTexture.active = previous;
            Color left = pixels.GetPixel(38, 64);
            Color right = pixels.GetPixel(90, 64);
            File.WriteAllBytes("render-preview.png", pixels.EncodeToPNG());
            if (left.r <= left.g * 1.1f || left.g < 0.05f || right.b <= right.r * 1.1f || right.g < 0.05f)
                throw new Exception("Material readback failed. Left=" + left + "; right=" + right);
            Renderer cubeRenderer = GameObject.Find("Cube (pipeline Unlit)").GetComponent<Renderer>();
            Color cubeColor = cubeRenderer.sharedMaterial.color;
            cubeRenderer.sharedMaterial.color = new Color(cubeColor.r, cubeColor.g, cubeColor.b, 0f);
            if (pipeline == null) camera.Render();
            else
            {
                RenderPipeline.SubmitRenderRequest(camera, new RenderPipeline.StandardRequest { destination = target });
                RenderPipeline.SubmitRenderRequest(camera, new RenderPipeline.StandardRequest { destination = target });
            }
            RenderTexture.active = target;
            pixels.ReadPixels(new Rect(0, 0, 128, 128), 0, 0);
            pixels.Apply();
            RenderTexture.active = previous;
            Color hidden = pixels.GetPixel(38, 64);
            Color background = pixels.GetPixel(5, 64);
            cubeRenderer.sharedMaterial.color = cubeColor;
            float difference = Mathf.Abs(hidden.r - background.r) + Mathf.Abs(hidden.g - background.g) + Mathf.Abs(hidden.b - background.b);
            if (difference > 0.06f) throw new Exception("Alpha-zero material did not reveal the background: " + hidden + " vs " + background);
            cubeRenderer.sharedMaterial = materials[0];
            GameObject.Find("Cylinder (pipeline Unlit)").GetComponent<Renderer>().sharedMaterial = materials[1];
            GameObject.Find("Directional Light").GetComponent<Light>().intensity = 20f;
            if (pipeline == null) camera.Render();
            else
            {
                RenderPipeline.SubmitRenderRequest(camera, new RenderPipeline.StandardRequest { destination = target });
                RenderPipeline.SubmitRenderRequest(camera, new RenderPipeline.StandardRequest { destination = target });
            }
            RenderTexture.active = target;
            pixels.ReadPixels(new Rect(0, 0, 128, 128), 0, 0);
            pixels.Apply();
            RenderTexture.active = previous;
            Color litLeft = pixels.GetPixel(38, 64);
            Color litRight = pixels.GetPixel(90, 64);
            if (litLeft.g < 0.02f || litRight.g < 0.02f ||
                (litLeft.r > 0.8f && litLeft.b > 0.8f && litLeft.g < 0.2f) ||
                (litRight.r > 0.8f && litRight.b > 0.8f && litRight.g < 0.2f))
                throw new Exception("Lit material readback failed: " + litLeft + "; " + litRight);
            camera.targetTexture = null;
            UnityEngine.Object.DestroyImmediate(pixels);
            target.Release();
            UnityEngine.Object.DestroyImmediate(target);
            bool analyticSample = false;
#if BO_VALIDATION_FULL
            if (AnalyticOptimizationExample.Evaluate(0.25f, 0.75f) != 0f || AnalyticOptimizationExample.Evaluate(0f, 0f) != 0.625f)
                throw new Exception("Analytic sample objective is incorrect.");
            var analyticScene = BoAnalyticDemoSetup.CreateScene("Assets/Validation/BO-analytic-demo.unity");
            foreach (GameObject root in analyticScene.GetRootGameObjects())
            {
                if (GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(root) != 0)
                    throw new Exception("Analytic sample contains missing scripts.");
            }
            foreach (string dependency in AssetDatabase.GetDependencies(analyticScene.path))
                if (dependency.Contains("QuestionnaireToolkit")) throw new Exception("Analytic sample depends on QT.");
            analyticSample = true;
#endif
            File.WriteAllText("render-validation.json", JsonUtility.ToJson(new Report {
                unity = Application.unityVersion, pipeline = pipelineName, device = SystemInfo.graphicsDeviceName,
                left = left, right = right, materialCount = materials.Length, passed = true, analyticSample = analyticSample, alphaValidated = true,
                litLeft = litLeft, litRight = litRight, litValidated = true
            }, true));
            Debug.Log("BO_RENDER_VALIDATION_PASS " + pipelineName + " left=" + left + " right=" + right);
            EditorApplication.Exit(0);
        }
        catch (Exception error)
        {
            Debug.LogException(error);
            EditorApplication.Exit(1);
        }
    }

    [Serializable]
    private sealed class Report
    {
        public string unity, pipeline, device;
        public Color left, right;
        public Color litLeft, litRight;
        public int materialCount;
        public bool passed, analyticSample, alphaValidated, litValidated;
    }
}
