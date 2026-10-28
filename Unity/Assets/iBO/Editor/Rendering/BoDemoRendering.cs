using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;

namespace BayesianOptimization.Editor.Rendering
{
    public enum BoDemoPipeline
    {
        BuiltIn,
        Universal,
        HighDefinition
    }

    /// <summary>Creates example materials for the user's existing render pipeline.</summary>
    public static class BoDemoRendering
    {
        public const string GeneratedRoot = "Assets/iBOGenerated/Rendering";

        public static BoDemoPipeline ActivePipeline
        {
            get
            {
                RenderPipelineAsset asset = GraphicsSettings.currentRenderPipeline;
                if (asset == null) return BoDemoPipeline.BuiltIn;
                for (Type type = asset.GetType(); type != null; type = type.BaseType)
                {
                    if (type.FullName == "UnityEngine.Rendering.Universal.UniversalRenderPipelineAsset")
                        return BoDemoPipeline.Universal;
                    if (type.FullName == "UnityEngine.Rendering.HighDefinition.HDRenderPipelineAsset")
                        return BoDemoPipeline.HighDefinition;
                }
                throw new NotSupportedException("The BO examples support Built-in, URP, and HDRP. Active asset: " + asset.GetType().FullName);
            }
        }

        public static string ShaderName(BoDemoPipeline pipeline, bool unlit)
        {
            switch (pipeline)
            {
                case BoDemoPipeline.BuiltIn: return unlit ? "Sprites/Default" : "Standard";
                case BoDemoPipeline.Universal: return unlit ? "Universal Render Pipeline/Unlit" : "Universal Render Pipeline/Lit";
                case BoDemoPipeline.HighDefinition: return unlit ? "HDRP/Unlit" : "HDRP/Lit";
                default: throw new ArgumentOutOfRangeException(nameof(pipeline));
            }
        }

        public static Material CreateMaterial(BoDemoPipeline pipeline, Color color, bool transparent = true, bool unlit = false)
        {
            string shaderName = ShaderName(pipeline, unlit);
            if (pipeline == BoDemoPipeline.BuiltIn && unlit && !transparent) shaderName = "Unlit/Color";
            Shader shader = Shader.Find(shaderName);
            if (shader == null || !shader.isSupported)
                throw new InvalidOperationException("Shader is unavailable on this platform: " + shaderName + ". Configure the selected pipeline with Unity's rendering tools first.");
            var material = new Material(shader) { name = "BO " + pipeline + (unlit ? " Unlit" : " Lit") };
            try
            {
                material.color = color;
                SetFloat(material, "_Metallic", 0f);
                SetFloat(material, "_Glossiness", 0.5f);
                SetFloat(material, "_Smoothness", 0.5f);
                SetFloat(material, "_AlphaClip", 0f);
                SetFloat(material, "_AlphaCutoffEnable", 0f);
                SetFloat(material, "_Mode", transparent ? 2f : 0f);
                SetFloat(material, "_Surface", transparent ? 1f : 0f);
                SetFloat(material, "_SurfaceType", transparent ? 1f : 0f);
                SetFloat(material, "_Blend", 0f);
                SetFloat(material, "_BlendMode", 0f);
                // Use the pipeline's public ShaderGUI validation to set its blend state,
                // keywords, render queue, and passes. No optional SRP assembly is required.
                var editor = UnityEditor.Editor.CreateEditor(material) as MaterialEditor;
                try
                {
                    if (editor != null && editor.customShaderGUI != null)
                        editor.customShaderGUI.ValidateMaterial(material);
                    else if (!unlit || pipeline != BoDemoPipeline.BuiltIn)
                        throw new InvalidOperationException("The shader's material validator is unavailable: " + shaderName);
                }
                finally
                {
                    if (editor != null) Object.DestroyImmediate(editor);
                }
                return material;
            }
            catch
            {
                Object.DestroyImmediate(material);
                throw;
            }
        }

        private static void SetFloat(Material material, string property, float value)
        {
            if (material.HasProperty(property)) material.SetFloat(property, value);
        }

        private static string EnsureFolder(string path)
        {
            path = (path ?? "").Replace('\\', '/').TrimEnd('/');
            if (!path.StartsWith("Assets/", StringComparison.Ordinal) || path.Contains(".."))
                throw new ArgumentException("Generated assets require a folder below Assets.", nameof(path));
            string[] parts = path.Split('/');
            string current = "Assets";
            for (int i = 1; i < parts.Length; ++i)
            {
                if (parts[i].Length == 0) throw new ArgumentException("Empty generated folder segment.", nameof(path));
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next)) AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
            return path;
        }

        private static Material SaveMaterial(string folder, string name, BoDemoPipeline pipeline, Color color, bool unlit)
        {
            string path = EnsureFolder(folder) + "/" + name + ".mat";
            Material material = CreateMaterial(pipeline, color, true, unlit);
            material.name = name;
            Material existing = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (existing != null)
            {
                EditorUtility.CopySerialized(material, existing);
                Object.DestroyImmediate(material);
                EditorUtility.SetDirty(existing);
                return existing;
            }
            AssetDatabase.CreateAsset(material, path);
            return material;
        }

        public static Material[] GenerateMaterials(string folder = null)
        {
            BoDemoPipeline pipeline = ActivePipeline;
            folder = folder ?? GeneratedRoot + "/" + pipeline;
            Color cube = new Color(1f, 0.5235849f, 0.5235849f, 1f);
            Color cylinder = new Color(0.5801887f, 0.85487413f, 1f, 1f);
            Material[] result = {
                SaveMaterial(folder, "CubeMat", pipeline, cube, false),
                SaveMaterial(folder, "CylinderMat", pipeline, cylinder, false),
                SaveMaterial(folder, "CubeUnlit", pipeline, cube, true),
                SaveMaterial(folder, "CylinderUnlit", pipeline, cylinder, true)
            };
            foreach (Material material in result) AssetDatabase.SaveAssetIfDirty(material);
            return result;
        }

        /// <summary>Replaces owned BO example materials in an open scene.</summary>
        public static int PrepareScene(Scene scene, string generatedFolder = null)
        {
            if (!scene.IsValid() || !scene.isLoaded) throw new ArgumentException("An open scene is required.", nameof(scene));
            Material[] variants = GenerateMaterials(generatedFolder);
            Material originalCube = AssetDatabase.LoadAssetAtPath<Material>("Assets/iBO/Examples/CubeMat.mat");
            Material originalCylinder = AssetDatabase.LoadAssetAtPath<Material>("Assets/iBO/Examples/CylinderMat.mat");
            int changed = 0;
            foreach (GameObject root in scene.GetRootGameObjects())
            foreach (Renderer renderer in root.GetComponentsInChildren<Renderer>(true))
            {
                Material[] materials = renderer.sharedMaterials;
                bool modified = false;
                for (int i = 0; i < materials.Length; ++i)
                {
                    Material source = materials[i];
                    Material replacement = null;
                    if (originalCube != null && source == originalCube) replacement = variants[0];
                    else if (originalCylinder != null && source == originalCylinder) replacement = variants[1];
                    else if (source != null)
                    {
                        string sourcePath = AssetDatabase.GetAssetPath(source);
                        bool owned = sourcePath.StartsWith("Assets/iBO/Samples/Rendering/", StringComparison.Ordinal) ||
                                     sourcePath.StartsWith(GeneratedRoot + "/", StringComparison.Ordinal);
                        if (owned)
                        {
                            switch (source.name)
                            {
                                case "CubeMat": replacement = variants[0]; break;
                                case "CylinderMat": replacement = variants[1]; break;
                                case "CubeUnlit": replacement = variants[2]; break;
                                case "CylinderUnlit": replacement = variants[3]; break;
                            }
                        }
                    }
                    if (replacement != null && source != replacement) { materials[i] = replacement; modified = true; }
                }
                if (!modified) continue;
                Undo.RecordObject(renderer, "Prepare BO example materials");
                renderer.sharedMaterials = materials;
                EditorUtility.SetDirty(renderer);
                ++changed;
            }
            if (changed > 0) EditorSceneManager.MarkSceneDirty(scene);
            return changed;
        }

        public static Scene CreatePreviewScene(string path)
        {
            string directory = Path.GetDirectoryName(path)?.Replace('\\', '/');
            EnsureFolder(directory);
            Material[] materials = GenerateMaterials(directory + "/Materials");
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var cameraObject = new GameObject("Main Camera", typeof(Camera));
            Camera camera = cameraObject.GetComponent<Camera>();
            cameraObject.tag = "MainCamera";
            cameraObject.transform.position = new Vector3(0f, 0f, -6f);
            camera.orthographic = true;
            camera.orthographicSize = 2f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.08f, 0.08f, 0.08f);
            var lightObject = new GameObject("Directional Light", typeof(Light));
            Light light = lightObject.GetComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1f;
            lightObject.transform.rotation = Quaternion.Euler(45f, -30f, 0f);
            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = "Cube (pipeline Unlit)";
            cube.transform.position = new Vector3(-0.8f, 0f, 0f);
            cube.GetComponent<Renderer>().sharedMaterial = materials[2];
            GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            cylinder.name = "Cylinder (pipeline Unlit)";
            cylinder.transform.position = new Vector3(0.8f, 0f, 0f);
            cylinder.transform.localScale = new Vector3(0.7f, 0.7f, 0.7f);
            cylinder.GetComponent<Renderer>().sharedMaterial = materials[3];
            if (!EditorSceneManager.SaveScene(scene, path)) throw new IOException("Could not save rendering preview: " + path);
            return scene;
        }

        [MenuItem("Tools/iBO/Rendering/Generate Materials for Active Pipeline")]
        private static void GenerateMenu()
        {
            Material[] materials = GenerateMaterials();
            Selection.activeObject = materials[0];
            EditorGUIUtility.PingObject(materials[0]);
            Debug.Log("BO materials generated for " + ActivePipeline + ". Project render pipeline settings were preserved.");
        }

        [MenuItem("Tools/iBO/Rendering/Prepare Open BO Demo Scene")]
        private static void PrepareMenu()
        {
            int changed = PrepareScene(SceneManager.GetActiveScene());
            Debug.Log("BO example renderers updated: " + changed + ". Save the scene to retain these material references.");
        }

        [MenuItem("Tools/iBO/Rendering/Create Material Preview Scene")]
        private static void PreviewMenu()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            CreatePreviewScene(GeneratedRoot + "/" + ActivePipeline + "/BO-rendering-preview.unity");
        }
    }
}
