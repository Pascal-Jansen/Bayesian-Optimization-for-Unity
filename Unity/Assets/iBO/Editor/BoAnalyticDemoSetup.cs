using System.IO;
using BayesianOptimization.Examples;
using BayesianOptimization.Scripts;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BayesianOptimization.Editor
{
    public static class BoAnalyticDemoSetup
    {
        [MenuItem("Tools/iBO/Create Analytic Demo")]
        public static void CreateDefaultScene()
        {
            if (Application.isBatchMode || EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                CreateScene("Assets/iBO/Samples/Analytic/BO-analytic-demo.unity");
        }

        public static Scene CreateScene(string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var study = new GameObject("Analytic Optimization", typeof(BoForUnityManager), typeof(PythonStarter),
                typeof(SocketNetwork), typeof(Optimizer), typeof(MainThreadDispatcher), typeof(AnalyticOptimizationExample));
            SceneManager.MoveGameObjectToScene(study, scene);
            var camera = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener));
            camera.tag = "MainCamera";
            camera.GetComponent<Camera>().clearFlags = CameraClearFlags.SolidColor;
            camera.GetComponent<Camera>().backgroundColor = new Color(0.08f, 0.1f, 0.13f);
            SceneManager.MoveGameObjectToScene(camera, scene);
            EditorSceneManager.SaveScene(scene, path);
            return scene;
        }
    }
}
