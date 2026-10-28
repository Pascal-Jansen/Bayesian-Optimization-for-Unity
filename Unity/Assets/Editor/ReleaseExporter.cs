using System;
using System.IO;
using System.Linq;
using BayesianOptimization.Editor;
using UnityEditor;
using UnityEditor.AssetPackage;

namespace BayesianOptimization.Distribution
{
    public static class ReleaseExporter
    {
        public static void Build()
        {
            string output = Environment.GetEnvironmentVariable("BO_RELEASE_DIRECTORY");
            string version = Environment.GetEnvironmentVariable("BO_RELEASE_VERSION");
            if (string.IsNullOrWhiteSpace(output) || string.IsNullOrWhiteSpace(version))
                throw new InvalidOperationException("BO_RELEASE_DIRECTORY and BO_RELEASE_VERSION are required.");

            Directory.CreateDirectory(output);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            BoAnalyticDemoSetup.CreateScene("Assets/BOforUnity/Samples/Analytic/BO-analytic-demo.unity");
            AssetDatabase.SaveAssets();
            string package = Path.Combine(Path.GetFullPath(output), "BayesianOptimization-Unity-" + version + ".unitypackage");
            string organization = Environment.GetEnvironmentVariable("BO_UNITY_ORG_ID") ?? string.Empty;
            string[] assets = AssetDatabase.GetAllAssetPaths().Where(IsReleaseAsset).ToArray();
            Package.Export(new ExportPackageParameters(
                assets, package, organization, ExportPackageOptions.Default));
            UnityEngine.Debug.Log("Exported " + package);
        }

        private static bool IsReleaseAsset(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return false;
            if (path.StartsWith("Assets/BOforUnity/Tests/", StringComparison.Ordinal)) return false;
            return path.StartsWith("Assets/BOforUnity/", StringComparison.Ordinal) ||
                   path.StartsWith("Assets/StreamingAssets/BOData/BayesianOptimization/", StringComparison.Ordinal) ||
                   path.StartsWith("Assets/StreamingAssets/BOData/InitData/", StringComparison.Ordinal) ||
                   path.StartsWith("Assets/StreamingAssets/BOData/MetaSources/", StringComparison.Ordinal) ||
                   path == "Assets/StreamingAssets/BOData/Installation/requirements.txt";
        }
    }
}
