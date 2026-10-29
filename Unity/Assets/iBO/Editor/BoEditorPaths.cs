using UnityEditor;

namespace BayesianOptimization.Editor
{
    /// <summary>Asset paths inside the iBO package, wherever the project keeps the package folder.</summary>
    public static class BoEditorPaths
    {
        // GUID of the package folder (Assets/iBO in this repository).
        private const string PackageFolderGuid = "d722b1fe6223d4c8fb159c3812b998c3";

        public static string PackageRoot
        {
            get
            {
                string path = AssetDatabase.GUIDToAssetPath(PackageFolderGuid);
                return string.IsNullOrEmpty(path) ? "Assets/iBO" : path;
            }
        }
    }
}
