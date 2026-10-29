using System.IO;
using UnityEditor;
using UnityEditor.Compilation;
using UnityEngine;

namespace QuestionnaireToolkit.Editor
{
    /// <summary>
    /// Locates the toolkit's own editor assets wherever the QuestionnaireToolkit folder lives in the project
    /// (directly below Assets or nested inside another package's folder), instead of assuming Assets/QuestionnaireToolkit.
    /// </summary>
    internal static class QTEditorPaths
    {
        private const string EditorAssemblyName = "QuestionnaireToolkit.Editor";
        private const string DefaultRoot = "Assets/QuestionnaireToolkit";

        private static string _root;

        /// <summary>Project-relative toolkit folder, e.g. "Assets/QuestionnaireToolkit".</summary>
        internal static string Root => _root ??= ResolveRoot();

        /// <summary>Loads a texture by its path relative to the toolkit folder, e.g. "Textures/Banner/SliderBanner.png".</summary>
        internal static Texture LoadTexture(string relativePath)
        {
            return AssetDatabase.LoadAssetAtPath<Texture>(Root + "/" + relativePath);
        }

        private static string ResolveRoot()
        {
            // <root>/Editor/QuestionnaireToolkit.Editor.asmdef
            string asmdefPath = CompilationPipeline.GetAssemblyDefinitionFilePathFromAssemblyName(EditorAssemblyName);
            if (string.IsNullOrEmpty(asmdefPath))
                return DefaultRoot;

            string editorDir = Path.GetDirectoryName(asmdefPath);
            string root = string.IsNullOrEmpty(editorDir) ? null : Path.GetDirectoryName(editorDir);
            return string.IsNullOrEmpty(root) ? DefaultRoot : root.Replace('\\', '/');
        }
    }
}
