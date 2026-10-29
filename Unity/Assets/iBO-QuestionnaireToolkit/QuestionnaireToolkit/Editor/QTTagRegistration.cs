using QuestionnaireToolkit.Scripts;
using UnityEditor;

namespace QuestionnaireToolkit.Editor
{
    /// <summary>
    /// Adds the toolkit's tags to the project as soon as the editor loads the toolkit (after importing it or opening
    /// the project). Questionnaire objects compare tags while a scene opens (OnValidate), and every comparison with a
    /// tag the project does not define yet logs an error; QTManager and the inspectors only add them later.
    /// </summary>
    [InitializeOnLoad]
    internal static class QTTagRegistration
    {
        static QTTagRegistration()
        {
            if (!TryRegister())
                EditorApplication.delayCall += () => TryRegister();
        }

        private static bool TryRegister()
        {
            try
            {
                TagsAndLayers.RefreshQtTags();
                return true;
            }
            catch (System.Exception)
            {
                return false; // TagManager not loadable yet (e.g. during an import): retried once the editor is idle
            }
        }
    }
}
