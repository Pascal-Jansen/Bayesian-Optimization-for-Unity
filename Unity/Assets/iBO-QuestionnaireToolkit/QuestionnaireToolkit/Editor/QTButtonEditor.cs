using QuestionnaireToolkit.Scripts;
using UnityEditor;
using UnityEngine;

namespace QuestionnaireToolkit.Editor
{
    [CanEditMultipleObjects]
    [CustomEditor(typeof(QTButton))]
    public class QTButtonEditor : UnityEditor.Editor
    {



        void OnEnable()
        {
        
        }

        public override void OnInspectorGUI()
        {
            var button = (QTButton) target;
            if (GUILayout.Button("Add Button")) { button.AddOption(); }
        
        }
    }
}
