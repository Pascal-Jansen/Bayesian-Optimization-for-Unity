using QuestionnaireToolkit.Scripts;
using UnityEditor;
using UnityEngine;

namespace QuestionnaireToolkit.Editor
{
    [CanEditMultipleObjects]
    [CustomEditor(typeof(QTCheckboxes))]
    public class QTCheckboxesEditor : UnityEditor.Editor
    {

        private SerializedProperty answerRequired;
        private SerializedProperty headerName;
        private SerializedProperty question;
        private SerializedProperty includeOtherOption;
        private ReorderableList options;
        private SerializedProperty answerOption;
        private SerializedProperty answerValue;

        private QTCheckboxes checkboxes;
        private Texture image;
        private Texture logo;

        void OnEnable()
        {
            answerRequired = serializedObject.FindProperty("answerRequired");
            headerName = serializedObject.FindProperty("headerName");
            question = serializedObject.FindProperty("question");
            includeOtherOption = serializedObject.FindProperty("includeOtherOption");
            answerOption = serializedObject.FindProperty("answerOption");
            answerValue = serializedObject.FindProperty("answerValue");
            options = new ReorderableList(serializedObject.FindProperty("options"), false, true, true);
            options.elementNameProperty = "Options";
            
            checkboxes = (QTCheckboxes) target;
            options.onChangedCallback += (list) =>
            {
                checkboxes.ReorderItems(list.Length, list.Index);
            };
            options.onSelectCallback += (list) =>
            {
                checkboxes.selectedIndex = list.Index;
                checkboxes.OptionSelected(list.Index);
            };
            options.onRemoveCallback += (list) =>
            {
                var i = list.Index;
                list.RemoveItem(list.Index);
                checkboxes.DeleteItem(list.Length, i);
            };
            
            image = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/Banner/CheckboxesBanner.png");
            logo = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/QT_Logo_Mini2_Right.png");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            
            GUILayout.Space(5);
            var rect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.Height(30));
            EditorGUI.DrawRect(new Rect(rect.x, rect.y + 3f, rect.width, rect.height), new Color(0.0f, 0.125f, 0.376f, 1));
            EditorGUI.DrawRect(new Rect(rect.x + 2, rect.y + 5, rect.width - 4f, rect.height - 4), Color.white);
            GUI.DrawTexture(new Rect(rect.x - 15, rect.y, 250, rect.height + 6f), image, ScaleMode.ScaleToFit);
            GUI.DrawTexture(new Rect(rect.x + rect.width - 50, rect.y + 4, 60, rect.height - 3), logo, ScaleMode.ScaleToFit);
            GUILayout.Space(5);

            GUILayout.BeginHorizontal();
            GUILayout.Label("Answer Mandatory",  GUILayout.Width(EditorGUIUtility.labelWidth));
            answerRequired.boolValue = EditorGUILayout.Toggle( answerRequired.boolValue);
            GUILayout.EndHorizontal();
        
            GUILayout.BeginHorizontal();
            GUILayout.Label("Header Name",  GUILayout.Width(EditorGUIUtility.labelWidth));
            headerName.stringValue = EditorGUILayout.TextArea( headerName.stringValue );
            GUILayout.EndHorizontal();
        
            GUILayout.BeginHorizontal();
            GUILayout.Label("Question",  GUILayout.Width(EditorGUIUtility.labelWidth));
            question.stringValue = EditorGUILayout.TextArea( question.stringValue );
            GUILayout.EndHorizontal();
        
            GUILayout.BeginHorizontal();
            GUILayout.Label("Include Other Option",  GUILayout.Width(EditorGUIUtility.labelWidth));
            includeOtherOption.boolValue = EditorGUILayout.Toggle( includeOtherOption.boolValue);
            GUILayout.EndHorizontal();
            
            //draw the list using GUILayout, you can of course specify your own position and label
            options.DoLayoutList();

            GUILayout.BeginHorizontal();
            GUILayout.Label("Answer Option",  GUILayout.Width(EditorGUIUtility.labelWidth));
            answerOption.stringValue = EditorGUILayout.TextArea( answerOption.stringValue );
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Answer Value",  GUILayout.Width(EditorGUIUtility.labelWidth));
            answerValue.stringValue = EditorGUILayout.TextArea( answerValue.stringValue );
            GUILayout.EndHorizontal();
            
            if (GUILayout.Button("Add Option")) { checkboxes.AddOption(); }
            if (GUILayout.Button("Edit Selected Option")) { checkboxes.EditOption(); }
            
            serializedObject.ApplyModifiedProperties();

        }
    }
}
