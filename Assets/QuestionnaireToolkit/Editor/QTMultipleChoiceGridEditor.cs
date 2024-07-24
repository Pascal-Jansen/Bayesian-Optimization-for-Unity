using QuestionnaireToolkit.Scripts;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace QuestionnaireToolkit.Editor
{
    [CanEditMultipleObjects]
    [CustomEditor(typeof(QTMultipleChoiceGrid))]
    public class QTMultipleChoiceGridEditor : UnityEditor.Editor
    {
   
        private SerializedProperty answerRequired;
        private SerializedProperty headerName;
        private SerializedProperty question;
        private SerializedProperty rows;
        private ReorderableList rowList;
        private SerializedProperty rowTexts;
        private SerializedProperty columns;
        private SerializedProperty columnTexts;

        private Texture image;
        private Texture logo;
        
        void OnEnable()
        {
            answerRequired = serializedObject.FindProperty("answerRequired");
            headerName = serializedObject.FindProperty("headerName");
            question = serializedObject.FindProperty("question");
            rows = serializedObject.FindProperty("rows");
            //test = new ReorderableList(serializedObject.FindProperty("rowTexts"));
            //test.elementNameProperty = "Row Texts";
            rowTexts = serializedObject.FindProperty("rowTexts");
            columns = serializedObject.FindProperty("columns");
            columnTexts = serializedObject.FindProperty("columnTexts");
            
            image = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/Banner/MCGridBanner.png");
            logo = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/QT_Logo_Mini2_Right.png");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            GUILayout.Space(5);
            var rect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.Height(30));
            EditorGUI.DrawRect(new Rect(rect.x, rect.y + 3f, rect.width, rect.height), new Color(0.0f, 0.125f, 0.376f, 1));
            EditorGUI.DrawRect(new Rect(rect.x + 2, rect.y + 5, rect.width - 4f, rect.height - 4), Color.white);
            GUI.DrawTexture(new Rect(rect.x - 24, rect.y, 250, rect.height + 6f), image, ScaleMode.ScaleToFit);
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
        
            //draw the list using GUILayout, you can of course specify your own position and label
            //rowTexts.DoLayoutList();
        
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(rowTexts, true);
            EditorGUILayout.PropertyField(columnTexts, true);

            if(EditorGUI.EndChangeCheck())
                serializedObject.ApplyModifiedProperties();

            serializedObject.ApplyModifiedProperties();
        
            //var checkboxesGrid = (CheckboxesGrid) target;
            //if (GUILayout.Button("(Re-)Build Grid")) { checkboxesGrid.BuildGrid(); }
        
        }
    }
}
