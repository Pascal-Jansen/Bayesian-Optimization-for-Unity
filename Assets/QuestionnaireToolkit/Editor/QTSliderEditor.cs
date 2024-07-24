using QuestionnaireToolkit.Scripts;
using UnityEditor;
using UnityEngine;

namespace QuestionnaireToolkit.Editor
{
    [CanEditMultipleObjects]
    [CustomEditor(typeof(QTSlider))]
    public class QTSliderEditor : UnityEditor.Editor
    {

        private SerializedProperty answerRequired;
        private SerializedProperty headerName;
        private SerializedProperty question;
        
        private SerializedProperty minValue;
        private SerializedProperty maxValue;
        private SerializedProperty wholeNumbers;
        
        private SerializedProperty showPanels;
        private SerializedProperty showIntermediatePanels;
        private SerializedProperty automaticLabelNames;
        private SerializedProperty labelZero;
        private SerializedProperty labelQuarter;
        private SerializedProperty labelHalf;
        private SerializedProperty labelThreeQuarters;
        private SerializedProperty labelFull;

        private Texture image;
        private Texture logo;
        
        void OnEnable()
        {
            answerRequired = serializedObject.FindProperty("answerRequired");
            headerName = serializedObject.FindProperty("headerName");
            question = serializedObject.FindProperty("question");
            
            minValue = serializedObject.FindProperty("minValue");
            maxValue = serializedObject.FindProperty("maxValue");
            wholeNumbers = serializedObject.FindProperty("wholeNumbers");
            
            showPanels = serializedObject.FindProperty("showPanels");
            showIntermediatePanels = serializedObject.FindProperty("showIntermediatePanels");
            automaticLabelNames = serializedObject.FindProperty("automaticLabelNames");
            labelZero = serializedObject.FindProperty("labelZero");
            labelQuarter = serializedObject.FindProperty("labelQuarter");
            labelHalf = serializedObject.FindProperty("labelHalf");
            labelThreeQuarters = serializedObject.FindProperty("labelThreeQuarters");
            labelFull = serializedObject.FindProperty("labelFull");
            
            image = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/Banner/SliderBanner.png");
            logo = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/QT_Logo_Mini2_Right.png");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            GUILayout.Space(5);
            var rect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.Height(30));
            EditorGUI.DrawRect(new Rect(rect.x, rect.y + 3f, rect.width, rect.height), new Color(0.0f, 0.125f, 0.376f, 1));
            EditorGUI.DrawRect(new Rect(rect.x + 2, rect.y + 5, rect.width - 4f, rect.height - 4), Color.white);
            GUI.DrawTexture(new Rect(rect.x - 12, rect.y, 250, rect.height + 6f), image, ScaleMode.ScaleToFit);
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
        
            GUILayout.Space(5);
            GUILayout.Label("Slider Settings", EditorStyles.boldLabel);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Min Value",  GUILayout.Width(EditorGUIUtility.labelWidth));
            minValue.intValue = EditorGUILayout.IntField(minValue.intValue);
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Max Value",  GUILayout.Width(EditorGUIUtility.labelWidth));
            maxValue.intValue = EditorGUILayout.IntField(maxValue.intValue);
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Whole Numbers",  GUILayout.Width(EditorGUIUtility.labelWidth));
            wholeNumbers.boolValue = EditorGUILayout.Toggle( wholeNumbers.boolValue); 
            GUILayout.EndHorizontal();
            
            GUILayout.Space(5);
            GUILayout.Label("Slider Visuals", EditorStyles.boldLabel);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Show Panels",  GUILayout.Width(EditorGUIUtility.labelWidth));
            showPanels.boolValue = EditorGUILayout.Toggle( showPanels.boolValue); 
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Show Intermediate Panels", GUILayout.Width(EditorGUIUtility.labelWidth));
            showIntermediatePanels.boolValue = EditorGUILayout.Toggle(showIntermediatePanels.boolValue);
            GUILayout.EndHorizontal();

            GUILayout.Space(5);
            GUILayout.Label("Slider Labels", EditorStyles.boldLabel);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Automatic Label Names", GUILayout.Width(EditorGUIUtility.labelWidth));
            automaticLabelNames.boolValue = EditorGUILayout.Toggle(automaticLabelNames.boolValue);
            GUILayout.EndHorizontal();
            if (!automaticLabelNames.boolValue)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label("Label Zero",  GUILayout.Width(EditorGUIUtility.labelWidth));
                labelZero.stringValue = EditorGUILayout.TextArea( labelZero.stringValue );
                GUILayout.EndHorizontal();
                GUILayout.BeginHorizontal();
                GUILayout.Label("Label Quarter",  GUILayout.Width(EditorGUIUtility.labelWidth));
                labelQuarter.stringValue = EditorGUILayout.TextArea( labelQuarter.stringValue );
                GUILayout.EndHorizontal();
                GUILayout.BeginHorizontal();
                GUILayout.Label("Label Half",  GUILayout.Width(EditorGUIUtility.labelWidth));
                labelHalf.stringValue = EditorGUILayout.TextArea( labelHalf.stringValue );
                GUILayout.EndHorizontal();
                GUILayout.BeginHorizontal();
                GUILayout.Label("Label Three Quarters",  GUILayout.Width(EditorGUIUtility.labelWidth));
                labelThreeQuarters.stringValue = EditorGUILayout.TextArea( labelThreeQuarters.stringValue );
                GUILayout.EndHorizontal();
                GUILayout.BeginHorizontal();
                GUILayout.Label("Label Full",  GUILayout.Width(EditorGUIUtility.labelWidth));
                labelFull.stringValue = EditorGUILayout.TextArea( labelFull.stringValue );
                GUILayout.EndHorizontal();
            }

            serializedObject.ApplyModifiedProperties();

        }
    }
}

