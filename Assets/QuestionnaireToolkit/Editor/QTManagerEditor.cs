using System;
using System.Collections;
using System.Collections.Generic;
using QuestionnaireToolkit.Editor;
using QuestionnaireToolkit.Scripts;
using UnityEditor;
using UnityEngine;

[CanEditMultipleObjects]
[CustomEditor(typeof(QTManager))]
public class QTManagerEditor : Editor
{
    private SerializedProperty pageHeight;
    private SerializedProperty dynamicHeight;
    private SerializedProperty useCustomTransform; 
    private SerializedProperty distanceToCamera;
    private SerializedProperty pageScaleFactor;
    
    private SerializedProperty pageBackgroundColor;
    private SerializedProperty pageBottomColor;
    private SerializedProperty highlightColor;
    private SerializedProperty sliderValue;
    
    private SerializedProperty showTopPanel;
    private SerializedProperty showBottomPanel;
    private SerializedProperty showPageNumber;
    private SerializedProperty showPrevButton;

    private ReorderableList questionnaires;

    private QTManager manager;
    private SerializedProperty displayMode;
    private SerializedProperty deviceType;
    private SerializedProperty orientation;
    private SerializedProperty colorScheme;

    private GUIStyle noteStyle;
    private Texture image;
    private Texture logo;
    
    void OnEnable()
    {
            noteStyle = new GUIStyle();
            noteStyle.wordWrap = true;
            noteStyle.normal.textColor = Color.white;
            noteStyle.fontStyle = FontStyle.Bold;

            displayMode = serializedObject.FindProperty("displayMode");
            orientation = serializedObject.FindProperty("orientation");
            colorScheme = serializedObject.FindProperty("colorScheme");
            deviceType = serializedObject.FindProperty("deviceType");

            pageHeight = serializedObject.FindProperty("pageHeight");
            dynamicHeight = serializedObject.FindProperty("dynamicHeight");
            useCustomTransform = serializedObject.FindProperty("useCustomTransform");
            distanceToCamera = serializedObject.FindProperty("distanceToCamera");
            pageScaleFactor = serializedObject.FindProperty("pageScaleFactor");
            
            pageBackgroundColor = serializedObject.FindProperty("pageBackgroundColor");
            pageBottomColor = serializedObject.FindProperty("pageBottomColor");
            highlightColor = serializedObject.FindProperty("highlightColor");
            sliderValue = serializedObject.FindProperty("sliderValue");
            
            showTopPanel = serializedObject.FindProperty("showTopPanel");
            showBottomPanel = serializedObject.FindProperty("showBottomPanel");
            showPageNumber = serializedObject.FindProperty("showPageNumber");
            showPrevButton = serializedObject.FindProperty("showPrevButton");

            questionnaires = new ReorderableList(serializedObject.FindProperty("questionnaires"), false, false, true);
            questionnaires.elementNameProperty = "Questionnaires";

            manager = (QTManager) target;
            questionnaires.onChangedCallback += (list) =>
            {
                manager.ReorderQuestionnaires(list.Length, list.Index);
            };
            questionnaires.onSelectCallback += (list) =>
            {
                manager.selectedQuestionnaire = list.Index;
                manager.ShowSelectedQuestionnaire(list.Index);
            };
            questionnaires.onRemoveCallback += (list) =>
            {
                var i = list.Index;
                list.RemoveItem(list.Index);
                manager.DeleteQuestionnaire(list.Length, i);
            };

            image = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/Banner/ManagerBanner.png");
            logo = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/QT_Logo_Mini2_Right.png");
        }
    
    public override void OnInspectorGUI()
        {
            TagsAndLayers.RefreshQtTags(); // always keep tag list fresh!
            
            serializedObject.Update();
            
            GUILayout.Space(5);
            var rect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.Height(30));
            EditorGUI.DrawRect(new Rect(rect.x, rect.y + 3f, rect.width, rect.height), new Color(0.0f, 0.125f, 0.376f, 1));
            EditorGUI.DrawRect(new Rect(rect.x + 2, rect.y + 5, rect.width - 4f, rect.height - 4), Color.white);
            GUI.DrawTexture(new Rect(rect.x - 14, rect.y, 250, rect.height + 6f), image, ScaleMode.ScaleToFit);
            GUI.DrawTexture(new Rect(rect.x + rect.width - 50, rect.y + 4, 60, rect.height - 3), logo, ScaleMode.ScaleToFit);
            GUILayout.Space(5);

            GUILayout.Label("General Display Settings", EditorStyles.boldLabel);
            //manager.displayMode = (QuestionnaireManager.DisplayMode)EditorGUILayout.EnumPopup("Display Mode", manager.displayMode);
            EditorGUILayout.PropertyField(displayMode);
            if (manager.displayMode == QTManager.DisplayMode.VR)
            {
                EditorGUILayout.PropertyField(deviceType);
            }
            //manager.orientation = (QuestionnaireManager.Orientation)EditorGUILayout.EnumPopup("Orientation", manager.orientation);
            EditorGUILayout.PropertyField(orientation);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Dynamic Page Height",  GUILayout.Width(EditorGUIUtility.labelWidth));
            dynamicHeight.boolValue = EditorGUILayout.Toggle( dynamicHeight.boolValue);
            GUILayout.EndHorizontal();
            if (!dynamicHeight.boolValue)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label("Page Height",  GUILayout.Width(EditorGUIUtility.labelWidth));
                pageHeight.floatValue = EditorGUILayout.FloatField( pageHeight.floatValue);
                GUILayout.EndHorizontal();
            }
            if (manager.displayMode == QTManager.DisplayMode.VR)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label("Use Custom Transform",  GUILayout.Width(EditorGUIUtility.labelWidth));
                useCustomTransform.boolValue = EditorGUILayout.Toggle( useCustomTransform.boolValue);
                GUILayout.EndHorizontal();
                if (!useCustomTransform.boolValue)
                {
                    GUILayout.BeginHorizontal();
                    GUILayout.Label("Distance To Camera",  GUILayout.Width(EditorGUIUtility.labelWidth));
                    distanceToCamera.floatValue = EditorGUILayout.FloatField( distanceToCamera.floatValue );
                    GUILayout.EndHorizontal();
                    GUILayout.BeginHorizontal();
                    GUILayout.Label("Page Scale Factor",  GUILayout.Width(EditorGUIUtility.labelWidth));
                    pageScaleFactor.floatValue = EditorGUILayout.FloatField( pageScaleFactor.floatValue );
                    GUILayout.EndHorizontal();
                }
            }
            //manager.colorScheme = (QuestionnaireManager.ColorScheme)EditorGUILayout.EnumPopup("Color Scheme", manager.colorScheme);
            EditorGUILayout.PropertyField(colorScheme);
            if (manager.colorScheme == QTManager.ColorScheme.Custom)
            {
                pageBackgroundColor.colorValue = EditorGUILayout.ColorField("Page Background Color", pageBackgroundColor.colorValue);
                pageBottomColor.colorValue = EditorGUILayout.ColorField("Page Bottom Color", pageBottomColor.colorValue);
                highlightColor.colorValue = EditorGUILayout.ColorField("Highlight Color", highlightColor.colorValue);
            }
            GUILayout.BeginHorizontal();
            GUILayout.Label("Background Transparency",  GUILayout.Width(EditorGUIUtility.labelWidth));
            sliderValue.floatValue = GUILayout.HorizontalSlider(sliderValue.floatValue, 0, 100);
            sliderValue.floatValue = EditorGUILayout.FloatField( Mathf.Round(sliderValue.floatValue), GUILayout.Width(EditorGUIUtility.fieldWidth));
            GUILayout.EndHorizontal();
            /*
            GUILayout.BeginHorizontal();
            GUILayout.Label("Show Top Panel",  GUILayout.Width(EditorGUIUtility.labelWidth));
            showTopPanel.boolValue = EditorGUILayout.Toggle( showTopPanel.boolValue);
            GUILayout.EndHorizontal();
            */
            GUILayout.BeginHorizontal();
            GUILayout.Label("Show Bottom Panel",  GUILayout.Width(EditorGUIUtility.labelWidth));
            showBottomPanel.boolValue = EditorGUILayout.Toggle( showBottomPanel.boolValue);
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Show Page Number",  GUILayout.Width(EditorGUIUtility.labelWidth));
            showPageNumber.boolValue = EditorGUILayout.Toggle( showPageNumber.boolValue);
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Show Prev Button",  GUILayout.Width(EditorGUIUtility.labelWidth));
            showPrevButton.boolValue = EditorGUILayout.Toggle( showPrevButton.boolValue);
            GUILayout.EndHorizontal();
            
            
            GUILayout.Label("Questionnaire Management", EditorStyles.boldLabel);
            //draw the list using GUILayout, you can of course specify your own position and label
            questionnaires.DoLayoutList();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Create Questionnaire")) { manager.CreateQuestionnaire(); }
            if (GUILayout.Button("Copy")) { manager.CopyQuestionnaire(); }
            if (GUILayout.Button(manager.selectedQuestionnaire > -1 && manager.selectedQuestionnaire < manager.questionnaires.Count ? 
                    "Delete (Element " + manager.selectedQuestionnaire + ")" : "Delete (Nothing selected)"))
            {
                manager.DeleteQuestionnaire();
            }
            GUILayout.EndHorizontal();
            EditorGUILayout.Space();

            serializedObject.ApplyModifiedProperties();
        }
}
