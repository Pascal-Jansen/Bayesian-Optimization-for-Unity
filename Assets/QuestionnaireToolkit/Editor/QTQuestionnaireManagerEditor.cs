using System;
using QuestionnaireToolkit.Scripts;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace QuestionnaireToolkit.Editor
{
    [CanEditMultipleObjects]
    [CustomEditor(typeof(QTQuestionnaireManager))]
    public class QTQuestionnaireManagerEditor : UnityEditor.Editor
    {
        private SerializedProperty qtMetaData;
        
        private SerializedProperty overrideManagerSettings;
        
        private SerializedProperty startWithScene;
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

        private SerializedProperty generateResultsFile;
        private SerializedProperty resultsSavePath;
        private SerializedProperty resultsFileName;
        private SerializedProperty newFileEachStart;

        private SerializedProperty importPath;
        private SerializedProperty exportPath;
        
        private SerializedProperty resultsHeaderItems;
        private SerializedProperty onQuestionnaireFinished;

        private SerializedProperty useCustomUserid;
        private SerializedProperty userId;
        private SerializedProperty runsPerUser;
        private SerializedProperty generateStartTimestamp;
        private SerializedProperty generateFinishTimestamp;
        private SerializedProperty overwriteResultsHeaderItems;
        
        private ReorderableList questionPages;
        private ReorderableList additionalCsvItems;

        private QTQuestionnaireManager manager;
        private SerializedProperty displayMode;
        private SerializedProperty deviceType;
        private SerializedProperty orientation;
        private SerializedProperty colorScheme;
        private SerializedProperty fileFormat;
        
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
            fileFormat = serializedObject.FindProperty("resultsFileFormat");
            
            qtMetaData = serializedObject.FindProperty("qtMetaData");

            overrideManagerSettings = serializedObject.FindProperty("overrideManagerSettings");
            
            startWithScene = serializedObject.FindProperty("startWithScene");
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
            
            generateResultsFile = serializedObject.FindProperty("generateResultsFile");
            resultsSavePath = serializedObject.FindProperty("resultsSavePath");
            resultsFileName = serializedObject.FindProperty("resultsFileName");
            newFileEachStart = serializedObject.FindProperty("newFileEachStart");
            
            importPath = serializedObject.FindProperty("importPath");
            exportPath = serializedObject.FindProperty("exportPath");
            
            resultsHeaderItems = serializedObject.FindProperty("resultsHeaderItems");
            onQuestionnaireFinished = serializedObject.FindProperty("onQuestionnaireFinished");

            useCustomUserid = serializedObject.FindProperty("customUserId");
            userId = serializedObject.FindProperty("userId");
            runsPerUser = serializedObject.FindProperty("runsPerUser");
            generateStartTimestamp = serializedObject.FindProperty("generateStartTimestamp");
            generateFinishTimestamp = serializedObject.FindProperty("generateFinishTimestamp");
            overwriteResultsHeaderItems = serializedObject.FindProperty("overwriteResultsHeaderItems");
            
            questionPages = new ReorderableList(serializedObject.FindProperty("questionPages"), false, false, true);
            questionPages.elementNameProperty = "Question Pages";

            additionalCsvItems = new ReorderableList(serializedObject.FindProperty("additionalCsvItems"), true, true, true);
            additionalCsvItems.elementNameProperty = "Additional CSV Items";

            manager = (QTQuestionnaireManager) target;
            questionPages.onChangedCallback += (list) =>
            {
                manager.ReorderPage(list.Length, list.Index);
            };
            questionPages.onSelectCallback += (list) =>
            {
                manager.selectedPage = list.Index;
                manager.ShowPage(list.Index);
            };
            questionPages.onRemoveCallback += (list) =>
            {
                var i = list.Index;
                list.RemoveItem(list.Index);
                manager.DeletePage(list.Length, i);
            };

            image = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/Banner/QuestionnaireBanner.png");
            logo = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/QT_Logo_Mini2_Right.png");
        }

        public override void OnInspectorGUI()
        {
            try
            {
                if (!manager._qtManager) // check for tags if there is no QTManager in this scene
                    TagsAndLayers.RefreshQtTags();

                if(manager._qtManager.selectedQuestionnaire != manager.transform.GetSiblingIndex()) 
                    manager._qtManager.ShowSelectedQuestionnaire(manager.transform.GetSiblingIndex());
            }
            catch (Exception) { }
            
            if (!EditorApplication.isPlayingOrWillChangePlaymode) 
                manager.BuildHeaderItems();

            EditorStyles.textField.wordWrap = true;
            serializedObject.Update();
            
            GUILayout.Space(5);
            var rect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.Height(30));
            EditorGUI.DrawRect(new Rect(rect.x, rect.y + 3f, rect.width, rect.height), new Color(0.0f, 0.125f, 0.376f, 1));
            EditorGUI.DrawRect(new Rect(rect.x + 2, rect.y + 5, rect.width - 4f, rect.height - 4), Color.white);
            GUI.DrawTexture(new Rect(rect.x - 14, rect.y, 250, rect.height + 6f), image, ScaleMode.ScaleToFit);
            GUI.DrawTexture(new Rect(rect.x + rect.width - 50, rect.y + 4, 60, rect.height - 3), logo, ScaleMode.ScaleToFit);
            GUILayout.Space(5);

            //EditorGUILayout.LabelField("QuestionnaireToolkit Editor", EditorStyles.centeredGreyMiniLabel);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Start With Scene",  GUILayout.Width(EditorGUIUtility.labelWidth));
            startWithScene.boolValue = EditorGUILayout.Toggle( startWithScene.boolValue);
            GUILayout.EndHorizontal();
            if (manager._qtManager != null)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label("Override Manager Display Settings",  GUILayout.Width(EditorGUIUtility.labelWidth));
                overrideManagerSettings.boolValue = EditorGUILayout.Toggle( overrideManagerSettings.boolValue);
                GUILayout.EndHorizontal();
            }

            if (manager._qtManager == null || overrideManagerSettings.boolValue)
            { 
                GUILayout.Label("Display Settings", EditorStyles.boldLabel);
                //manager.displayMode = (QuestionnaireManager.DisplayMode)EditorGUILayout.EnumPopup("Display Mode", manager.displayMode);
            EditorGUILayout.PropertyField(displayMode);
            if (manager.displayMode == QTQuestionnaireManager.DisplayMode.VR)
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
            if (manager.displayMode == QTQuestionnaireManager.DisplayMode.VR)
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
            if (manager.colorScheme == QTQuestionnaireManager.ColorScheme.Custom)
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
            }
            
            GUILayout.Label("Import/Export Questionnaire", EditorStyles.boldLabel);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Import from", GUILayout.Width(100))) { manager.ImportPages(); }
            //importPath.stringValue = EditorGUILayout.TextField( importPath.stringValue );
            GUILayout.Label(importPath.stringValue);
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Export to", GUILayout.Width(100))) { manager.ExportPages(); }
            //exportPath.stringValue = EditorGUILayout.TextField( exportPath.stringValue );
            GUILayout.Label(exportPath.stringValue);
            GUILayout.EndHorizontal();
            
            GUILayout.Label("Page Management", EditorStyles.boldLabel);
            //draw the list using GUILayout, you can of course specify your own position and label
            questionPages.DoLayoutList();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Create Page")) { manager.CreatePage(); }
            if (GUILayout.Button("Copy")) { manager.CopyPage(); }
            if (GUILayout.Button(manager.selectedPage > -1 && manager.selectedPage < manager.questionPages.Count ? 
                    "Delete (Element " + manager.selectedPage + ")" : "Delete (Nothing selected)"))
            {
                manager.DeletePage();
            }
            GUILayout.EndHorizontal();
            EditorGUILayout.Space();
            
            
            GUILayout.Label("Results File Settings", EditorStyles.boldLabel);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Generate Results File",  GUILayout.Width(EditorGUIUtility.labelWidth));
            generateResultsFile.boolValue = EditorGUILayout.Toggle( generateResultsFile.boolValue);
            GUILayout.EndHorizontal();
            if (generateResultsFile.boolValue)
            {
                GUILayout.Label("Results Save Path:",  GUILayout.Width(EditorGUIUtility.labelWidth));
                //GUILayout.Label(Application.persistentDataPath);
                GUILayout.BeginHorizontal();
                EditorGUILayout.TextArea( Application.persistentDataPath );
                resultsSavePath.stringValue = EditorGUILayout.TextArea( resultsSavePath.stringValue );
                GUILayout.EndHorizontal();
                GUILayout.BeginHorizontal();
                GUILayout.Label("Results File Name",  GUILayout.Width(EditorGUIUtility.labelWidth));
                resultsFileName.stringValue = EditorGUILayout.TextField( resultsFileName.stringValue );
                GUILayout.EndHorizontal();
                //manager.resultsFileFormat = (QuestionnaireManager.FileFormat)EditorGUILayout.EnumPopup("File Format", manager.resultsFileFormat);
                EditorGUILayout.PropertyField(fileFormat);
                GUILayout.BeginHorizontal();
                GUILayout.Label("New File Each Start",  GUILayout.Width(EditorGUIUtility.labelWidth));
                newFileEachStart.boolValue = EditorGUILayout.Toggle( newFileEachStart.boolValue);
                GUILayout.EndHorizontal();
                
                GUILayout.Label("Results Visualization", EditorStyles.boldLabel);
                GUILayout.BeginHorizontal();
                GUILayout.Label("Use Custom User Id",  GUILayout.Width(EditorGUIUtility.labelWidth));
                useCustomUserid.boolValue = EditorGUILayout.Toggle( useCustomUserid.boolValue);
                GUILayout.EndHorizontal();
                if (useCustomUserid.boolValue)
                {
                    EditorGUILayout.LabelField("A custom user id can NOT be set with the editor at runtime." +
                                               "\n If needed, set the public variable via code instead.", noteStyle);
                    GUILayout.BeginHorizontal();
                    GUILayout.Label("User Id",  GUILayout.Width(EditorGUIUtility.labelWidth));
                    userId.stringValue = EditorGUILayout.TextField( userId.stringValue );
                    GUILayout.EndHorizontal();
                }
                GUILayout.BeginHorizontal();
                GUILayout.Label("Runs Per User",  GUILayout.Width(EditorGUIUtility.labelWidth));
                runsPerUser.intValue = EditorGUILayout.IntField( runsPerUser.intValue );
                GUILayout.EndHorizontal();
                GUILayout.BeginHorizontal();
                GUILayout.Label("Generate Start Timestamp",  GUILayout.Width(EditorGUIUtility.labelWidth));
                generateStartTimestamp.boolValue = EditorGUILayout.Toggle( generateStartTimestamp.boolValue);
                GUILayout.EndHorizontal();
                GUILayout.BeginHorizontal();
                GUILayout.Label("Generate Finish Timestamp",  GUILayout.Width(EditorGUIUtility.labelWidth));
                generateFinishTimestamp.boolValue = EditorGUILayout.Toggle( generateFinishTimestamp.boolValue);
                GUILayout.EndHorizontal();

                additionalCsvItems.DoLayoutList();
            }
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Overwrite Results Header Items",  GUILayout.Width(EditorGUIUtility.labelWidth));
            overwriteResultsHeaderItems.boolValue = EditorGUILayout.Toggle( overwriteResultsHeaderItems.boolValue);
            GUILayout.EndHorizontal();
            if (overwriteResultsHeaderItems.boolValue)
            {
                EditorGUILayout.LabelField("It is strongly recommended to NOT overwrite the results header items! " +
                                           "\nNow, the list below will NOT automatically update anymore! " +
                                           "\nYou have to do this by yourself.", noteStyle);
            }
            EditorGUILayout.PropertyField(resultsHeaderItems, true);
            
            EditorGUILayout.Space();
            EditorGUILayout.PropertyField(onQuestionnaireFinished, true);

            serializedObject.ApplyModifiedProperties();
        }
    }
}
