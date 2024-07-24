using System;
using QuestionnaireToolkit.Scripts;
using UnityEditor;
using UnityEngine;

namespace QuestionnaireToolkit.Editor
{
    [CanEditMultipleObjects]
    [CustomEditor(typeof(QTQuestionPageManager))]
    public class QTQuestionPageManagerEditor : UnityEditor.Editor
    {
        private SerializedProperty showFullscreenText;
        private SerializedProperty fullscreenText;
        
        private SerializedProperty showTopPanel;
        private SerializedProperty instructionText;
        
        private ReorderableList questionItems;
        
        private SerializedProperty headerName;
        private SerializedProperty question;
        private SerializedProperty automaticFill;

        private SerializedProperty selectedItem;
        private SerializedProperty desiredPage;
        private SerializedProperty moveItemHint;

        private QTQuestionPageManager pageManager;
        private GUIStyle noteStyle;
        private Texture image;
        private Texture logo;
        
        void OnEnable()
        {
            noteStyle = new GUIStyle();
            noteStyle.wordWrap = true;
            noteStyle.normal.textColor = Color.white;
            noteStyle.fontStyle = FontStyle.Bold;
            
            showFullscreenText = serializedObject.FindProperty("showFullscreenText");
            fullscreenText = serializedObject.FindProperty("fullscreenText");
            
            showTopPanel = serializedObject.FindProperty("showTopPanel");
            instructionText = serializedObject.FindProperty("instructionText");
            headerName = serializedObject.FindProperty("headerName");
            question = serializedObject.FindProperty("question");
            automaticFill = serializedObject.FindProperty("automaticFill");
            selectedItem = serializedObject.FindProperty("selectedItem");
            desiredPage = serializedObject.FindProperty("desiredPage");
            
            moveItemHint = serializedObject.FindProperty("moveItemHint");
            
            questionItems = new ReorderableList(serializedObject.FindProperty("questionItems"), false, true, true);
            questionItems.elementNameProperty = "Question Items";
            
            pageManager = (QTQuestionPageManager) target;
            questionItems.onChangedCallback += (list) =>
            {
                pageManager.ReorderItems(list.Length, list.Index);
            };
            questionItems.onSelectCallback += (list) =>
            {
                pageManager.selectedIndex = list.Index;
                pageManager.selectedItem = pageManager.questionItems[list.Index];
            };
            questionItems.onRemoveCallback += (list) =>
            {
                var i = list.Index;
                list.RemoveItem(list.Index);
                pageManager.DeleteItem(list.Length, i);
            };
            
            image = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/Banner/PageBanner.png");
            logo = AssetDatabase.LoadAssetAtPath<Texture>("Assets/QuestionnaireToolkit/Textures/QT_Logo_Mini2_Right.png");
        }

        public override void OnInspectorGUI()
        {
            try
            {
                if (pageManager._questionnaireManager.selectedPage != pageManager.transform.GetSiblingIndex()
                    && pageManager._questionnaireManager.transform.GetSiblingIndex() == pageManager._qtManager.selectedQuestionnaire
                    && !pageManager._questionnaireManager.running)
                    pageManager.ShowThisPage();
            }
            catch (Exception)
            {
                if(pageManager._questionnaireManager.selectedPage != pageManager.transform.GetSiblingIndex() && !pageManager._questionnaireManager.running) 
                    pageManager.ShowThisPage();
            }
            
            serializedObject.Update();

            GUILayout.Space(5);
            var rect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.Height(30));
            EditorGUI.DrawRect(new Rect(rect.x, rect.y + 3f, rect.width, rect.height), new Color(0.0f, 0.125f, 0.376f, 1));
            EditorGUI.DrawRect(new Rect(rect.x + 2, rect.y + 5, rect.width - 4f, rect.height - 4), Color.white);
            GUI.DrawTexture(new Rect(rect.x - 29, rect.y, 250, rect.height + 6f), image, ScaleMode.ScaleToFit);
            GUI.DrawTexture(new Rect(rect.x + rect.width - 50, rect.y + 4, 60, rect.height - 3), logo, ScaleMode.ScaleToFit);
            GUILayout.Space(5);
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Show Fullscreen Text",  GUILayout.Width(EditorGUIUtility.labelWidth));
            showFullscreenText.boolValue = EditorGUILayout.Toggle( showFullscreenText.boolValue);
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Fullscreen Text",  GUILayout.Width(EditorGUIUtility.labelWidth));
            fullscreenText.stringValue = EditorGUILayout.TextArea( fullscreenText.stringValue );
            GUILayout.EndHorizontal();
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Show Instruction Panel",  GUILayout.Width(EditorGUIUtility.labelWidth));
            showTopPanel.boolValue = EditorGUILayout.Toggle( showTopPanel.boolValue);
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Instruction Text",  GUILayout.Width(EditorGUIUtility.labelWidth));
            instructionText.stringValue = EditorGUILayout.TextArea( instructionText.stringValue );
            GUILayout.EndHorizontal();

            if (!showFullscreenText.boolValue)
            {
                //draw the list using GUILayout, you can of course specify your own position and label
                questionItems.DoLayoutList();
                
                GUILayout.Label("Question Item Creation", EditorStyles.boldLabel);
                pageManager.type = (QTQuestionPageManager.QuestionItemsEnum)EditorGUILayout.EnumPopup("Type", pageManager.type);
                
                GUILayout.BeginHorizontal();
                GUILayout.Label("Header Name",  GUILayout.Width(EditorGUIUtility.labelWidth));
                headerName.stringValue = EditorGUILayout.TextArea( headerName.stringValue );
                GUILayout.EndHorizontal();
            
                GUILayout.BeginHorizontal();
                GUILayout.Label("Question",  GUILayout.Width(EditorGUIUtility.labelWidth));
                question.stringValue = EditorGUILayout.TextArea( question.stringValue );
                GUILayout.EndHorizontal();
            
                GUILayout.BeginHorizontal();
                GUILayout.Label("Automatic Fill",  GUILayout.Width(EditorGUIUtility.labelWidth));
                automaticFill.boolValue = EditorGUILayout.Toggle( automaticFill.boolValue);
                GUILayout.EndHorizontal();

                if (GUILayout.Button("Create Item")) { pageManager.AddItem(); }
                GUILayout.BeginHorizontal();
                if (GUILayout.Button("Copy Selection")) { pageManager.CopySelection(); }
                if (pageManager.moveItemPressed)
                {
                    if (GUILayout.Button("Cancel"))
                    {
                        pageManager.moveItemPressed = false;
                        pageManager.desiredPage = pageManager.transform.parent.childCount-1;
                    }
                }
                else if (GUILayout.Button("Move Item"))
                {
                    if (pageManager.selectedIndex >= 0)
                    {
                        pageManager.moveItemPressed = true;
                        pageManager.desiredPage = pageManager.transform.parent.childCount-1;
                    }
                }
                GUILayout.EndHorizontal();

                if (pageManager.moveItemPressed)
                {
                    GUILayout.Label("Move Item To Desired Page", EditorStyles.boldLabel);
                    if (pageManager.wrongPage)
                    {
                        GUILayout.Label(moveItemHint.stringValue);
                    }
                    GUILayout.BeginHorizontal();
                    GUILayout.Label("Selected Item",  GUILayout.Width(EditorGUIUtility.labelWidth));
                    selectedItem.objectReferenceValue = EditorGUILayout.ObjectField(selectedItem.objectReferenceValue, typeof(GameObject), false);
                    GUILayout.EndHorizontal();
                    GUILayout.BeginHorizontal();
                    GUILayout.Label("Desired Page Index",  GUILayout.Width(EditorGUIUtility.labelWidth));
                    desiredPage.intValue = EditorGUILayout.IntField( desiredPage.intValue );
                    GUILayout.EndHorizontal();
                    if (GUILayout.Button("Confirm")) { pageManager.MoveItem(); }
                }
            }

            serializedObject.ApplyModifiedProperties();

        }
    }
}

