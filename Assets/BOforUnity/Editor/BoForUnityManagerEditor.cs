/*  ────────────────────────────────────────────────────────────────────────────
    BoForUnityManagerEditor.cs
    Full inspector script with support for a third ReorderableList: **contexts**
    (same Key / Value structure as parameters and objectives).
    ─────────────────────────────────────────────────────────────────────────── */

using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace BOforUnity.Editor
{
    [CustomEditor(typeof(BoForUnityManager))]
    public class BoForUnityManagerEditor : UnityEditor.Editor
    {
        // ───────── cached original strings (server settings) ─────────
        private string originalUploadURL, originalGroupName, originalDownloadURLGroup;
        private string originalLongitudinalName, originalDownloadURLLongitudinal;
        private string originalLockFileName, originalDownloadURLLock, originalPythonPath;

        // ───────── SerializedProperty references (UI widgets) ────────
        private SerializedProperty outputTextProp, loadingObjProp, nextButtonProp;
        private SerializedProperty welcomePanelProp, optimizerStatePanelProp;

        private SerializedProperty batchSizeProp, numRestartsProp, rawSamplesProp;
        private SerializedProperty nIterationsProp, mcSamplesProp, nInitialProp, seedProp;
        private SerializedProperty warmStartProp, perfectRatingActiveProp, perfectRatingInInitialRoundsProp;
        private SerializedProperty initialParametersDataPathProp, initialObjectivesDataPathProp;

        private SerializedProperty totalIterationsProp;
        private SerializedProperty userIdProp, conditionIdProp, groupIdProp;

        // ───────── Reorderable lists ─────────
        private ReorderableList parameterList;
        private ReorderableList objectiveList;
        private ReorderableList contextList;          // NEW

        private string initDataPath;

        // =====================================================================
        //  OnEnable   –   cache properties & build lists
        // =====================================================================
        private void OnEnable()
        {
            /* PARAMETER LIST -------------------------------------------------- */
            parameterList = new ReorderableList(serializedObject,
                                                serializedObject.FindProperty("parameters"),
                                                true, true, true, true);
            parameterList.drawHeaderCallback    = rect => EditorGUI.LabelField(rect, "Parameters");
            parameterList.drawElementCallback   = DrawParameterListItems;
            parameterList.elementHeightCallback = GetParameterListItemHeight;

            /* OBJECTIVE LIST -------------------------------------------------- */
            var objectivesProp = serializedObject.FindProperty("objectives");
            objectiveList = new ReorderableList(serializedObject, objectivesProp,
                                                true, true, true, true);
            objectiveList.drawHeaderCallback    = r => EditorGUI.LabelField(r, "Objectives");
            objectiveList.drawElementCallback   = DrawObjectiveListItems;
            objectiveList.elementHeightCallback = GetObjectiveElementHeight;

            /* CONTEXT LIST ---------------------------------------------------- */
            var contextsProp = serializedObject.FindProperty("contexts");
            contextList = new ReorderableList(serializedObject, contextsProp,
                                              true, true, true, true);
            contextList.drawHeaderCallback    = r => EditorGUI.LabelField(r, "Contexts");
            contextList.drawElementCallback   = DrawContextListItems;
            contextList.elementHeightCallback = GetContextElementHeight;

            /* UI references --------------------------------------------------- */
            outputTextProp          = serializedObject.FindProperty("outputText");
            loadingObjProp          = serializedObject.FindProperty("loadingObj");
            nextButtonProp          = serializedObject.FindProperty("nextButton");
            welcomePanelProp        = serializedObject.FindProperty("welcomePanel");
            optimizerStatePanelProp = serializedObject.FindProperty("optimizerStatePanel");

            /* BO hyper-parameters -------------------------------------------- */
            batchSizeProp   = serializedObject.FindProperty("batchSize");
            numRestartsProp = serializedObject.FindProperty("numRestarts");
            rawSamplesProp  = serializedObject.FindProperty("rawSamples");
            nIterationsProp = serializedObject.FindProperty("nIterations");
            mcSamplesProp   = serializedObject.FindProperty("mcSamples");
            nInitialProp    = serializedObject.FindProperty("nInitial");
            seedProp        = serializedObject.FindProperty("seed");
            warmStartProp   = serializedObject.FindProperty("warmStart");
            perfectRatingActiveProp        = serializedObject.FindProperty("perfectRatingActive");
            perfectRatingInInitialRoundsProp = serializedObject.FindProperty("perfectRatingInInitialRounds");
            initialParametersDataPathProp  = serializedObject.FindProperty("initialParametersDataPath");
            initialObjectivesDataPathProp  = serializedObject.FindProperty("initialObjectivesDataPath");

            /* Study settings -------------------------------------------------- */
            totalIterationsProp = serializedObject.FindProperty("totalIterations");
            userIdProp          = serializedObject.FindProperty("userId");
            conditionIdProp     = serializedObject.FindProperty("conditionId");
            groupIdProp         = serializedObject.FindProperty("groupId");

            initDataPath = Path.Combine(Application.dataPath, "StreamingAssets", "BOData", "InitData");
        }

        // =====================================================================
        //  Inspector GUI
        // =====================================================================
        public override void OnInspectorGUI()
        {
            var script = (BoForUnityManager)target;
            serializedObject.Update();

            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(2));
            parameterList.DoLayoutList();
            EditorGUILayout.Space();

            objectiveList.DoLayoutList();
            EditorGUILayout.Space();

            contextList.DoLayoutList();   // show contexts list
            EditorGUILayout.Space();

            CheckAndSetDefaultValues(script);
            DrawSettingsConfiguration(script);

            serializedObject.ApplyModifiedProperties();
        }

        // =====================================================================
        //  DRAWERS : Parameters
        // =====================================================================
        private void DrawParameterListItems(Rect rect, int index,bool isActive,bool isFocused)
        {
            SerializedProperty element = parameterList.serializedProperty.GetArrayElementAtIndex(index);
            SerializedProperty key   = element.FindPropertyRelative("key");
            SerializedProperty value = element.FindPropertyRelative("value");

            // internal value props
            SerializedProperty Value         = value.FindPropertyRelative("Value");
            SerializedProperty lowerBound    = value.FindPropertyRelative("lowerBound");
            SerializedProperty upperBound    = value.FindPropertyRelative("upperBound");
            SerializedProperty isDiscrete    = value.FindPropertyRelative("isDiscrete");
            SerializedProperty step          = value.FindPropertyRelative("step");
            SerializedProperty scriptRef     = value.FindPropertyRelative("scriptReference");
            SerializedProperty variableName  = value.FindPropertyRelative("variableName");
            SerializedProperty goNameProp    = value.FindPropertyRelative("gameObjectName");
            SerializedProperty scriptNameProp= value.FindPropertyRelative("scriptName");

            float pad = 5f, line = EditorGUIUtility.singleLineHeight;
            float y = rect.y + pad/2;

            EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), key, GUIContent.none);
            y += line + pad/2;

            value.isExpanded = EditorGUI.Foldout(new Rect(rect.x, y, rect.width, line),
                                                 value.isExpanded, "", true);
            y += line + pad/2;

            if (!value.isExpanded) return;
            EditorGUI.indentLevel++;

            EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), Value);        y+=line+pad/2;
            EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), lowerBound);   y+=line+pad/2;
            EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), upperBound);   y+=line+pad/2;
            EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), isDiscrete);   y+=line+pad/2;
            EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), step);         y+=line+pad/2;

            EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), scriptRef,
                                    new GUIContent("Script Reference"));
            y += line + pad/2;

            MonoBehaviour scriptObj = scriptRef.objectReferenceValue as MonoBehaviour;
            if (scriptObj != null)
            {
                goNameProp.stringValue = scriptObj.gameObject.name;
                scriptNameProp.stringValue = scriptObj.GetType().Name;

                var vars = new List<string>();
                foreach (FieldInfo f in scriptObj.GetType().GetFields(BindingFlags.Instance|BindingFlags.Public))
                    if (f.FieldType == typeof(float)) vars.Add(f.Name);
                foreach (PropertyInfo p in scriptObj.GetType().GetProperties(BindingFlags.Instance|BindingFlags.Public))
                    if (p.PropertyType == typeof(float)) vars.Add(p.Name);

                int sel = Mathf.Max(0, vars.IndexOf(variableName.stringValue));
                sel = EditorGUI.Popup(new Rect(rect.x, y, rect.width, line),
                                      "Script Variable", sel, vars.ToArray());
                variableName.stringValue = vars.Count>0 ? vars[sel] : "";
                y += line + pad/2;
            }
            else
            {
                EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), variableName);
                y += line + pad/2;
            }

            EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), goNameProp);    y+=line+pad/2;
            EditorGUI.PropertyField(new Rect(rect.x, y, rect.width, line), scriptNameProp);
            EditorGUI.indentLevel--;
        }

        private float GetParameterListItemHeight(int index)
        {
            SerializedProperty element = parameterList.serializedProperty.GetArrayElementAtIndex(index);
            bool expanded = element.FindPropertyRelative("value").isExpanded;
            float line = EditorGUIUtility.singleLineHeight, pad = 5;
            return expanded ? line*11 + pad*11 : line*2 + pad*2;
        }

        // =====================================================================
        //  DRAWERS : Objectives
        // =====================================================================
        private void DrawObjectiveListItems(Rect rect,int index,bool isActive,bool isFocused)
        {
            SerializedProperty element = objectiveList.serializedProperty.GetArrayElementAtIndex(index);
            SerializedProperty key   = element.FindPropertyRelative("key");
            SerializedProperty value = element.FindPropertyRelative("value");

            float pad  = 5f;
            rect.y    += pad/2;

            EditorGUI.PropertyField(
                new Rect(rect.x, rect.y, rect.width, EditorGUIUtility.singleLineHeight),
                key, GUIContent.none);

            EditorGUI.indentLevel++;
            EditorGUI.PropertyField(
                new Rect(rect.x,
                         rect.y + EditorGUIUtility.singleLineHeight + 2,
                         rect.width,
                         EditorGUI.GetPropertyHeight(value)),
                value, GUIContent.none, true);
            EditorGUI.indentLevel--;
        }

        private float GetObjectiveElementHeight(int index)
        {
            SerializedProperty element = objectiveList.serializedProperty.GetArrayElementAtIndex(index);
            float pad = 5f;
            return EditorGUIUtility.singleLineHeight +
                   EditorGUI.GetPropertyHeight(element.FindPropertyRelative("value")) +
                   EditorGUIUtility.standardVerticalSpacing + 2 + pad;
        }

        // =====================================================================
        //  DRAWERS : Contexts   (mirror objectives)
        // =====================================================================
        private void DrawContextListItems(Rect rect,int index,bool isActive,bool isFocused)
        {
            SerializedProperty element = contextList.serializedProperty.GetArrayElementAtIndex(index);
            SerializedProperty key   = element.FindPropertyRelative("key");
            SerializedProperty value = element.FindPropertyRelative("value");

            float pad  = 5f;
            rect.y    += pad/2;

            EditorGUI.PropertyField(
                new Rect(rect.x, rect.y, rect.width, EditorGUIUtility.singleLineHeight),
                key, GUIContent.none);

            EditorGUI.indentLevel++;
            EditorGUI.PropertyField(
                new Rect(rect.x,
                         rect.y + EditorGUIUtility.singleLineHeight + 2,
                         rect.width,
                         EditorGUI.GetPropertyHeight(value)),
                value, GUIContent.none, true);
            EditorGUI.indentLevel--;
        }

        private float GetContextElementHeight(int index)
        {
            SerializedProperty element = contextList.serializedProperty.GetArrayElementAtIndex(index);
            float pad = 5f;
            return EditorGUIUtility.singleLineHeight +
                   EditorGUI.GetPropertyHeight(element.FindPropertyRelative("value")) +
                   EditorGUIUtility.standardVerticalSpacing + 2 + pad;
        }

        // =====================================================================
        //  SETTINGS PANELS  (unchanged from your original script)
        // =====================================================================
        private void DrawSettingsConfiguration(BoForUnityManager script)
        {
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));

            EditorGUILayout.LabelField("Python Settings", EditorStyles.boldLabel);
            GUIContent manualLbl = new GUIContent("Manually Installed Python",
                "Python was manually installed and not through the project's installer");
            script.setLocalPython(EditorGUILayout.Toggle(manualLbl, script.getLocalPython()));
            EditorGUILayout.Space();

            if (script.getLocalPython())
            {
                script.setPythonPath(EditorGUILayout.TextField("Path of Python Executable:",
                                                               script.getPythonPath()));
                EditorGUILayout.LabelField(
                    "Ensure path matches your OS (macOS vs Windows).", EditorStyles.helpBox);
            }

            // Study IDs -------------------------------------------------------
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Study Settings", EditorStyles.boldLabel);
            script.userId      = string.IsNullOrEmpty(script.userId)      ? "-1" : script.userId;
            script.conditionId = string.IsNullOrEmpty(script.conditionId) ? "-1" : script.conditionId;
            script.groupId     = string.IsNullOrEmpty(script.groupId)     ? "-1" : script.groupId;

            EditorGUILayout.PropertyField(userIdProp);
            EditorGUILayout.PropertyField(conditionIdProp);
            EditorGUILayout.PropertyField(groupIdProp);
            EditorGUILayout.LabelField("Default for all IDs is -1.", EditorStyles.helpBox);

            // Warm-start & perfect-rating ------------------------------------
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Warm Start & Perfect Rating Settings", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(warmStartProp);
            EditorGUILayout.PropertyField(perfectRatingActiveProp);
            if (perfectRatingActiveProp.boolValue)
                EditorGUILayout.PropertyField(perfectRatingInInitialRoundsProp);

            EditorGUILayout.LabelField(
                "Warm start skips N-initial rounds; perfect rating may terminate early.",
                EditorStyles.helpBox);

            if (warmStartProp.boolValue)
            {
                EditorGUILayout.PropertyField(initialParametersDataPathProp);
                EditorGUILayout.LabelField(initDataPath + "/" +
                                           initialParametersDataPathProp.stringValue);
                EditorGUILayout.PropertyField(initialObjectivesDataPathProp);
                EditorGUILayout.LabelField(initDataPath + "/" +
                                           initialObjectivesDataPathProp.stringValue);
                EditorGUILayout.LabelField(
                    "Provide only file name (no _ or ,).", EditorStyles.helpBox);
                script.nInitial = 0;
            }

            // Hyper-parameters -----------------------------------------------
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("BO Hyper-parameters", EditorStyles.boldLabel);
            if (!warmStartProp.boolValue) EditorGUILayout.PropertyField(nInitialProp);
            EditorGUILayout.PropertyField(nIterationsProp);
            EditorGUILayout.LabelField(
                "Total iterations = N-Initial + N-Iterations (N-Initial ≥ 2).", EditorStyles.helpBox);

            int total = (warmStartProp.boolValue ? 0 : nInitialProp.intValue) +
                        nIterationsProp.intValue;
            totalIterationsProp.intValue = total;
            EditorGUILayout.LabelField("Total Iterations", total.ToString(), EditorStyles.boldLabel);
            EditorGUILayout.Space();

            EditorGUILayout.PropertyField(batchSizeProp);
            EditorGUILayout.PropertyField(numRestartsProp);
            EditorGUILayout.PropertyField(rawSamplesProp);
            EditorGUILayout.PropertyField(mcSamplesProp);
            EditorGUILayout.PropertyField(seedProp);

            // Game object refs ----------------------------------------------
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("GameObject References", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(outputTextProp);
            EditorGUILayout.PropertyField(loadingObjProp);
            EditorGUILayout.PropertyField(nextButtonProp);
            EditorGUILayout.PropertyField(welcomePanelProp);
            EditorGUILayout.PropertyField(optimizerStatePanelProp);
        }

        private void CheckAndSetDefaultValues(BoForUnityManager script)
        {
            if (string.IsNullOrEmpty(script.getPythonPath()))
                script.setPythonPath("/usr/local/bin/python3");
        }
    }
}

