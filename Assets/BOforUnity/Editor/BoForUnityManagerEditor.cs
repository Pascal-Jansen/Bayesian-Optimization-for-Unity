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
        private string originalUploadURL;
        private string originalGroupName;
        private string originalDownloadURLGroup;
        private string originalLongitudinalName;
        private string originalDownloadURLLongitudinal;
        private string originalLockFileName;
        private string originalDownloadURLLock;

        private SerializedProperty outputTextProp;
        private SerializedProperty loadingObjProp;
        private SerializedProperty nextButtonProp;
        //private SerializedProperty endSimProp;
        private SerializedProperty welcomePanelProp;
        private SerializedProperty optimizerStatePanelProp;
        
        private SerializedProperty nSamplingIterProp;
        private SerializedProperty nOptimizationIterProp;

        private SerializedProperty batchSizeProp;
        private SerializedProperty numRestartsProp;
        private SerializedProperty rawSamplesProp;
        private SerializedProperty mcSamplesProp;
        private SerializedProperty seedProp;
        private SerializedProperty warmStartProp;
        private SerializedProperty perfectRatingActiveProp;
        private SerializedProperty perfectRatingInInitialRoundsProp;
        private SerializedProperty initialParametersDataPathProp;
        private SerializedProperty initialObjectivesDataPathProp;

        private SerializedProperty totalIterationsProp;

        private SerializedProperty userIdProp;
        private SerializedProperty conditionIdProp;
        private SerializedProperty groupIdProp;
        
        private ReorderableList parameterList;
        private ReorderableList objectiveList;

        private string initDataPath;

        private SerializedProperty enableSamplingEditProp;

        private void OnEnable()
        {
            SerializedProperty parametersProperty = serializedObject.FindProperty("parameters");
            // Initialize your ReorderableList and set the elementHeightCallback
            parameterList = new ReorderableList(serializedObject, serializedObject.FindProperty("parameters"), true, true, true, true)
            {
                drawElementCallback = DrawParameterListItems,
                elementHeightCallback = GetParameterListItemHeight
            };
            parameterList.drawHeaderCallback = (Rect rect) => EditorGUI.LabelField(rect, "Parameters");
            
            SerializedProperty objectivesProperty = serializedObject.FindProperty("objectives");
            objectiveList = new ReorderableList(serializedObject, objectivesProperty, true, true, true, true);
            objectiveList.drawHeaderCallback = (Rect rect) => EditorGUI.LabelField(rect, "Objectives");
            objectiveList.drawElementCallback = DrawObjectiveListItems;
            objectiveList.elementHeightCallback = GetObjectiveElementHeight;
            
            outputTextProp = serializedObject.FindProperty("outputText");
            loadingObjProp = serializedObject.FindProperty("loadingObj");
            nextButtonProp = serializedObject.FindProperty("nextButton");
            welcomePanelProp = serializedObject.FindProperty("welcomePanel");
            optimizerStatePanelProp = serializedObject.FindProperty("optimizerStatePanel");
            //endSimProp = serializedObject.FindProperty("endOfSimulation");
            
            nSamplingIterProp = serializedObject.FindProperty("numSamplingIterations");
            nOptimizationIterProp = serializedObject.FindProperty("numOptimizationIterations");
            
            batchSizeProp = serializedObject.FindProperty("batchSize");
            numRestartsProp = serializedObject.FindProperty("numRestarts");
            rawSamplesProp = serializedObject.FindProperty("rawSamples");
            mcSamplesProp = serializedObject.FindProperty("mcSamples");
            seedProp = serializedObject.FindProperty("seed");
            warmStartProp = serializedObject.FindProperty("warmStart");
            perfectRatingActiveProp = serializedObject.FindProperty("perfectRatingActive");
            perfectRatingInInitialRoundsProp = serializedObject.FindProperty("perfectRatingInInitialRounds");
            initialParametersDataPathProp = serializedObject.FindProperty("initialParametersDataPath");
            initialObjectivesDataPathProp = serializedObject.FindProperty("initialObjectivesDataPath");

            totalIterationsProp = serializedObject.FindProperty("totalIterations");
            
            userIdProp = serializedObject.FindProperty("userId");
            conditionIdProp = serializedObject.FindProperty("conditionId");
            groupIdProp = serializedObject.FindProperty("groupId");
            
            initDataPath = Path.Combine(Application.dataPath, "StreamingAssets", "BOData", "InitData");

            enableSamplingEditProp = serializedObject.FindProperty("enableSamplingEdit");
    }

        public override void OnInspectorGUI()
        {
            BoForUnityManager script = (BoForUnityManager)target;

            serializedObject.Update();

            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(2));
            // Draw Parameter and Objective Lists first
            parameterList.DoLayoutList();
            EditorGUILayout.Space();
            objectiveList.DoLayoutList();

            DrawSettingsConfiguration(script);

            serializedObject.ApplyModifiedProperties();
        }

        private void DrawSettingsConfiguration(BoForUnityManager script)
        {
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));

            // ── Python Settings ─────────────────────────────────────────────────────
            EditorGUILayout.LabelField("Python Settings", EditorStyles.boldLabel);
            var localPyLabel = new GUIContent("Manually Installed Python",
                "Use a locally installed Python instead of the project’s installer.");
            script.setLocalPython(EditorGUILayout.Toggle(localPyLabel, script.getLocalPython()));
            EditorGUILayout.Space();

            if (script.getLocalPython())
            {
                script.setPythonPath(EditorGUILayout.TextField("Path of Python Executable:", ""));
                EditorGUILayout.LabelField(
                    "Ensure a valid path for your OS (Windows/macOS differ).",
                    EditorStyles.helpBox
                );
            }

            // ── Study Settings ──────────────────────────────────────────────────────
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Study Settings", EditorStyles.boldLabel);

            if (string.IsNullOrEmpty(script.userId))      script.userId = "-1";
            if (string.IsNullOrEmpty(script.conditionId)) script.conditionId = "-1";
            if (string.IsNullOrEmpty(script.groupId))     script.groupId = "-1";

            EditorGUILayout.PropertyField(userIdProp);
            EditorGUILayout.PropertyField(conditionIdProp);
            EditorGUILayout.PropertyField(groupIdProp);
            EditorGUILayout.LabelField("Default values for userID, conditionID and groupID are -1.", EditorStyles.helpBox);

            // ── Problem Setup (not hyperparameters) ─────────────────────────────────
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Problem Setup", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Design Parameters (d)", parameterList.count.ToString(), EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Design Objectives (m)", objectiveList.count.ToString(), EditorStyles.boldLabel);

            // ── Optimization Budget (iterations & termination) ──────────────────────
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Optimization Budget", EditorStyles.boldLabel);

            // Warm start & perfect rating belong to run control / termination
            EditorGUILayout.PropertyField(warmStartProp, new GUIContent("Warm Start",
                "Skip sampling by loading initial data. Sets Sampling Iterations to 0."));
            EditorGUILayout.PropertyField(perfectRatingActiveProp, new GUIContent("Perfect Rating Active",
                "Terminate early when perfect rating is reached."));
            if (perfectRatingActiveProp.boolValue)
            {
                EditorGUILayout.PropertyField(perfectRatingInInitialRoundsProp, new GUIContent(
                    "Allow Perfect Rating in Sampling",
                    "Permit termination during the sampling phase."));
            }

            if (warmStartProp.boolValue)
            {
                EditorGUILayout.PropertyField(initialParametersDataPathProp, new GUIContent("Initial Parameters File"));
                EditorGUILayout.LabelField(initDataPath + "/" + initialParametersDataPathProp.stringValue, EditorStyles.label);
                EditorGUILayout.PropertyField(initialObjectivesDataPathProp, new GUIContent("Initial Objectives File"));
                EditorGUILayout.LabelField(initDataPath + "/" + initialObjectivesDataPathProp.stringValue, EditorStyles.label);
                EditorGUILayout.LabelField(
                    "Provide only the file name. Avoid '_' and ',' in names.",
                    EditorStyles.helpBox
                );
                // Force sampling to zero when warm start is on
                script.numSamplingIterations = 0;
            }

            // Sampling iterations: default (2*d)+1 unless user explicitly enables manual edit
            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(warmStartProp.boolValue))
            {
                EditorGUILayout.PropertyField(
                    enableSamplingEditProp,
                    new GUIContent("Set Sampling Iterations Manually",
                        "Default is (2·d)+1. Enable to override.")
                );

                int defaultSampling = (2 * parameterList.count) + 1;

                // When manual edit is OFF, keep the default and lock the field.
                if (!enableSamplingEditProp.boolValue)
                {
                    if (nSamplingIterProp.intValue != defaultSampling)
                        nSamplingIterProp.intValue = defaultSampling;

                    using (new EditorGUI.DisabledScope(true))
                    {
                        EditorGUILayout.IntField(new GUIContent("Sampling Iterations"), nSamplingIterProp.intValue);
                    }
                }
                else
                {
                    // Manual edit ON
                    EditorGUILayout.PropertyField(nSamplingIterProp, new GUIContent("Sampling Iterations"));
                }

                EditorGUILayout.LabelField(
                    "Recommended sampling iterations default: (2 · d) + 1, where d is the number of design parameters.",
                    EditorStyles.helpBox
                );
            }

            EditorGUILayout.PropertyField(nOptimizationIterProp, new GUIContent("Optimization Iterations"));

            // Total iterations = sampling (or 0 with warm start) + optimization
            int sampling = warmStartProp.boolValue ? 0 : nSamplingIterProp.intValue;
            int total    = sampling + nOptimizationIterProp.intValue;
            totalIterationsProp.intValue = total;

            EditorGUILayout.LabelField("Total Iterations", total.ToString(), EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                "Total = Sampling Iterations + Optimization Iterations. Sampling must be ≥ 3 unless warm start is used.",
                EditorStyles.helpBox
            );

            // ── Model & Algorithm Hyperparameters ───────────────────────────────────
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Model & Algorithm Hyperparameters", EditorStyles.boldLabel);

            EditorGUILayout.PropertyField(batchSizeProp,     new GUIContent("Batch Size",
                "q evaluations per BO step."));
            EditorGUILayout.PropertyField(numRestartsProp,   new GUIContent("Optimizer Restarts",
                "LBFGS restarts for acquisition optimization."));
            EditorGUILayout.PropertyField(rawSamplesProp,    new GUIContent("Raw Samples",
                "Sobol samples for starting points."));
            EditorGUILayout.PropertyField(mcSamplesProp,     new GUIContent("MC Samples",
                "Samples for Monte Carlo acquisition estimates."));
            EditorGUILayout.PropertyField(seedProp,          new GUIContent("Random Seed",
                "Seed for reproducibility."));

            // ── GameObject References ───────────────────────────────────────────────
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("GameObject References", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(outputTextProp);
            EditorGUILayout.PropertyField(loadingObjProp);
            EditorGUILayout.PropertyField(nextButtonProp);
            EditorGUILayout.PropertyField(welcomePanelProp);
            EditorGUILayout.PropertyField(optimizerStatePanelProp);
        }

        /*
        private void DrawServerClientConnectionSettings(BoForUnityManager script)
        {
            EditorGUILayout.LabelField("Server Client Connection for LogFiles", EditorStyles.boldLabel);
            script.setServerClientCommunication(EditorGUILayout.Toggle("Server", script.getServerClientCommunication()));
            EditorGUILayout.Space();
            
            if (!script.getServerClientCommunication())
            {
                EditorGUILayout.LabelField("Attention! If the server is not used, make sure to select GroupID and LongitudinalID in the Object Parameter Controller before starting.", EditorStyles.helpBox);
            }
            else
            {
                script.setUploadURL(EditorGUILayout.TextField("Upload URL:", originalUploadURL));
                script.setGroupDatabaseName(EditorGUILayout.TextField("File Name GroupID Database:", originalGroupName));
                script.setDownloadURLGroupID(EditorGUILayout.TextField("Download URL GroupID:", originalDownloadURLGroup));
                script.setLongitudinalDatabaseName(EditorGUILayout.TextField("File Name LongitudinalID Database:", originalLongitudinalName));
                script.setDownloadURLLongitudinalID(EditorGUILayout.TextField("Download URL LongitudinalID:", originalDownloadURLLongitudinal));
                script.setLockFileName(EditorGUILayout.TextField("File Name LockFile:", originalLockFileName));
                script.setLockFileUrl(EditorGUILayout.TextField("Download URL Lock File:", originalDownloadURLLock));
            }
        }
        */
        private void BackupServerClientCommunicationValues(BoForUnityManager script)
        {
            /*
            originalUploadURL = script.getUploadURL();
            originalGroupName = script.getGroupDatabaseName();
            originalDownloadURLGroup = script.getDownloadURLGroupID();
            originalLongitudinalName = script.getLongitudinalDatabaseName();
            originalDownloadURLLongitudinal = script.getDownloadURLLongitudinalID();
            originalLockFileName = script.getLockFileName();
            originalDownloadURLLock = script.getLockFileUrl();
            */
        }

        private void DrawParameterListItems(Rect rect, int index, bool isActive, bool isFocused)
        {
            SerializedProperty element = parameterList.serializedProperty.GetArrayElementAtIndex(index);
            SerializedProperty key = element.FindPropertyRelative("key");
            SerializedProperty value = element.FindPropertyRelative("value");

            // Fields within ParameterArgs
            SerializedProperty Value = value.FindPropertyRelative("Value");
            SerializedProperty lowerBound = value.FindPropertyRelative("lowerBound");
            SerializedProperty upperBound = value.FindPropertyRelative("upperBound");

            float padding = 1.5f;
            float singleLineHeight = EditorGUIUtility.singleLineHeight;
            float fieldHeight = singleLineHeight + padding;
            float yOffset = rect.y + padding / 2;

            // Draw the key field
            EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), key, GUIContent.none);
            yOffset += fieldHeight;

            // Draw foldout for ParameterArgs
            value.isExpanded = EditorGUI.Foldout(new Rect(rect.x, yOffset, rect.width, singleLineHeight), value.isExpanded, "", true);
            yOffset += fieldHeight;

            if (value.isExpanded)
            {
                EditorGUI.indentLevel++;

                // Draw the existing fields in ParameterArgs
                EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), Value);
                yOffset += fieldHeight;
                EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), lowerBound);
                yOffset += fieldHeight;
                EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), upperBound);

                EditorGUI.indentLevel--;
            }
        }



        private float GetParameterListItemHeight(int index)
        {
            SerializedProperty element = parameterList.serializedProperty.GetArrayElementAtIndex(index);
            SerializedProperty value = element.FindPropertyRelative("value");

            float singleLineHeight = EditorGUIUtility.singleLineHeight;
            float totalHeight = singleLineHeight * 2 + 10f; // Key + Foldout

            if (value.isExpanded)
            {
                totalHeight += singleLineHeight * 3 + 1.5f * 3; // 5 fields
            }

            return totalHeight;
        }



        private void DrawObjectiveListItems(Rect rect, int index, bool isActive, bool isFocused)
        {
            SerializedProperty element = objectiveList.serializedProperty.GetArrayElementAtIndex(index);
            SerializedProperty key = element.FindPropertyRelative("key");
            SerializedProperty value = element.FindPropertyRelative("value");

            float padding = 5f;
            rect.y += padding / 2;

            EditorGUI.PropertyField(new Rect(rect.x, rect.y, rect.width, EditorGUIUtility.singleLineHeight), key, GUIContent.none);
            EditorGUI.indentLevel++;
            EditorGUI.PropertyField(new Rect(rect.x, rect.y + EditorGUIUtility.singleLineHeight + 2, rect.width, EditorGUI.GetPropertyHeight(value)), value, GUIContent.none, true);
            EditorGUI.indentLevel--;
            rect.y += padding;
        }

        private float GetParameterElementHeight(int index)
        {
            SerializedProperty element = parameterList.serializedProperty.GetArrayElementAtIndex(index);
            float padding = 5;
            return EditorGUIUtility.singleLineHeight + EditorGUI.GetPropertyHeight(element.FindPropertyRelative("value")) + EditorGUIUtility.standardVerticalSpacing + 2 + padding;
        }

        private float GetObjectiveElementHeight(int index)
        {
            SerializedProperty element = objectiveList.serializedProperty.GetArrayElementAtIndex(index);
            float padding = 5;
            return EditorGUIUtility.singleLineHeight + EditorGUI.GetPropertyHeight(element.FindPropertyRelative("value")) + EditorGUIUtility.standardVerticalSpacing + 2 + padding;
        }
    }
}
