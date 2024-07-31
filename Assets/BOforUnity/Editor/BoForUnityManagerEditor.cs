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
        private string originalPythonPath;

        private SerializedProperty outputTextProp;
        private SerializedProperty loadingObjProp;
        private SerializedProperty nextButtonProp;
        //private SerializedProperty endSimProp;
        private SerializedProperty welcomePanelProp;
        //private SerializedProperty optimizerStatePanelProp;
        
        private SerializedProperty batchSizeProp;
        private SerializedProperty numRestartsProp;
        private SerializedProperty rawSamplesProp;
        private SerializedProperty nIterationsProp;
        private SerializedProperty mcSamplesProp;
        private SerializedProperty nInitialProp;
        private SerializedProperty seedProp;
        private SerializedProperty warmStartProp;
        private SerializedProperty initialParametersDataPathProp;
        private SerializedProperty initialObjectivesDataPathProp;

        private SerializedProperty totalIterationsProp;

        private SerializedProperty userIdProp;
        private SerializedProperty conditionIdProp;
        
        private ReorderableList parameterList;
        private ReorderableList objectiveList;

        private SerializedProperty optScene;
        private SerializedProperty simScene;
        private SerializedProperty questScene;
        private SerializedProperty finScene;

        private string initDataPath;
        
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
            //optimizerStatePanelProp = serializedObject.FindProperty("optimizerStatePanel");
            //endSimProp = serializedObject.FindProperty("endOfSimulation");
            
            batchSizeProp = serializedObject.FindProperty("batchSize");
            numRestartsProp = serializedObject.FindProperty("numRestarts");
            rawSamplesProp = serializedObject.FindProperty("rawSamples");
            nIterationsProp = serializedObject.FindProperty("nIterations");
            mcSamplesProp = serializedObject.FindProperty("mcSamples");
            nInitialProp = serializedObject.FindProperty("nInitial");
            seedProp = serializedObject.FindProperty("seed");
            warmStartProp = serializedObject.FindProperty("warmStart");
            initialParametersDataPathProp = serializedObject.FindProperty("initialParametersDataPath");
            initialObjectivesDataPathProp = serializedObject.FindProperty("initialObjectivesDataPath");

            totalIterationsProp = serializedObject.FindProperty("totalIterations");
            
            userIdProp = serializedObject.FindProperty("userId");
            conditionIdProp = serializedObject.FindProperty("conditionId");

            optScene = serializedObject.FindProperty("optimizerScene");
            simScene = serializedObject.FindProperty("simulationScene");
            questScene = serializedObject.FindProperty("questionnaireScene");
            finScene = serializedObject.FindProperty("finalScene");
            
            initDataPath = Path.Combine(Application.dataPath, "StreamingAssets", "BOData", "InitData");
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

            CheckAndSetDefaultValues(script);

            /*
            if (script.getServerClientCommunication())
            {
                BackupServerClientCommunicationValues(script);
            }*/

            if (script.getLocalPython())
            {
                originalPythonPath = script.getPythonPath();
            }

            DrawSettingsConfiguration(script);

            serializedObject.ApplyModifiedProperties();
        }

        private void DrawSettingsConfiguration(BoForUnityManager script)
        {
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            
            EditorGUILayout.LabelField("Python Settings", EditorStyles.boldLabel);
            // Server Client Connection settings
            //DrawServerClientConnectionSettings(script);

            //EditorGUILayout.LabelField("Location of Python Executable", EditorStyles.boldLabel);
            GUIContent manuallyInstalledPythonLabel = new GUIContent("Manually Installed Python", "Python was manually installed and not through the project's installation program");
            script.setLocalPython(EditorGUILayout.Toggle(manuallyInstalledPythonLabel, script.getLocalPython()));
            EditorGUILayout.Space();

            if (script.getLocalPython())
            {
                // Show the Python path input field only if Python is locally installed
                script.setPythonPath(EditorGUILayout.TextField("Path of Python Executable:", originalPythonPath));
                EditorGUILayout.LabelField("Attention! There are differences between macOS and Windows path. Please ensure you provide the correct path for your operating system.", EditorStyles.helpBox);
            }
            
            //EditorGUILayout.Space();
            //EditorGUILayout.LabelField("Iteration Controls", EditorStyles.largeLabel);
            //EditorGUILayout.PropertyField(endSimProp);
            
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            
            EditorGUILayout.LabelField("Study Settings", EditorStyles.boldLabel);
            // Set default values if properties are not assigned
            if (string.IsNullOrEmpty(script.userId))
            {
                script.userId = "-1";
            }
            if (string.IsNullOrEmpty(script.conditionId))
            {
                script.conditionId = "-1";
            }
            EditorGUILayout.PropertyField(userIdProp);
            EditorGUILayout.PropertyField(conditionIdProp);
            EditorGUILayout.LabelField("Default values for userID and conditionID is -1.", EditorStyles.helpBox);
            
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            
            EditorGUILayout.LabelField("Warm Start Settings", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(warmStartProp);
            EditorGUILayout.LabelField("Attention! If warm start is TRUE, the N Initial rounds will be skipped.", EditorStyles.helpBox);
            if (warmStartProp.boolValue)
            {
                EditorGUILayout.PropertyField(initialParametersDataPathProp);
                EditorGUILayout.LabelField(initDataPath + "/" + initialParametersDataPathProp.stringValue, EditorStyles.label);
                EditorGUILayout.PropertyField(initialObjectivesDataPathProp);
                EditorGUILayout.LabelField(initDataPath + "/" + initialObjectivesDataPathProp.stringValue, EditorStyles.label);
                EditorGUILayout.LabelField("Remember: You only need to provide the file name. No '_' or ',' allowed in the file name.", EditorStyles.helpBox);
            }
            
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));

            EditorGUILayout.LabelField("BO Hyper-parameters", EditorStyles.boldLabel);
            if (!warmStartProp.boolValue)
            {
                EditorGUILayout.PropertyField(nInitialProp);
            }
            EditorGUILayout.PropertyField(nIterationsProp);
            EditorGUILayout.LabelField("Attention! For the total number of iterations, these two numbers are added (N Initial + N Iterations).", EditorStyles.helpBox);
            // Calculate and display the sum of nInitial and nIterations
            var val = (warmStartProp.boolValue ? 0: nInitialProp.intValue) + nIterationsProp.intValue;
            totalIterationsProp.intValue = val;
            EditorGUILayout.LabelField("Total Iterations", val.ToString(), EditorStyles.boldLabel);
            EditorGUILayout.Space();
            EditorGUILayout.PropertyField(batchSizeProp);
            EditorGUILayout.PropertyField(numRestartsProp);
            EditorGUILayout.PropertyField(rawSamplesProp);
            EditorGUILayout.PropertyField(mcSamplesProp);
            EditorGUILayout.PropertyField(seedProp);
            
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));

            EditorGUILayout.LabelField("Scene References", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(optScene);
            EditorGUILayout.PropertyField(simScene);
            EditorGUILayout.PropertyField(questScene);
            EditorGUILayout.PropertyField(finScene);
            
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            
            EditorGUILayout.LabelField("GameObject References", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(outputTextProp);
            EditorGUILayout.PropertyField(loadingObjProp);
            EditorGUILayout.PropertyField(nextButtonProp);
            EditorGUILayout.PropertyField(welcomePanelProp);
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

        private void CheckAndSetDefaultValues(BoForUnityManager script)
        {
            /*
            if (string.IsNullOrEmpty(script.getUploadURL()))
            {
                script.setUploadURL("https://barakuda.de/longitudinal_save_csv.php");
            }

            if (string.IsNullOrEmpty(script.getGroupDatabaseName()))
            {
                script.setGroupDatabaseName("GroupID_Database.csv");
            }

            if (string.IsNullOrEmpty(script.getDownloadURLGroupID()))
            {
                script.setDownloadURLGroupID("https://barakuda.de/longitudinal_uploads/GroupID_Database.csv");
            }

            if (string.IsNullOrEmpty(script.getLongitudinalDatabaseName()))
            {
                script.setLongitudinalDatabaseName("LongitudinalID_Database.csv");
            }

            if (string.IsNullOrEmpty(script.getDownloadURLLongitudinalID()))
            {
                script.setDownloadURLLongitudinalID("https://barakuda.de/longitudinal_uploads/LongitudinalID_Database.csv");
            }

            if (string.IsNullOrEmpty(script.getLockFileName()))
            {
                script.setLockFileName("S_Lock_file.lock");
            }

            if (string.IsNullOrEmpty(script.getLockFileUrl()))
            {
                script.setLockFileUrl("https://barakuda.de/longitudinal_uploads/S_Lock_file.lock");
            }
            */

            if (string.IsNullOrEmpty(script.getPythonPath()))
            {
                script.setPythonPath("/usr/local/bin/python3");
            }
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
            SerializedProperty isDiscrete = value.FindPropertyRelative("isDiscrete");
            SerializedProperty step = value.FindPropertyRelative("step");
            SerializedProperty scriptReference = value.FindPropertyRelative("scriptReference");
            SerializedProperty variableName = value.FindPropertyRelative("variableName");

            float padding = 5f;
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
                yOffset += fieldHeight;
                EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), isDiscrete);
                yOffset += fieldHeight;
                EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), step);
                yOffset += fieldHeight;
                
                // Draw script reference field
                EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), scriptReference, new GUIContent("Script Reference"));
                yOffset += fieldHeight;

                MonoBehaviour script = scriptReference.objectReferenceValue as MonoBehaviour;
                float variableValue = 0.0f;

                if (script != null)
                {
                    value.FindPropertyRelative("gameObjectName").stringValue = script.gameObject.name;
                    value.FindPropertyRelative("scriptName").stringValue = script.GetType().Name;
                    
                    var variableNames = new List<string>();
                    FieldInfo[] fields = script.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public);
                    foreach (FieldInfo field in fields)
                    {
                        if (field.FieldType == typeof(float))
                        {
                            variableNames.Add(field.Name);
                        }
                    }

                    PropertyInfo[] properties = script.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public);
                    foreach (PropertyInfo property in properties)
                    {
                        if (property.PropertyType == typeof(float))
                        {
                            variableNames.Add(property.Name);
                        }
                    }

                    if (variableNames.Count > 0)
                    {
                        // Get the current selected index
                        int selectedIndex = Mathf.Max(0, variableNames.IndexOf(variableName.stringValue));

                        // Show the popup to select a variable
                        selectedIndex = EditorGUI.Popup(new Rect(rect.x, yOffset, rect.width, singleLineHeight), "Script Variable", selectedIndex, variableNames.ToArray());
                        string selectedVariable = variableNames[selectedIndex];

                        // Update the variable name property
                        variableName.stringValue = selectedVariable;
                        yOffset += fieldHeight;

                        // Get and show the Value field from the selected variable
                        FieldInfo selectedField = script.GetType().GetField(selectedVariable, BindingFlags.Instance | BindingFlags.Public);
                        PropertyInfo selectedProperty = script.GetType().GetProperty(selectedVariable, BindingFlags.Instance | BindingFlags.Public);
                        
                        if (selectedField != null)
                        {
                            variableValue = (float)selectedField.GetValue(script);
                        }
                        else if (selectedProperty != null)
                        {
                            variableValue = (float)selectedProperty.GetValue(script);
                        }
                    }
                }
                else
                {
                    EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), variableName);
                    yOffset += fieldHeight;
                }

                //EditorGUI.LabelField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), "Script Value: " + variableValue);
                EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), value.FindPropertyRelative("gameObjectName"));
                yOffset += fieldHeight;
                EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), value.FindPropertyRelative("scriptName"));
                
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
                totalHeight += singleLineHeight * 9 + 5f * 9; // 5 fields + script reference + variable dropdown + value field
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
