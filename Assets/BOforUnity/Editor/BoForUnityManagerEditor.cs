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
        private SerializedProperty optimizerStatePanelProp;
        
        private ReorderableList parameterList;
        private ReorderableList objectiveList;

        private void OnEnable()
        {
            SerializedProperty parametersProperty = serializedObject.FindProperty("parameters");
            parameterList = new ReorderableList(serializedObject, parametersProperty, true, true, true, true);
            parameterList.drawHeaderCallback = (Rect rect) => EditorGUI.LabelField(rect, "Parameters");
            parameterList.drawElementCallback = DrawParameterListItems;
            parameterList.elementHeightCallback = GetParameterElementHeight;

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
        }

        public override void OnInspectorGUI()
        {
            BoForUnityManager script = (BoForUnityManager)target;

            serializedObject.Update();

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
            EditorGUILayout.LabelField("Settings Configuration", EditorStyles.largeLabel);
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
            EditorGUILayout.LabelField("GameObject References", EditorStyles.largeLabel);
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

            float padding = 5f;
            rect.y += padding / 2;

            EditorGUI.PropertyField(new Rect(rect.x, rect.y, rect.width, EditorGUIUtility.singleLineHeight), key, GUIContent.none);
            EditorGUI.indentLevel++;
            EditorGUI.PropertyField(new Rect(rect.x, rect.y + EditorGUIUtility.singleLineHeight + 2, rect.width, EditorGUI.GetPropertyHeight(value)), value, GUIContent.none, true);
            EditorGUI.indentLevel--;
            rect.y += padding;
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
