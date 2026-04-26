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
        private SerializedProperty localPythonProp;
        private SerializedProperty pythonPathProp;
        
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
        private SerializedProperty warmStartObjectiveFormatProp;
        private SerializedProperty optimizerBackendProp;
        private SerializedProperty cabopObjectiveModeProp;
        private SerializedProperty cabopUseCostAwareAcquisitionProp;
        private SerializedProperty cabopUpdateRuleProp;
        private SerializedProperty cabopEnableCostBudgetProp;
        private SerializedProperty cabopMaxCumulativeCostProp;
        private SerializedProperty cabopGroupCostsProp;
        private SerializedProperty iterationAdvanceModeProp;
        private SerializedProperty automaticAdvanceDelaySecProp;
        private SerializedProperty reloadSceneOnIterationAdvanceProp;
        private SerializedProperty enableFinalDesignRoundProp;
        private SerializedProperty finalDesignDistanceEpsilonProp;
        private SerializedProperty finalDesignMaximinEpsilonProp;
        private SerializedProperty finalDesignAggressionEpsilonProp;

        private SerializedProperty totalIterationsProp;

        private SerializedProperty userIdProp;
        private SerializedProperty conditionIdProp;
        private SerializedProperty groupIdProp;
        private SerializedProperty enablePriorSliderRatingHintProp;
        private SerializedProperty priorSliderRatingHintAlphaProp;
        
        private ReorderableList parameterList;
        private ReorderableList objectiveList;

        private string initDataPath;
        private static readonly string[] WarmStartObjectiveFormatOptions =
            { "auto", "raw", "normalized_max", "normalized_native" };

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
            localPythonProp = serializedObject.FindProperty("localPython");
            pythonPathProp = serializedObject.FindProperty("pythonPath");
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
            warmStartObjectiveFormatProp = serializedObject.FindProperty("warmStartObjectiveFormat");
            optimizerBackendProp = serializedObject.FindProperty("optimizerBackend");
            cabopObjectiveModeProp = serializedObject.FindProperty("cabopObjectiveMode");
            cabopUseCostAwareAcquisitionProp = serializedObject.FindProperty("cabopUseCostAwareAcquisition");
            cabopUpdateRuleProp = serializedObject.FindProperty("cabopUpdateRule");
            cabopEnableCostBudgetProp = serializedObject.FindProperty("cabopEnableCostBudget");
            cabopMaxCumulativeCostProp = serializedObject.FindProperty("cabopMaxCumulativeCost");
            cabopGroupCostsProp = serializedObject.FindProperty("cabopGroupCosts");
            iterationAdvanceModeProp = serializedObject.FindProperty("iterationAdvanceMode");
            automaticAdvanceDelaySecProp = serializedObject.FindProperty("automaticAdvanceDelaySec");
            reloadSceneOnIterationAdvanceProp = serializedObject.FindProperty("reloadSceneOnIterationAdvance");
            enableFinalDesignRoundProp = serializedObject.FindProperty("enableFinalDesignRound");
            finalDesignDistanceEpsilonProp = serializedObject.FindProperty("finalDesignDistanceEpsilon");
            finalDesignMaximinEpsilonProp = serializedObject.FindProperty("finalDesignMaximinEpsilon");
            finalDesignAggressionEpsilonProp = serializedObject.FindProperty("finalDesignAggressionEpsilon");

            totalIterationsProp = serializedObject.FindProperty("totalIterations");
            
            userIdProp = serializedObject.FindProperty("userId");
            conditionIdProp = serializedObject.FindProperty("conditionId");
            groupIdProp = serializedObject.FindProperty("groupId");
            enablePriorSliderRatingHintProp = serializedObject.FindProperty("enablePriorSliderRatingHint");
            priorSliderRatingHintAlphaProp = serializedObject.FindProperty("priorSliderRatingHintAlpha");
            
            initDataPath = Path.Combine(Application.dataPath, "StreamingAssets", "BOData", "InitData");

            enableSamplingEditProp = serializedObject.FindProperty("enableSamplingEdit");
    }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(2));
            // Draw Parameter and Objective Lists first
            parameterList.DoLayoutList();
            EditorGUILayout.Space();
            objectiveList.DoLayoutList();

            DrawSettingsConfiguration();

            serializedObject.ApplyModifiedProperties();
        }

        private void DrawSettingsConfiguration()
        {
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));

            // ── Python Settings ─────────────────────────────────────────────────────
            EditorGUILayout.LabelField("Python Settings", EditorStyles.boldLabel);
            var localPyLabel = new GUIContent("Manually Installed Python",
                "Use a locally installed Python instead of the project’s installer.");
            EditorGUILayout.PropertyField(localPythonProp, localPyLabel);
            EditorGUILayout.Space();

            if (localPythonProp.boolValue)
            {
                EditorGUILayout.PropertyField(pythonPathProp, new GUIContent("Path of Python Executable:"));
                EditorGUILayout.LabelField(
                    "Ensure a valid path for your OS (Windows/macOS differ).",
                    EditorStyles.helpBox
                );
            }

            // ── Study Settings ──────────────────────────────────────────────────────
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Study Settings", EditorStyles.boldLabel);

            if (string.IsNullOrEmpty(userIdProp.stringValue)) userIdProp.stringValue = "-1";
            if (string.IsNullOrEmpty(conditionIdProp.stringValue)) conditionIdProp.stringValue = "-1";
            if (string.IsNullOrEmpty(groupIdProp.stringValue)) groupIdProp.stringValue = "-1";

            EditorGUILayout.PropertyField(userIdProp);
            EditorGUILayout.PropertyField(conditionIdProp);
            EditorGUILayout.PropertyField(groupIdProp);
            EditorGUILayout.LabelField("Default values for userID, conditionID and groupID are -1.", EditorStyles.helpBox);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Questionnaire Prior Rating Hint", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(
                enablePriorSliderRatingHintProp,
                new GUIContent(
                    "Show Prior Rating Hint",
                    "Show a subtle marker on questionnaire sliders and Likert (linear scale) items indicating the participant's rating from the previous completed iteration."
                )
            );
            if (enablePriorSliderRatingHintProp.boolValue)
            {
                EditorGUILayout.Slider(
                    priorSliderRatingHintAlphaProp,
                    0.05f,
                    0.45f,
                    new GUIContent(
                        "Hint Opacity",
                        "Lower opacity reduces bias while still making the prior rating visible."
                    )
                );
                EditorGUILayout.LabelField(
                    "The hint is visual-only. It does not prefill slider/Likert answers and does not count as an answer.",
                    EditorStyles.helpBox
                );
            }

            // ── Problem Setup (not hyperparameters) ─────────────────────────────────
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Problem Setup", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Design Parameters (d)", parameterList.count.ToString(), EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Design Objectives (m)", objectiveList.count.ToString(), EditorStyles.boldLabel);

            // ── Optimizer Backend / CABOP ──────────────────────────────────────────
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Optimizer Backend", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(
                optimizerBackendProp,
                new GUIContent(
                    "Backend",
                    "Choose BoTorch (bo.py/mobo.py) or CABOP (cost-aware backend)."
                )
            );

            bool useCabop = (BoForUnityManager.OptimizerBackend)optimizerBackendProp.enumValueIndex ==
                            BoForUnityManager.OptimizerBackend.CABOP;
            if (useCabop)
            {
                EditorGUILayout.PropertyField(
                    cabopObjectiveModeProp,
                    new GUIContent(
                        "CABOP Objective Mode",
                        "SingleObjective requires exactly one objective. MultiObjectiveScalarized requires two or more objectives."
                    )
                );
                EditorGUILayout.PropertyField(
                    cabopUseCostAwareAcquisitionProp,
                    new GUIContent("CABOP Use Cost Aware Acquisition")
                );
                EditorGUILayout.PropertyField(
                    cabopUpdateRuleProp,
                    new GUIContent("CABOP Update Rule")
                );
                EditorGUILayout.PropertyField(
                    cabopEnableCostBudgetProp,
                    new GUIContent(
                        "CABOP Enable Cost Budget",
                        "If enabled, CABOP stops when cumulative evaluation cost reaches the configured limit."
                    )
                );
                if (cabopEnableCostBudgetProp.boolValue)
                {
                    EditorGUILayout.PropertyField(
                        cabopMaxCumulativeCostProp,
                        new GUIContent("CABOP Max Cumulative Cost")
                    );
                }

                EditorGUILayout.PropertyField(
                    cabopGroupCostsProp,
                    new GUIContent(
                        "CABOP Group Costs",
                        "Per-group unchanged/swapped/acquired costs for model and realized execution costs."
                    ),
                    true
                );

                int objectiveCount = objectiveList.count;
                int mode = cabopObjectiveModeProp.enumValueIndex;
                bool singleInvalid = mode == (int)BoForUnityManager.CabopObjectiveMode.SingleObjective && objectiveCount != 1;
                bool multiInvalid = mode == (int)BoForUnityManager.CabopObjectiveMode.MultiObjectiveScalarized && objectiveCount < 2;
                if (singleInvalid || multiInvalid)
                {
                    EditorGUILayout.HelpBox(
                        "CABOP mode/objective count mismatch. " +
                        "SingleObjective needs exactly 1 objective; MultiObjectiveScalarized needs at least 2.",
                        MessageType.Warning
                    );
                }
            }

            // ── Optimization Budget (iterations & termination) ──────────────────────
            EditorGUILayout.Space();
            GUILayout.Box(GUIContent.none, GUILayout.ExpandWidth(true), GUILayout.Height(3));
            EditorGUILayout.LabelField("Optimization Budget", EditorStyles.boldLabel);

            // Warm start & perfect rating belong to run control / termination
            EditorGUILayout.PropertyField(warmStartProp, new GUIContent("Warm Start",
                "Skip sampling by loading initial data. The saved sampling iteration setting is preserved."));
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
            }

            if (warmStartObjectiveFormatProp != null)
            {
                string format = (warmStartObjectiveFormatProp.stringValue ?? "auto").Trim().ToLowerInvariant();
                int idx = System.Array.IndexOf(WarmStartObjectiveFormatOptions, format);
                if (idx < 0) idx = 0;

                using (new EditorGUI.DisabledScope(!warmStartProp.boolValue))
                {
                    int selected = EditorGUILayout.Popup(
                        new GUIContent(
                            "Warm Start Objective Format",
                            "How warm-start objective CSV values are interpreted: auto/raw/normalized_max/normalized_native."
                        ),
                        idx,
                        WarmStartObjectiveFormatOptions
                    );
                    warmStartObjectiveFormatProp.stringValue = WarmStartObjectiveFormatOptions[selected];
                }
            }

            // Sampling iterations: default 2(d+1) unless user explicitly enables manual edit
            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(warmStartProp.boolValue))
            {
                EditorGUILayout.PropertyField(
                    enableSamplingEditProp,
                    new GUIContent("Set Sampling Iterations Manually",
                        "Default is 2·(d+1). Enable to override.")
                );

                int defaultSampling = BoForUnityManager.ComputeRecommendedSamplingIterations(parameterList.count);

                // When manual edit is OFF, keep the default and lock the field.
                if (!warmStartProp.boolValue && !enableSamplingEditProp.boolValue)
                {
                    if (nSamplingIterProp.intValue != defaultSampling)
                        nSamplingIterProp.intValue = defaultSampling;
                }

                if (!enableSamplingEditProp.boolValue)
                {
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
                    "Recommended sampling iterations default: 2 · (d + 1), where d is the number of design parameters.",
                    EditorStyles.helpBox
                );
            }

            EditorGUILayout.PropertyField(nOptimizationIterProp, new GUIContent("Optimization Iterations"));

            // Total iterations = sampling (or 0 with warm start) + optimization
            int sampling = warmStartProp.boolValue ? 0 : Mathf.Max(0, nSamplingIterProp.intValue);
            int optimization = Mathf.Max(0, nOptimizationIterProp.intValue);
            int total = sampling + optimization;
            totalIterationsProp.intValue = total;

            EditorGUILayout.LabelField("Total Iterations", total.ToString(), EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                "Total = effective Sampling Iterations + Optimization Iterations. Warm start uses 0 sampling iterations without changing the saved sampling setting.",
                EditorStyles.helpBox
            );

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Iteration Progression", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(
                iterationAdvanceModeProp,
                new GUIContent(
                    "Iteration Advance Mode",
                    "Choose how the next iteration starts: Next Button, External Signal (call RequestNextIteration), or Automatic."
                )
            );
            if ((BoForUnityManager.IterationAdvanceMode)iterationAdvanceModeProp.enumValueIndex ==
                BoForUnityManager.IterationAdvanceMode.Automatic)
            {
                EditorGUILayout.PropertyField(
                    automaticAdvanceDelaySecProp,
                    new GUIContent("Automatic Advance Delay (s)", "Delay before auto-starting the next iteration.")
                );
            }
            EditorGUILayout.PropertyField(
                reloadSceneOnIterationAdvanceProp,
                new GUIContent(
                    "Reload Scene On Advance",
                    "If disabled, the manager will not reload the active scene when advancing to the next iteration."
                )
            );

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Final Design Round", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(
                enableFinalDesignRoundProp,
                new GUIContent(
                    "Enable Final Design Round",
                    "After optimization ends, select one final design from the observation CSV and run one last evaluation round without BO updates."
                )
            );
            if (enableFinalDesignRoundProp.boolValue)
            {
                EditorGUILayout.PropertyField(
                    finalDesignDistanceEpsilonProp,
                    new GUIContent("Utopia Distance Epsilon", "Tie tolerance for closest-to-utopia selection.")
                );
                EditorGUILayout.PropertyField(
                    finalDesignMaximinEpsilonProp,
                    new GUIContent("Maximin Epsilon", "Tie tolerance for maximin tie-break.")
                );
                EditorGUILayout.PropertyField(
                    finalDesignAggressionEpsilonProp,
                    new GUIContent("Aggression Epsilon", "Tie tolerance for least-aggressive parameter tie-break.")
                );
                EditorGUILayout.LabelField(
                    "This adds one extra participant-facing round (totalIterations + 1). " +
                    "The final round does not send objectives back to Python.",
                    EditorStyles.helpBox
                );
            }

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
            if ((BoForUnityManager.IterationAdvanceMode)iterationAdvanceModeProp.enumValueIndex ==
                BoForUnityManager.IterationAdvanceMode.NextButton)
            {
                EditorGUILayout.PropertyField(nextButtonProp, new GUIContent("Next Button"));
            }
            else
            {
                EditorGUILayout.PropertyField(nextButtonProp, new GUIContent("Next Button (Optional)"));
            }
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
            SerializedProperty cabopGroup = value.FindPropertyRelative("cabopGroup");
            SerializedProperty cabopTolerance = value.FindPropertyRelative("cabopTolerance");
            SerializedProperty cabopPrefabricatedValues = value.FindPropertyRelative("cabopPrefabricatedValues");
            bool showCabopFields = optimizerBackendProp != null &&
                                   (BoForUnityManager.OptimizerBackend)optimizerBackendProp.enumValueIndex ==
                                   BoForUnityManager.OptimizerBackend.CABOP;

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
                yOffset += fieldHeight;
                if (showCabopFields)
                {
                    EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), cabopGroup);
                    yOffset += fieldHeight;
                    EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), cabopTolerance);
                    yOffset += fieldHeight;
                    float prefabHeight = EditorGUI.GetPropertyHeight(cabopPrefabricatedValues, true);
                    EditorGUI.PropertyField(
                        new Rect(rect.x, yOffset, rect.width, prefabHeight),
                        cabopPrefabricatedValues,
                        new GUIContent("CABOP Prefabricated Values"),
                        true
                    );
                }

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
                totalHeight += singleLineHeight * 3 + 1.5f * 3;
                bool showCabopFields = optimizerBackendProp != null &&
                                       (BoForUnityManager.OptimizerBackend)optimizerBackendProp.enumValueIndex ==
                                       BoForUnityManager.OptimizerBackend.CABOP;
                if (showCabopFields)
                {
                    SerializedProperty cabopPrefabricatedValues = value.FindPropertyRelative("cabopPrefabricatedValues");
                    totalHeight += singleLineHeight * 2 + 1.5f * 2;
                    totalHeight += EditorGUI.GetPropertyHeight(cabopPrefabricatedValues, true) + 1.5f;
                }
            }

            return totalHeight;
        }



        private void DrawObjectiveListItems(Rect rect, int index, bool isActive, bool isFocused)
        {
            SerializedProperty element = objectiveList.serializedProperty.GetArrayElementAtIndex(index);
            SerializedProperty key = element.FindPropertyRelative("key");
            SerializedProperty value = element.FindPropertyRelative("value");
            SerializedProperty numberOfSubMeasures = value.FindPropertyRelative("numberOfSubMeasures");
            SerializedProperty values = value.FindPropertyRelative("values");
            SerializedProperty lowerBound = value.FindPropertyRelative("lowerBound");
            SerializedProperty upperBound = value.FindPropertyRelative("upperBound");
            SerializedProperty smallerIsBetter = value.FindPropertyRelative("smallerIsBetter");
            SerializedProperty cabopWeight = value.FindPropertyRelative("cabopWeight");
            bool showCabopFields = IsCabopBackendSelected();

            float padding = 5f;
            float singleLineHeight = EditorGUIUtility.singleLineHeight;
            float spacing = EditorGUIUtility.standardVerticalSpacing;
            rect.y += padding / 2;
            float yOffset = rect.y;

            EditorGUI.PropertyField(new Rect(rect.x, yOffset, rect.width, singleLineHeight), key, GUIContent.none);
            yOffset += singleLineHeight + spacing;

            value.isExpanded = EditorGUI.Foldout(
                new Rect(rect.x, yOffset, rect.width, singleLineHeight),
                value.isExpanded,
                "",
                true
            );
            yOffset += singleLineHeight + spacing;

            if (!value.isExpanded)
                return;

            EditorGUI.indentLevel++;
            DrawObjectiveChildProperty(rect, ref yOffset, numberOfSubMeasures);
            DrawObjectiveChildProperty(rect, ref yOffset, values, includeChildren: true);
            DrawObjectiveChildProperty(rect, ref yOffset, lowerBound);
            DrawObjectiveChildProperty(rect, ref yOffset, upperBound);
            DrawObjectiveChildProperty(rect, ref yOffset, smallerIsBetter);
            if (showCabopFields)
            {
                DrawObjectiveChildProperty(rect, ref yOffset, cabopWeight);
            }
            EditorGUI.indentLevel--;
        }

        private static void DrawObjectiveChildProperty(
            Rect rect,
            ref float yOffset,
            SerializedProperty property,
            bool includeChildren = false)
        {
            float propertyHeight = EditorGUI.GetPropertyHeight(property, includeChildren);
            EditorGUI.PropertyField(
                new Rect(rect.x, yOffset, rect.width, propertyHeight),
                property,
                includeChildren
            );
            yOffset += propertyHeight + EditorGUIUtility.standardVerticalSpacing;
        }

        private bool IsCabopBackendSelected()
        {
            return optimizerBackendProp != null &&
                   (BoForUnityManager.OptimizerBackend)optimizerBackendProp.enumValueIndex ==
                   BoForUnityManager.OptimizerBackend.CABOP;
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
            SerializedProperty value = element.FindPropertyRelative("value");
            float padding = 5f;
            float spacing = EditorGUIUtility.standardVerticalSpacing;
            float totalHeight = padding + (EditorGUIUtility.singleLineHeight + spacing) * 2;

            if (value.isExpanded)
            {
                totalHeight += GetObjectiveChildPropertyHeight(value, "numberOfSubMeasures") + spacing;
                totalHeight += GetObjectiveChildPropertyHeight(value, "values", includeChildren: true) + spacing;
                totalHeight += GetObjectiveChildPropertyHeight(value, "lowerBound") + spacing;
                totalHeight += GetObjectiveChildPropertyHeight(value, "upperBound") + spacing;
                totalHeight += GetObjectiveChildPropertyHeight(value, "smallerIsBetter") + spacing;

                if (IsCabopBackendSelected())
                {
                    totalHeight += GetObjectiveChildPropertyHeight(value, "cabopWeight") + spacing;
                }
            }

            return totalHeight;
        }

        private static float GetObjectiveChildPropertyHeight(
            SerializedProperty value,
            string propertyName,
            bool includeChildren = false)
        {
            return EditorGUI.GetPropertyHeight(value.FindPropertyRelative(propertyName), includeChildren);
        }
    }
}
