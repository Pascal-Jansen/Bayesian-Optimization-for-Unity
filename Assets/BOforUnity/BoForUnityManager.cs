using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using BOforUnity.Scripts;
using QuestionnaireToolkit.Scripts;
using TMPro;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.Assertions.Must;
using UnityEngine.SceneManagement;
using UnityEngine.Serialization;
using UnityEngine.UI;
using PythonStarter = BOforUnity.Scripts.PythonStarter;

namespace BOforUnity
{
    public class BoForUnityManager : MonoBehaviour
    {
        public enum IterationAdvanceMode
        {
            NextButton = 0,
            ExternalSignal = 1,
            Automatic = 2
        }

        public PythonStarter pythonStarter;
        public Optimizer optimizer;
        public MainThreadDispatcher mainThreadDispatcher;
        public SocketNetwork socketNetwork;

        private static BoForUnityManager _instance;
        
        //-----------------------------------------------
        // DESIGN PARAMETERS and DESIGN OBJECTIVES
        public List<ParameterEntry> parameters = new List<ParameterEntry>();
        public List<ObjectiveEntry> objectives = new List<ObjectiveEntry>();
        //-----------------------------------------------
        
        //-----------------------------------------------
        // ITERATION CONTROLLER
        [SerializeField]
        public int currentIteration;  // Current iteration value.
        public int totalIterations;
        public bool perfectRating;   // Flag indicating perfect rating.
        public bool perfectRatingStart;  // Flag indicating the start of perfect rating.
        public int perfectRatingIteration;
        public bool initialized = false;
        public bool simulationRunning = false;
        private bool _waitingForPythonProcess = false;
        
        // BO Hyper-parameters
        public int batchSize = 1;
        public int numRestarts = 10;
        public int rawSamples = 1024;
        public int mcSamples = 512;
        public int numSamplingIterations = 5; // in typical MOBO problems, this should be 2(d+1), where d is the number of objectives
        public int numOptimizationIterations = 10;
        public int seed = 3;
        
        [SerializeField] private bool enableSamplingEdit = false; // checkbox in inspector
        
        public bool warmStart = false;
        public bool perfectRatingActive = false;
        public bool perfectRatingInInitialRounds = false;
        public string initialParametersDataPath;
        public string initialObjectivesDataPath;
        public string warmStartObjectiveFormat = "auto";

        [Header("Loop Progression")]
        public IterationAdvanceMode iterationAdvanceMode = IterationAdvanceMode.NextButton;
        [Min(0f)] public float automaticAdvanceDelaySec = 0f;
        public bool reloadSceneOnIterationAdvance = true;

        [Header("Final Design Round")]
        public bool enableFinalDesignRound = false;
        [Min(0f)] public float finalDesignDistanceEpsilon = 1e-6f;
        [Min(0f)] public float finalDesignMaximinEpsilon = 1e-6f;
        [Min(0f)] public float finalDesignAggressionEpsilon = 1e-6f;

        public string userId = "-1";
        public string conditionId = "-1";
        public string groupId = "-1";

        public bool hasNewDesignParameterValues;
        private bool _pendingAdvanceRequest = false;
        private bool _loopTerminated = false;
        private Coroutine _automaticAdvanceCoroutine = null;
        private bool _warnedMissingNextButton = false;
        private bool _finalDesignRoundPrepared = false;
        private bool _finalDesignRoundInProgress = false;
        private bool _finalDesignRoundLogged = false;
        private string _finalDesignObservationCsvPath = null;
        //-----------------------------------------------
        
        //-----------------------------------------------
        private void Awake()
        {
            // If there is already an instance of this object, destroy the new one
            if (_instance != null)
            {
                Destroy(gameObject);
                return;
            }
            // Mark this object as the single instance and make it persistent
            _instance = this;
            DontDestroyOnLoad(gameObject);
            
            pythonStarter = gameObject.GetComponent<PythonStarter>();
            optimizer = gameObject.GetComponent<Optimizer>();
            mainThreadDispatcher = gameObject.GetComponent<MainThreadDispatcher>();
            socketNetwork = gameObject.GetComponent<SocketNetwork>();

            currentIteration = 1;
            totalIterations = numSamplingIterations + numOptimizationIterations; // set how many iterations the optimizer should run for
        }
        
        void Start()
        {
            SetLoadingVisible(true);
            SetNextButtonVisible(false);

            initialized = false;
            _waitingForPythonProcess = true;
            perfectRating = false;
            perfectRatingStart = false;
            optimizationFinished = false;
            _pendingAdvanceRequest = false;
            _loopTerminated = false;
            _warnedMissingNextButton = false;
            _finalDesignRoundPrepared = false;
            _finalDesignRoundInProgress = false;
            _finalDesignRoundLogged = false;
            _finalDesignObservationCsvPath = null;
            simulationRunning = true; // the simulation to true to prevent 
        }
        
        void Update()
        {
            if (_waitingForPythonProcess && pythonStarter != null && pythonStarter.isPythonProcessRunning && pythonStarter.isSystemStarted)
            {
                _waitingForPythonProcess = false;
                PythonInitializationDone();
            }
        }
        //-----------------------------------------------
        
        
        // CONTROLLER SCENE
        //-----------------------------------------------
        public TMP_Text outputText;
        public GameObject loadingObj;
        public GameObject nextButton;

        public GameObject welcomePanel;
        public GameObject optimizerStatePanel;
        
        public bool optimizationRunning = false;
        public bool optimizationFinished = false;

        // Legacy UI hook; use RequestNextIteration() for non-button flows.
        public void ButtonNextIteration()
        {
            RequestNextIteration();
        }

        // Public API for external mechanisms (questionnaire callbacks, timers, custom UI, etc.)
        public void RequestNextIteration()
        {
            if (_loopTerminated)
                return;

            // In NextButton mode, never queue "early" requests while no fresh design is ready.
            // This prevents double-click/external duplicate events from auto-advancing the next cycle.
            if (iterationAdvanceMode == IterationAdvanceMode.NextButton && !hasNewDesignParameterValues)
                return;

            _pendingAdvanceRequest = true;
            TryConsumeAdvanceRequest();
        }
        
        public void OptimizationStart()
        {
            if (_loopTerminated)
            {
                Debug.LogWarning("OptimizationStart ignored because optimization loop is already finished.");
                return;
            }
            if (_finalDesignRoundInProgress)
            {
                if (!_finalDesignRoundLogged)
                {
                    if (TryAppendFinalDesignObservationRow(out string logError))
                    {
                        _finalDesignRoundLogged = true;
                    }
                    else
                    {
                        Debug.LogWarning($"Could not append finaldesign row to ObservationsPerEvaluation.csv: {logError}");
                    }
                }

                Debug.Log("Final design round completed. Exiting loop.");
                _finalDesignRoundInProgress = false;
                CompleteLoop();
                return;
            }
            if (optimizationFinished)
            {
                Debug.LogWarning("OptimizationStart ignored because optimization is already finished.");
                return;
            }
            if (optimizationRunning)
            {
                Debug.LogWarning("OptimizationStart ignored because optimization is already running.");
                return;
            }
            if (socketNetwork == null)
            {
                Debug.LogError("OptimizationStart failed because SocketNetwork is not assigned.");
                return;
            }

            Debug.Log("Optimization START");
            CancelAutomaticAdvance();
            if (iterationAdvanceMode == IterationAdvanceMode.NextButton)
            {
                // Defensive reset against stale requests from duplicate UI events.
                _pendingAdvanceRequest = false;
            }

            socketNetwork.SendObjectives(); // send the current objective values to the Python process
            hasNewDesignParameterValues = false; // the current design parameter values are obsolete
            optimizationRunning = true;
            simulationRunning = false;

            SetOptimizerStatePanelVisible(true); // show that the optimizer is running
            SetLoadingVisible(true);
            SetNextButtonVisible(false);
            SetOutputText("The system is loading, please wait ...");
        }
        
        public void OptimizationDone()
        {
            Debug.Log("Optimization DONE");
            currentIteration++; // increase iteration counter
            HandleParametersReady("The system has finished loading.\nYou can now proceed.");
        }
        
        public void InitializationDone()
        {
            Debug.Log("Initialization DONE");
            initialized = true;
            HandleParametersReady("The system has been started successfully!\nYou can now start the study.");
        }

        public void OnOptimizationFinishedFromBackend()
        {
            if (_loopTerminated)
                return;

            Debug.Log(">>>>>> Optimization finished!");
            optimizationFinished = true;
            if (!enableFinalDesignRound)
            {
                CompleteLoop();
                return;
            }

            if (!TryPrepareFinalDesignRound(out var selectionError))
            {
                Debug.LogWarning(
                    "Final design round is enabled, but no final design could be selected. " +
                    $"Falling back to normal completion. Reason: {selectionError}"
                );
                CompleteLoop();
                return;
            }

            HandleParametersReady(
                "Optimization has finished.\nThe selected final design is ready for one last evaluation round."
            );
        }
        
        private void PythonInitializationDone()
        {
            Debug.Log("Python Process Initialization DONE");
            // Initialize the optimizer and socket connection ... only for Debug
            // optimizer.DebugOptimizer();
            // Start Optimization to receive the initialized parameter values for the first iteration
            if (socketNetwork == null)
            {
                Debug.LogError("PythonInitializationDone failed because SocketNetwork is not assigned.");
                return;
            }
            socketNetwork.InitSocket();
        }

        private void HandleParametersReady(string statusText)
        {
            if (_loopTerminated)
                return;

            hasNewDesignParameterValues = true;
            optimizationRunning = false;
            simulationRunning = false;

            // Keep the status panel visible only when the ready-state UI lives inside it.
            SetOptimizerStatePanelVisible(RequiresOptimizerPanelForReadyStateUi());
            SetLoadingVisible(false);
            SetOutputText(statusText);

            switch (iterationAdvanceMode)
            {
                case IterationAdvanceMode.NextButton:
                    SetNextButtonVisible(true);
                    if (nextButton == null && !_warnedMissingNextButton)
                    {
                        _warnedMissingNextButton = true;
                        Debug.LogWarning(
                            "IterationAdvanceMode is set to NextButton, but no Next Button is assigned. " +
                            "Assign a button, switch mode, or call RequestNextIteration() from your own logic."
                        );
                    }
                    break;
                case IterationAdvanceMode.ExternalSignal:
                    SetNextButtonVisible(false);
                    break;
                case IterationAdvanceMode.Automatic:
                    SetNextButtonVisible(false);
                    ScheduleAutomaticAdvance();
                    break;
            }

            // External-signal mode can otherwise stall before the very first evaluation
            // when no UI button exists. Auto-kick only the initial transition.
            if (iterationAdvanceMode == IterationAdvanceMode.ExternalSignal && currentIteration == 1)
            {
                _pendingAdvanceRequest = true;
            }

            // If an external signal was sent early while Python was still computing, honor it now.
            TryConsumeAdvanceRequest();
        }

        private void TryConsumeAdvanceRequest()
        {
            if (_loopTerminated || !_pendingAdvanceRequest || !hasNewDesignParameterValues)
                return;

            _pendingAdvanceRequest = false;
            AdvanceToNextIterationOrFinish();
        }

        private void AdvanceToNextIterationOrFinish()
        {
            SetLoadingVisible(true); // show loading while transitioning to next evaluation
            SetNextButtonVisible(false);
            // Lock progression until new parameters arrive from the backend.
            hasNewDesignParameterValues = false;

            if (currentIteration == 0)
            {
                SetWelcomePanelVisible(false);
                SetOptimizerStatePanelVisible(false);
                SetLoadingVisible(false);
                return;
            }

            if (_finalDesignRoundPrepared)
            {
                _finalDesignRoundPrepared = false;
                _finalDesignRoundInProgress = true;

                Debug.Log("--------------------------------------Current Iteration (Final Design Round): " + currentIteration);
                simulationRunning = true;
                SetOptimizerStatePanelVisible(false);

                if (reloadSceneOnIterationAdvance)
                {
                    SceneManager.LoadScene(SceneManager.GetActiveScene().name);
                }
                else
                {
                    SetLoadingVisible(false);
                }
                return;
            }

            bool isPerfect = ShouldStopForPerfectRating();
            if (isPerfect)
            {
                Debug.Log(">>>>> Perfect Rating");
            }

            if (optimizationFinished || currentIteration > totalIterations || isPerfect)
            {
                CompleteLoop();
                return;
            }

            // hide the panel as the next iteration starts after scene transition
            SetOptimizerStatePanelVisible(false);

            Debug.Log("--------------------------------------Current Iteration: " + currentIteration);

            simulationRunning = true; // waiting for the simulation to finish

            if (reloadSceneOnIterationAdvance)
            {
                SceneManager.LoadScene(SceneManager.GetActiveScene().name); // reload scene
            }
            else
            {
                SetLoadingVisible(false);
            }
        }

        private bool ShouldStopForPerfectRating()
        {
            if (!perfectRatingActive)
                return false;
            if (!perfectRatingInInitialRounds && currentIteration <= numSamplingIterations)
                return false;
            return IsPerfectRating();
        }

        private void ScheduleAutomaticAdvance()
        {
            CancelAutomaticAdvance();
            _automaticAdvanceCoroutine = StartCoroutine(AutomaticAdvanceRoutine());
        }

        private void CancelAutomaticAdvance()
        {
            if (_automaticAdvanceCoroutine == null)
                return;

            StopCoroutine(_automaticAdvanceCoroutine);
            _automaticAdvanceCoroutine = null;
        }

        private System.Collections.IEnumerator AutomaticAdvanceRoutine()
        {
            if (automaticAdvanceDelaySec > 0f)
            {
                yield return new WaitForSeconds(automaticAdvanceDelaySec);
            }

            _automaticAdvanceCoroutine = null;
            RequestNextIteration();
        }

        private void CompleteLoop()
        {
            if (_loopTerminated)
                return;

            _loopTerminated = true;
            _pendingAdvanceRequest = false;
            _finalDesignRoundPrepared = false;
            _finalDesignRoundInProgress = false;
            _finalDesignRoundLogged = false;
            _finalDesignObservationCsvPath = null;
            CancelAutomaticAdvance();

            Debug.Log("<<<<<<< Exiting the loop ... ");
            Debug.Log("------------------------------------------------");

            simulationRunning = false;
            optimizationRunning = false;

            SetOptimizerStatePanelVisible(false);
            SetLoadingVisible(false);
            SetNextButtonVisible(false);
            SetOutputText("The simulation has finished!\nYou can now close the application.");

            try
            {
                socketNetwork?.SocketQuit();
            }
            catch (Exception e)
            {
                Debug.LogWarning($"SocketQuit failed during loop termination: {e.Message}");
            }
        }

        private bool RequiresOptimizerPanelForReadyStateUi()
        {
            if (optimizerStatePanel == null)
                return false;

            var panelTransform = optimizerStatePanel.transform;
            if (outputText != null && outputText.transform.IsChildOf(panelTransform))
                return true;

            if (iterationAdvanceMode == IterationAdvanceMode.NextButton &&
                nextButton != null &&
                nextButton.transform.IsChildOf(panelTransform))
            {
                return true;
            }

            return false;
        }

        private void SetLoadingVisible(bool visible)
        {
            if (loadingObj != null)
                loadingObj.SetActive(visible);
        }

        private void SetNextButtonVisible(bool visible)
        {
            if (nextButton != null)
                nextButton.SetActive(visible);
        }

        private void SetOptimizerStatePanelVisible(bool visible)
        {
            if (optimizerStatePanel != null)
                optimizerStatePanel.SetActive(visible);
        }

        private void SetWelcomePanelVisible(bool visible)
        {
            if (welcomePanel != null)
                welcomePanel.SetActive(visible);
        }

        private void SetOutputText(string value)
        {
            if (outputText != null)
                outputText.text = value;
        }

        private bool TryPrepareFinalDesignRound(out string error)
        {
            _finalDesignRoundPrepared = false;
            _finalDesignRoundInProgress = false;
            _finalDesignRoundLogged = false;
            _finalDesignObservationCsvPath = null;

            FinalDesignSelector.SelectionResult selected = null;
            string selectedCsvPath = null;
            string selectionError = null;

            string[] logRootCandidates = GetFinalDesignLogRootCandidates();
            bool selectedFromAnyRoot = false;
            foreach (string logRoot in logRootCandidates)
            {
                if (FinalDesignSelector.TrySelectFromLatestObservationCsv(
                        logRootPath: logRoot,
                        userId: userId,
                        parameters: parameters,
                        objectives: objectives,
                        distanceEpsilon: finalDesignDistanceEpsilon,
                        maximinEpsilon: finalDesignMaximinEpsilon,
                        aggressionEpsilon: finalDesignAggressionEpsilon,
                        selection: out selected,
                        selectedCsvPath: out selectedCsvPath,
                        error: out selectionError))
                {
                    selectedFromAnyRoot = true;
                    if (!string.Equals(logRoot, logRootCandidates[0], StringComparison.Ordinal))
                    {
                        Debug.LogWarning(
                            "Final design selector used fallback log root: " +
                            $"{logRoot}. Primary path was: {logRootCandidates[0]}"
                        );
                    }
                    break;
                }
            }

            if (!selectedFromAnyRoot)
            {
                error =
                    "No eligible observation log was found in any known log root. " +
                    "Last error: " + selectionError + ". " +
                    "Checked roots: " + string.Join(", ", logRootCandidates);
                return false;
            }

            if (selected.ParameterRaw.Length != parameters.Count)
            {
                error = "Selected final-design parameter count does not match current parameter list.";
                return false;
            }

            for (int i = 0; i < parameters.Count; i++)
            {
                float selectedValue = selected.ParameterRaw[i];
                if (float.IsNaN(selectedValue) || float.IsInfinity(selectedValue))
                {
                    error = $"Selected parameter '{parameters[i].key}' is non-finite.";
                    return false;
                }

                float lo = parameters[i].value.lowerBound;
                float hi = parameters[i].value.upperBound;
                float eps = 1e-4f;
                if (selectedValue < lo - eps || selectedValue > hi + eps)
                {
                    error = $"Selected parameter '{parameters[i].key}'={selectedValue} is outside bounds [{lo}, {hi}].";
                    return false;
                }

                parameters[i].value.Value = Mathf.Clamp(selectedValue, lo, hi);
            }

            currentIteration = totalIterations + 1;
            _finalDesignRoundPrepared = true;
            _finalDesignRoundInProgress = false;
            _finalDesignRoundLogged = false;
            _finalDesignObservationCsvPath = selectedCsvPath;

            Debug.Log(
                "Selected final design for last evaluation round: " +
                $"iteration={selected.Iteration}, utopiaDist={selected.UtopiaDistance}, " +
                $"maximin={selected.Maximin}, aggression={selected.Aggression}, csv={selectedCsvPath}"
            );

            error = null;
            return true;
        }

        private bool TryAppendFinalDesignObservationRow(out string error)
        {
            error = null;

            if (string.IsNullOrWhiteSpace(_finalDesignObservationCsvPath))
            {
                error = "Final-design CSV path is empty.";
                return false;
            }
            if (!File.Exists(_finalDesignObservationCsvPath))
            {
                error = $"Final-design CSV does not exist: {_finalDesignObservationCsvPath}";
                return false;
            }

            string headerLine;
            try
            {
                headerLine = File.ReadLines(_finalDesignObservationCsvPath).FirstOrDefault();
            }
            catch (Exception ex)
            {
                error = $"Could not read CSV header: {ex.Message}";
                return false;
            }

            if (string.IsNullOrWhiteSpace(headerLine))
            {
                error = "Observation CSV has no header.";
                return false;
            }

            string[] header = headerLine.Split(';');
            if (header.Length == 0)
            {
                error = "Observation CSV header is empty.";
                return false;
            }

            header[0] = header[0].Trim('\uFEFF');
            var columnIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < header.Length; i++)
            {
                string key = (header[i] ?? string.Empty).Trim();
                if (!string.IsNullOrEmpty(key) && !columnIndex.ContainsKey(key))
                    columnIndex[key] = i;
            }

            var row = Enumerable.Repeat(string.Empty, header.Length).ToArray();

            void SetColumn(string columnName, string value)
            {
                if (columnIndex.TryGetValue(columnName, out int idx))
                    row[idx] = value ?? string.Empty;
            }

            SetColumn("UserID", userId);
            SetColumn("ConditionID", conditionId);
            SetColumn("GroupID", groupId);
            SetColumn("Timestamp", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
            SetColumn("Iteration", currentIteration.ToString(CultureInfo.InvariantCulture));
            SetColumn("Phase", "finaldesign");
            SetColumn("IsPareto", "NULL");
            SetColumn("IsBest", "NULL");

            foreach (var objective in objectives)
            {
                if (objective == null || string.IsNullOrWhiteSpace(objective.key))
                    continue;

                float objectiveValue = GetObjectiveAverageForLogging(objective);
                SetColumn(objective.key, FormatCsvFloat(objectiveValue));
            }

            foreach (var parameter in parameters)
            {
                if (parameter == null || string.IsNullOrWhiteSpace(parameter.key) || parameter.value == null)
                    continue;

                float rawValue = parameter.value.Value;
                if (!IsFinite(rawValue))
                {
                    float fallback = 0.5f * (parameter.value.lowerBound + parameter.value.upperBound);
                    Debug.LogWarning(
                        $"Parameter '{parameter.key}' is non-finite ({rawValue}) during finaldesign logging. " +
                        $"Using midpoint fallback {fallback}."
                    );
                    rawValue = fallback;
                }

                float lo = Mathf.Min(parameter.value.lowerBound, parameter.value.upperBound);
                float hi = Mathf.Max(parameter.value.lowerBound, parameter.value.upperBound);
                if (rawValue < lo || rawValue > hi)
                {
                    float unclamped = rawValue;
                    rawValue = Mathf.Clamp(rawValue, lo, hi);
                    Debug.LogWarning(
                        $"Parameter '{parameter.key}' value {unclamped} is outside bounds [{lo}, {hi}] " +
                        $"during finaldesign logging. Clamped to {rawValue}."
                    );
                }

                SetColumn(parameter.key, FormatCsvFloat(rawValue));
            }

            string rowLine = string.Join(";", row.Select(EscapeCsvCell));
            string prefix = string.Empty;
            try
            {
                using (var fs = new FileStream(_finalDesignObservationCsvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                {
                    if (fs.Length > 0)
                    {
                        fs.Seek(-1, SeekOrigin.End);
                        int last = fs.ReadByte();
                        if (last != '\n' && last != '\r')
                            prefix = Environment.NewLine;
                    }
                }
            }
            catch (Exception ex)
            {
                error = $"Could not inspect CSV newline state: {ex.Message}";
                return false;
            }

            try
            {
                File.AppendAllText(_finalDesignObservationCsvPath, prefix + rowLine + Environment.NewLine);
            }
            catch (Exception ex)
            {
                error = $"Could not append finaldesign row: {ex.Message}";
                return false;
            }

            Debug.Log(
                $"Final design evaluation appended to observations CSV (phase=finaldesign, marker=NULL): {_finalDesignObservationCsvPath}"
            );
            return true;
        }

        private float GetObjectiveAverageForLogging(ObjectiveEntry objective)
        {
            if (objective == null || objective.value == null)
                return 0f;

            var arg = objective.value;
            float lo = Mathf.Min(arg.lowerBound, arg.upperBound);
            float hi = Mathf.Max(arg.lowerBound, arg.upperBound);

            if (arg.values == null || arg.values.Count == 0)
            {
                float fallback = 0.5f * (lo + hi);
                Debug.LogWarning(
                    $"Objective '{objective.key}' has no values during finaldesign logging. " +
                    $"Using midpoint fallback {fallback}."
                );
                return fallback;
            }

            int n = Mathf.Max(1, arg.numberOfSubMeasures);
            int start = Mathf.Max(0, arg.values.Count - n);
            int count = arg.values.Count - start;
            if (count <= 0)
            {
                float fallback = 0.5f * (lo + hi);
                Debug.LogWarning(
                    $"Objective '{objective.key}' has no usable sub-measures during finaldesign logging. " +
                    $"Using midpoint fallback {fallback}."
                );
                return fallback;
            }

            double sum = 0.0;
            for (int i = start; i < arg.values.Count; i++)
            {
                float v = arg.values[i];
                if (!IsFinite(v))
                {
                    float fallback = 0.5f * (lo + hi);
                    Debug.LogWarning(
                        $"Objective '{objective.key}' contains non-finite value ({v}) during finaldesign logging. " +
                        $"Using midpoint fallback {fallback}."
                    );
                    return fallback;
                }
                sum += v;
            }

            float avg = (float)(sum / count);
            if (!IsFinite(avg))
            {
                float fallback = 0.5f * (lo + hi);
                Debug.LogWarning(
                    $"Objective '{objective.key}' average is non-finite during finaldesign logging. " +
                    $"Using midpoint fallback {fallback}."
                );
                return fallback;
            }

            if (avg < lo || avg > hi)
            {
                float raw = avg;
                avg = Mathf.Clamp(avg, lo, hi);
                Debug.LogWarning(
                    $"Objective '{objective.key}' value {raw} is outside configured bounds [{lo}, {hi}] " +
                    $"during finaldesign logging. Clamped to {avg}."
                );
            }

            return avg;
        }

        private static string FormatCsvFloat(float value)
        {
            if (!IsFinite(value))
                return string.Empty;
            return Math.Round(value, 3, MidpointRounding.AwayFromZero).ToString("0.###", CultureInfo.InvariantCulture);
        }

        private static string EscapeCsvCell(string value)
        {
            if (string.IsNullOrEmpty(value))
                return string.Empty;

            bool mustQuote =
                value.IndexOf(';') >= 0 ||
                value.IndexOf('"') >= 0 ||
                value.IndexOf('\n') >= 0 ||
                value.IndexOf('\r') >= 0;

            if (!mustQuote)
                return value;

            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private string[] GetFinalDesignLogRootCandidates()
        {
            // Current runtime location used by Python process working directory.
            string current = Path.Combine(
                Application.dataPath,
                "StreamingAssets",
                "BOData",
                "LogData"
            );

            // Legacy location from earlier versions / docs.
            string legacy = Path.Combine(
                Application.dataPath,
                "StreamingAssets",
                "BOData",
                "BayesianOptimization",
                "LogData"
            );

            return new[] { current, legacy }.Distinct().ToArray();
        }
        
        private bool IsPerfectRating()
        {
            foreach (var ob in objectives)
            {
                if (ob.value.values.Count == 0)
                {
                    return false;
                }
                
                if (ob.value.smallerIsBetter ?
                        ob.value.values.Average() > ob.value.lowerBound : 
                        ob.value.values.Average() < ob.value.upperBound)
                {
                    return false; // the rating is imperfect!
                }
            }
            
            Debug.Log("Could be perfect rating ...");
            
            switch (perfectRatingStart)
            {
                case false:
                    perfectRatingStart = true;
                    perfectRating = false;
                    perfectRatingIteration = currentIteration; // remember the current iteration for this perfect rating
                    break;
                case true when currentIteration - perfectRatingIteration == 1:
                    Debug.Log("It is a perfect rating (i.e., perfect two times in a row)!");
                    perfectRatingStart = false;
                    perfectRating = true; // the rating was perfect after two consecutive iterations
                    return true;
                default:
                    perfectRatingStart = false; // the perfect rating was more than one iteration ago
                    perfectRating = false;
                    break;
            }
            return false;
        }
        
        public void EndApplication()
        {
#if UNITY_EDITOR
            EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
        }
        //-----------------------------------------------
        
        
        //--------------------------------------------
        [Header("Location of Python executable")]
        public bool localPython;
        public string pythonPath;

        public bool getLocalPython() { return localPython; }
        public void setLocalPython(bool a) { localPython = a; }
        
        public string getPythonPath() { return pythonPath; }
        public void setPythonPath(string newPath) { pythonPath = newPath; }
        
        //-----------------------------------------------
    }
    
            // ------------------
        // the objective entries:
        // ------------------
        [System.Serializable]
        public class ObjectiveEntry
        {
            public string key;
            public ObjectiveArgs value;
            public ObjectiveEntry(string key, ObjectiveArgs value)
            {
                this.key = key;
                this.value = value;
            }
        }
        
        [System.Serializable]
        public class ObjectiveArgs
        {
            /// <summary>
            /// optSeqOrder: an integer that represents the order of this objective in a sequence of objectives
            /// values: a List of floats that stores the values obtained for this objective in a sequence of trials.
            /// lowerBound: a float that represents the lower bound of the acceptable range of values for this objective.
            /// upperBound: a float that represents the upper bound of the acceptable range of values for this objective
            /// smallerIsBetter: a bool that specifies whether a smaller value is considered better for this objective.
            /// hasMultipleValues: a bool that specifies whether this objective should have multiple values.
            /// </summary>
            [HideInInspector] public int optSeqOrder;
            public int numberOfSubMeasures;
            public List<float> values = new List<float>();
            public float lowerBound = 0.0f;
            public float upperBound = 0.0f;
            public bool smallerIsBetter = false;

            /// <summary>
            /// ObjectiveArgs(): a constructor that creates an empty instance of the ObjectiveArgs class.
            /// </summary>
            public ObjectiveArgs() { }

            /// <summary>
            /// ObjectiveArgs(lowerBound, upperBound, smallerIsBetter): a constructor that creates an instance of the
            /// ObjectiveArgs class and sets the lower and upper bounds of the acceptable range of values, as well as the
            /// smallerIsBetter flag.
            /// </summary>
            /// <param name="lowerBound"></param>
            /// <param name="upperBound"></param>
            /// <param name="smallerIsBetter"></param>
            /// <param name="numberOfSubMeasures"></param>
            public ObjectiveArgs(float lowerBound, float upperBound, bool smallerIsBetter, int numberOfSubMeasures)
            {
                this.lowerBound = lowerBound;
                this.upperBound = upperBound;
                this.smallerIsBetter = smallerIsBetter;
                this.numberOfSubMeasures = numberOfSubMeasures;
            }
        }
        // ------------------
        
        
        // ------------------
        // the parameter entries:
        // ------------------
        [System.Serializable]
        public class ParameterEntry
        {
            public string key;
            public ParameterArgs value;
            public ParameterEntry(string key, ParameterArgs value)
            {
                this.key = key;
                this.value = value;
            }
        }
        
        [System.Serializable]
        public class ParameterArgs
        {
            /// <summary>
            /// optSeqOrder: an integer that represents the order of this parameter in a sequence of parameters.
            /// isDiscrete: a bool that specifies whether the parameter takes discrete (quantized) values.
            /// lowerBound: a float that represents the lower bound of the acceptable range of values for this parameter.
            /// upperBound: a float that represents the upper bound of the acceptable range of values for this parameter.
            /// step: a float that represents the increment between two consecutive values for a discrete parameter
            /// Value: a float that represents the current value of this parameter.
            /// reference: a float reference that can be used to keep track of the previous value of this parameter, if needed.
            /// </summary>
            [HideInInspector] public int optSeqOrder;
            public float lowerBound = 0.0f;
            public float upperBound = 0.0f;
            public float Value = 0.0f;

            /// <summary>
            /// ParameterArgs(): a constructor that creates an empty instance of the ParameterArgs class.
            /// </summary>
            public ParameterArgs() { }

            /// <summary>
            /// ParameterArgs(lowerBound, upperBound): a constructor that creates an instance of the ParameterArgs class and sets
            /// the lower and upper bounds of the acceptable range of values for this parameter.
            /// </summary>
            /// <param name="lowerBound"></param>
            /// <param name="upperBound"></param>
            public ParameterArgs(float lowerBound, float upperBound)
            {
                this.lowerBound = lowerBound;
                this.upperBound = upperBound;
            }
        }
}
