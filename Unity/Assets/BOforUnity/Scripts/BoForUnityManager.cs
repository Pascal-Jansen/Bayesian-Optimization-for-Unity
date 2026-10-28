using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using BOforUnity.Scripts;
using QuestionnaireToolkit.Scripts;
using TMPro;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.SceneManagement;
using UnityEngine.Serialization;
using UnityEngine.UI;
using PythonStarter = BOforUnity.Scripts.PythonStarter;

namespace BOforUnity
{
    public class BoForUnityManager : MonoBehaviour, IQuestionnaireOptimizationBridge
    {
        public enum IterationAdvanceMode
        {
            NextButton = 0,
            ExternalSignal = 1,
            Automatic = 2
        }

        public enum OptimizerBackend
        {
            BoTorch = 0,
            CABOP = 1,
            // Multi-objective Meta-BO (TAF-EHVI): qLogNEHVI blended with hypervolume
            // improvement terms from population models built from prior runs.
            MetaTAF = 2,
            // Dynamic BO: the GP covariance is multiplied by a fitted temporal decay
            // alpha^|t-t'|, for objectives that drift during a session (adaptation,
            // learning, fatigue). Single-objective. See Kim & Sergi, RA-L 2026.
            DBO = 3
        }

        public enum MetaWeightMode
        {
            // Weights from objective-wise pairwise ranking agreement with the current
            // user's observations (each observation pair scored once per objective).
            TafR = 0,
            // Weights from meta-feature similarity (dimension + objective moments).
            TafM = 1,
            // Ablation only: the former TafR — Pareto-dominance agreement, which discards
            // mutually non-dominated pairs and so starves the similarity estimate near
            // the front.
            TafRPareto = 2
        }

        public enum CabopObjectiveMode
        {
            SingleObjective = 0,
            MultiObjectiveScalarized = 1
        }

        public enum CabopUpdateRule
        {
            Actual = 0,
            Intended = 1,
            Both = 2
        }

        public enum DboSpatialKernel
        {
            // Squared exponential with ARD — what the reference DBO kernel uses.
            Rbf = 0,
            Matern52 = 1
        }

        public enum DboAlphaParameterization
        {
            // Fit 1 - alpha in log space, reproducing the reference implementation.
            Decay = 0,
            // Fit alpha directly under an interval constraint; better behaved when
            // drift is fast (alpha expected well below 1).
            Direct = 1
        }

        public enum ContextEmbeddingSource
        {
            // A low-dimensional embedding per context is learned from data (LCE-M default).
            Learned = 0,
            // Each context provides its own embedding vector (definable in the inspector or via code).
            Manual = 1,
            // Each context provides an image; Python embeds it with an open_clip vision
            // transformer (default ViT-bigG-14, the open_clip release of ViT-G/14).
            Image = 2
        }

        public PythonStarter pythonStarter;
        public Optimizer optimizer;
        public MainThreadDispatcher mainThreadDispatcher;
        public SocketNetwork socketNetwork;

        private static BoForUnityManager _instance;

        /// <summary>
        /// The persistent manager that runs the study (null before the first manager's Awake). Duplicate managers
        /// in reloaded scenes are discarded, so scripts should use this instead of searching the scene.
        /// </summary>
        public static BoForUnityManager Instance => _instance;

        //-----------------------------------------------
        // DESIGN PARAMETERS and DESIGN OBJECTIVES
        public List<ParameterEntry> parameters = new List<ParameterEntry>();
        public List<ObjectiveEntry> objectives = new List<ObjectiveEntry>();
        //-----------------------------------------------
        
        //-----------------------------------------------
        // ITERATION CONTROLLER
        [SerializeField]
        public int currentIteration;  // Current iteration value.
        /// <summary>
        /// The Iteration value Python will log the current design under in ObservationsPerEvaluation.csv
        /// (global index: warm-start rows and sampling included), or -1 when the backend did not send it.
        /// Use this to join Unity-side logs with the Python logs.
        /// </summary>
        public int LastSuggestionLogIteration { get; internal set; } = -1;
        /// <summary>
        /// The Iteration value the finaldesign row will be logged under in ObservationsPerEvaluation.csv (one past the
        /// last Iteration logged for this User/Condition/Group), or -1 until the final design has been selected.
        /// Use it to log the final round in Unity-side logs.
        /// </summary>
        public int FinalDesignLogIteration { get; private set; } = -1;
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
        // 128 QMC samples: 2.4-9x faster multi-objective suggestions than 512 for 2-5 objectives, with
        // candidates within 3e-3 (unit cube) of the 512-sample ones.
        public int mcSamples = 128;
        public int numSamplingIterations = 4; // Auto-default is 2(d+1), where d is the number of design parameters.
        public int numOptimizationIterations = 10;
        public int seed = 3;
        
        [SerializeField] private bool enableSamplingEdit = false; // checkbox in inspector
        
        public bool warmStart = false;
        public bool perfectRatingActive = false;
        public bool perfectRatingInInitialRounds = false;
        public string initialParametersDataPath;
        public string initialObjectivesDataPath;
        public string warmStartObjectiveFormat = "auto";

        public OptimizerBackend optimizerBackend = OptimizerBackend.BoTorch;

        public CabopObjectiveMode cabopObjectiveMode = CabopObjectiveMode.SingleObjective;
        public bool cabopUseCostAwareAcquisition = true;
        public CabopUpdateRule cabopUpdateRule = CabopUpdateRule.Actual;
        public bool cabopEnableCostBudget = false;
        [Min(-1f)] public float cabopMaxCumulativeCost = -1f;
        public List<CabopGroupCostEntry> cabopGroupCosts = new List<CabopGroupCostEntry>();

        // Meta-TAF (multi-objective Meta-BO) settings; only read when
        // optimizerBackend == MetaTAF. Population models are generated offline with
        // meta_train.py and placed under StreamingAssets/BOData/<metaSourceDir>.
        public string metaSourceDir = "MetaSources";
        // Abort the run when no population model survives frame validation, instead of
        // silently continuing as plain qLogNEHVI (which would turn a MetaTAF study
        // condition into the no-transfer control). Keep this ON for studies; only
        // disable it if a source-less run is genuinely intended.
        public bool metaRequireSources = true;
        public MetaWeightMode metaWeightMode = MetaWeightMode.TafR;
        [Min(0.0001f)] public float metaRho = 1.0f;
        [Min(0.0001f)] public float metaTargetWeight = 1.0f;
        [Min(0)] public int metaWarmupIters = 1;
        [Min(0)] public int metaDecayStartIter = 2;
        [Range(0f, 1f)] public float metaDecayRate = 0.3f;

        // Dynamic BO settings; only read when optimizerBackend == DBO. The fitted
        // decay rate alpha is logged per iteration to DboDiagnosticsPerEvaluation.csv —
        // if it stays near 1.0 for a whole run, the objective did not measurably
        // drift and plain BoTorch BO would have done the same job.
        public DboSpatialKernel dboSpatialKernel = DboSpatialKernel.Rbf;
        public DboAlphaParameterization dboAlphaParameterization = DboAlphaParameterization.Decay;
        // Strictly below 1: alpha cannot be fitted away from exactly 1. To pin alpha = 1,
        // use dboStationaryBaseline instead.
        [Range(0.01f, 0.999f)] public float dboInitialAlpha = 0.99f;
        // 0 scores acquisition candidates at the current time (reference behaviour);
        // 1 scores them at the time they will actually be evaluated.
        [Min(0f)] public float dboAcquisitionTimeOffset = 0f;
        // Every N iterations apply the model's best estimate instead of an
        // acquisition-driven point, so optimisers can be compared without their
        // exploration policies confounding the result. 0 disables.
        [Min(0)] public int dboValidationEvery = 0;
        // Tail probability of the validation upper confidence bound (0.01 ~ mean + 2.33 sd).
        [Range(0.0001f, 0.5f)] public float dboValidationConfidence = 0.01f;
        // Restrict validation candidates to already-evaluated inputs (reference
        // behaviour). Off searches the continuous domain, which tracks fast drift better.
        public bool dboValidationVisitedOnly = true;
        // Re-search with inflated signal variance when the acquisition collapses onto
        // a point the model is already sure about. 0 disables (plain EI). The source
        // study used 0.1, favouring exploitation.
        [Range(0f, 1f)] public float dboExplorationRatio = 0.1f;
        // Pin alpha = 1, reducing DBO to stationary BO — the ablation/baseline condition.
        public bool dboStationaryBaseline = false;

        // Contextual optimization (LCE-M multi-task GP; BoTorch backend only).
        // Observations are tagged with the current context; warm-start data from
        // other contexts (e.g. other users, devices, or environments) informs the
        // model through a learned or user-definable context embedding.
        public bool contextualOptimization = false;
        public ContextEmbeddingSource contextEmbeddingSource = ContextEmbeddingSource.Learned;
        public string currentContextKey = "";
        public bool normalizeContextEmbeddings = true;
        public string contextEmbeddingModel = "ViT-bigG-14";
        public string contextEmbeddingPretrained = "laion2b_s39b_b160k";
        public List<ContextEntry> contexts = new List<ContextEntry>();

        public IterationAdvanceMode iterationAdvanceMode = IterationAdvanceMode.NextButton;
        [Min(0f)] public float automaticAdvanceDelaySec = 0f;
        public bool reloadSceneOnIterationAdvance = true;

        public bool enableFinalDesignRound = false;
        [Min(0f)] public float finalDesignDistanceEpsilon = 1e-6f;
        [Min(0f)] public float finalDesignMaximinEpsilon = 1e-6f;
        [Min(0f)] public float finalDesignAggressionEpsilon = 1e-6f;

        public bool enablePriorSliderRatingHint = false;
        [Range(0.05f, 0.45f)] public float priorSliderRatingHintAlpha = 0.16f;

        public string userId = "-1";
        public string conditionId = "-1";
        public string groupId = "-1";

        public bool hasNewDesignParameterValues;
        private bool _runtimeUserFolderReserved = false;
        // User/condition IDs the folder was resolved for (the resolved token is written back to userId), so a
        // later change by code is noticed and resolved again.
        private string _resolvedFolderUserId;
        private string _resolvedFolderConditionId;
        // Configuration as this manager was created (before folder resolution rewrote userId and before code
        // reconfigured it); a discarded duplicate from a reloaded scene is compared against it.
        private bool _configurationCaptured;
        private List<string> _configuredParameterKeys;
        private List<string> _configuredObjectiveKeys;
        private OptimizerBackend _configuredBackend;
        private string _configuredUserId;
        private string _configuredConditionId;
        private string _configuredGroupId;
        private bool _pendingAdvanceRequest = false;
        private bool _loopTerminated = false;
        private Coroutine _automaticAdvanceCoroutine = null;
        private bool _warnedMissingNextButton = false;
        private bool _finalDesignRoundPrepared = false;
        private bool _finalDesignRoundInProgress = false;
        private bool _finalDesignRoundLogged = false;
        private string _finalDesignObservationCsvPath = null;
        private readonly Dictionary<string, float> _priorSliderRatingHints = new Dictionary<string, float>(StringComparer.Ordinal);
        private readonly Dictionary<string, string> _priorLinearScaleRatingHints = new Dictionary<string, string>(StringComparer.Ordinal);
        //-----------------------------------------------
        
        //-----------------------------------------------
        // Static state survives play sessions when "Enter Play Mode Options" disable domain reload; a stale
        // instance would make the next session's manager discard itself.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void ResetStaticState()
        {
            _instance = null;
        }

        private void Awake()
        {
            // If there is already an instance of this object, destroy the new one
            if (_instance != null && _instance != this)
            {
                WarnIfDiscardedManagerDiffers(_instance);
                // Destroy is deferred to the end of the frame. Deactivate first so scripts of the reloaded scene
                // cannot find (and bind to) this doomed copy in their Awake/Start, and its PythonStarter and
                // SocketNetwork never start.
                gameObject.SetActive(false);
                Destroy(gameObject);
                return;
            }
            // Mark this object as the single instance and make it persistent
            _instance = this;
            DontDestroyOnLoad(gameObject);

            SyncSamplingIterationDefaults();
            CaptureConfiguration();

            pythonStarter = gameObject.GetComponent<PythonStarter>();
            optimizer = gameObject.GetComponent<Optimizer>();
            mainThreadDispatcher = gameObject.GetComponent<MainThreadDispatcher>();
            socketNetwork = gameObject.GetComponent<SocketNetwork>();
            if (mainThreadDispatcher == null)
            {
                mainThreadDispatcher = gameObject.AddComponent<MainThreadDispatcher>();
                Debug.LogWarning(
                    "BoForUnityManager added a missing MainThreadDispatcher component at runtime. " +
                    "Please add MainThreadDispatcher to the manager prefab/scene object."
                );
            }

            currentIteration = 1;
            totalIterations = GetConfiguredTotalIterations(); // set how many iterations the optimizer should run for

            // The log folder is not reserved here: Awake order is undefined, and scripts commonly set the IDs (or
            // disable this manager for a baseline condition) in their own Awake. Reserving now would create
            // LogData/<serialized user>/<serialized condition> on disk, which later forces a suffixed user folder
            // when that participant runs that condition. Start reserves it, and UserId resolves it on first use
            // (a questionnaire that initializes before Start gets the final token).
        }

        private void CaptureConfiguration()
        {
            if (_configurationCaptured)
                return;
            _configurationCaptured = true;
            _configuredParameterKeys = NormalizedKeys(parameters?.Select(p => p?.key));
            _configuredObjectiveKeys = NormalizedKeys(objectives?.Select(o => o?.key));
            _configuredBackend = optimizerBackend;
            _configuredUserId = userId;
            _configuredConditionId = conditionId;
            _configuredGroupId = groupId;
        }

        private void WarnIfDiscardedManagerDiffers(BoForUnityManager running)
        {
            running.CaptureConfiguration();
            var differences = new List<string>();
            if (!NormalizedKeys(parameters?.Select(p => p?.key))
                    .SequenceEqual(running._configuredParameterKeys, StringComparer.OrdinalIgnoreCase))
                differences.Add("parameter keys");
            if (!NormalizedKeys(objectives?.Select(o => o?.key))
                    .SequenceEqual(running._configuredObjectiveKeys, StringComparer.OrdinalIgnoreCase))
                differences.Add("objective keys");
            if (optimizerBackend != running._configuredBackend)
                differences.Add($"backend ({optimizerBackend} vs. {running._configuredBackend})");
            if (!SameId(userId, running._configuredUserId))
                differences.Add("User ID");
            if (!SameId(conditionId, running._configuredConditionId))
                differences.Add("Condition ID");
            if (!SameId(groupId, running._configuredGroupId))
                differences.Add("Group ID");

            if (differences.Count == 0)
                return;

            Debug.LogWarning(
                $"BoForUnityManager in scene '{gameObject.scene.name}' was discarded because the manager from scene " +
                $"'{running.gameObject.scene.name}' persists across scene loads and keeps running. The discarded " +
                $"manager is configured differently ({string.Join(", ", differences)}), but the running manager's " +
                "configuration stays in effect. To run another condition, start a new play session/application run, " +
                "or destroy BoForUnityManager.Instance's GameObject before loading the next condition scene."
            );
        }

        private static List<string> NormalizedKeys(IEnumerable<string> keys)
        {
            return (keys ?? Enumerable.Empty<string>()).Select(k => (k ?? string.Empty).Trim()).ToList();
        }

        private static bool SameId(string a, string b)
        {
            return string.Equals((a ?? string.Empty).Trim(), (b ?? string.Empty).Trim(), StringComparison.Ordinal);
        }

        private void OnValidate()
        {
            SyncSamplingIterationDefaults();
            totalIterations = GetConfiguredTotalIterations();
        }
        
        void Start()
        {
            // A startup error reported before this Start (TerminateWithError, e.g. from the launcher) keeps its
            // message on screen instead of being replaced by the loading state.
            if (_loopTerminated)
                return;

            // Idempotent; resolves again only if code changed the IDs after an earlier resolution (UserId).
            EnsureUniqueRuntimeUserFolder();
            EnsureNextButtonListener();
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
            FinalDesignLogIteration = -1;
            ClearPriorSliderRatingHints();
            ClearPriorLinearScaleRatingHints();
            // Start each run from a clean measurement state so the first objective payload
            // cannot reuse stale values from a prior session or serialized inspector data.
            ClearObjectiveMeasurements();
            simulationRunning = true; // the simulation to true to prevent 
            totalIterations = GetConfiguredTotalIterations();
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
        [Tooltip("Optional. Shows \"Iteration x / N\" (or \"Final design\") whenever a new design is ready.")]
        public TMP_Text progressText;

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

            // Prevent duplicate UI events from queuing a stale request after a ready state was consumed.
            // External-signal mode may still queue while the optimizer is running.
            if (!hasNewDesignParameterValues &&
                (iterationAdvanceMode == IterationAdvanceMode.NextButton ||
                 (iterationAdvanceMode == IterationAdvanceMode.ExternalSignal && !optimizationRunning)))
            {
                return;
            }

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
            if (!initialized || hasNewDesignParameterValues || !simulationRunning)
            {
                // Only an evaluation in progress can be reported. Before the optimizer is ready, or
                // after new parameters arrived but before the next evaluation started (e.g. a second
                // questionnaire finishing, or a pre-study questionnaire in the scene), the
                // measurements would be recorded against a design the participant never saw.
                Debug.LogWarning(
                    "OptimizationStart ignored because no design is currently being evaluated " +
                    $"(initialized={initialized}, newParametersPending={hasNewDesignParameterValues}, " +
                    $"evaluationRunning={simulationRunning})."
                );
                return;
            }
            if (socketNetwork == null)
            {
                Debug.LogError("OptimizationStart failed because SocketNetwork is not assigned.");
                TerminateWithError("Optimizer connection is not configured.\nCheck the manager setup and logs.");
                return;
            }

            Debug.Log("Optimization START");
            CancelAutomaticAdvance();
            if (iterationAdvanceMode == IterationAdvanceMode.NextButton)
            {
                // Defensive reset against stale requests from duplicate UI events.
                _pendingAdvanceRequest = false;
            }

            try
            {
                socketNetwork.SendObjectives(); // send the current objective values to the Python process
            }
            catch (Exception e)
            {
                Debug.LogError($"OptimizationStart failed while sending objectives: {e.Message}");
                // Ends the loop as well (status panel, pending automatic advance, Python stopped): the evaluation
                // cannot reach the optimizer, and a second OptimizationStart must not try again.
                TerminateWithError("Could not send objective values to the optimizer. Check configuration and logs.");
                return;
            }
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
            // optimizationFinished is already set when the run ended early (perfect rating) and the final-design
            // round was prepared; a late message must not select the final design a second time.
            if (_loopTerminated || optimizationFinished)
                return;

            Debug.Log(">>>>>> Optimization finished!");
            optimizationFinished = true;
            EnterFinalDesignRoundOrComplete();
        }

        /// <summary>
        /// Ends the study with <paramref name="statusText"/> on screen: shows the status panel that holds the output
        /// text, hides loading and Next, cancels a pending automatic advance, ignores further advance requests and
        /// stops the optimizer connection and the Python process. Does nothing once the loop has ended.
        /// </summary>
        public void TerminateWithError(string statusText)
        {
            if (_loopTerminated)
                return;

            Debug.LogError("BoForUnityManager: the study was stopped. " + statusText);
            CompleteLoop(
                string.IsNullOrWhiteSpace(statusText)
                    ? "Optimizer connection failed.\nCheck parameter/objective configuration and Python logs, then restart."
                    : statusText
            );
        }

        private void EnterFinalDesignRoundOrComplete()
        {
            if (!enableFinalDesignRound)
            {
                CompleteLoop();
                return;
            }

            // The backend's part is over (it sent optimization_finished or was told to stop); the final round is
            // evaluated in Unity only. End the session now: SocketQuit lets Python exit by itself (saving logs that
            // were locked) within its grace period, so the selection below reads finished logs, and Python's exit
            // cannot be reported as a crash while the participant evaluates the final design.
            try
            {
                socketNetwork?.SocketQuit();
            }
            catch (Exception e)
            {
                Debug.LogWarning($"SocketQuit failed before the final design round: {e.Message}");
            }

            if (!TryPrepareFinalDesignRound(out var selectionError))
            {
                Debug.LogWarning(
                    "Final design round is enabled, but no final design could be selected. " +
                    $"Falling back to normal completion. Reason: {selectionError}"
                );
                CompleteLoop(
                    "Optimization has finished, but no final design could be selected.\n" +
                    "Check the console and observation CSV configuration before running a final evaluation."
                );
                return;
            }

            _pendingAdvanceRequest = false;
            HandleParametersReady(
                "Optimization has finished.\nThe selected final design is ready for one last evaluation round.",
                forceManualAdvance: true,
                nextButtonText: "Start Final Evaluation"
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

            // Last check before the init message: these settings would otherwise crash the backend at init or,
            // worse, after the participant finished the sampling phase. Code may have changed the configuration
            // since Awake, so check it now rather than at startup.
            EnsureUniqueRuntimeUserFolder();
            if (!BoConfigValidator.TryValidate(this, out string configError))
            {
                TerminateWithError("Invalid optimizer configuration.\n" + configError);
                return;
            }
            if (reloadSceneOnIterationAdvance && !CanReloadScene(SceneManager.GetActiveScene()))
            {
                TerminateWithError(GetSceneNotReloadableMessage(SceneManager.GetActiveScene()));
                return;
            }

            socketNetwork.InitSocket();
        }

        private void HandleParametersReady(string statusText, bool forceManualAdvance = false, string nextButtonText = null)
        {
            if (_loopTerminated)
                return;

            hasNewDesignParameterValues = true;
            optimizationRunning = false;
            simulationRunning = false;
            UpdateProgressText();

            if (!string.IsNullOrWhiteSpace(nextButtonText))
                SetNextButtonText(nextButtonText);

            // External-signal flows may already have queued the next transition while
            // Python was optimizing. Consume that ready state without flashing the
            // manual ready/Next UI for a frame.
            if (!forceManualAdvance &&
                iterationAdvanceMode == IterationAdvanceMode.ExternalSignal &&
                currentIteration == 1)
            {
                _pendingAdvanceRequest = true;
            }

            if (!forceManualAdvance &&
                iterationAdvanceMode == IterationAdvanceMode.ExternalSignal &&
                _pendingAdvanceRequest)
            {
                SetNextButtonVisible(false);
                SetLoadingVisible(false);
                SetOptimizerStatePanelVisible(false);
                SetOutputText(statusText);
                TryConsumeAdvanceRequest();
                return;
            }

            SetNextButtonVisible(false);
            SetLoadingVisible(false);
            SetOutputText(statusText);
            // Keep the status panel visible only when the ready-state UI lives inside it.
            SetOptimizerStatePanelVisible(RequiresOptimizerPanelForReadyStateUi(forceManualAdvance));

            if (forceManualAdvance)
            {
                SetNextButtonVisible(true);
                if (nextButton == null && !_warnedMissingNextButton)
                {
                    _warnedMissingNextButton = true;
                    Debug.LogWarning(
                        "A manual advance is required, but no Next Button is assigned. " +
                        "Assign a button or call RequestNextIteration() from your own logic."
                    );
                }
                return;
            }

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

            if (_finalDesignRoundPrepared)
            {
                _finalDesignRoundPrepared = false;
                _finalDesignRoundInProgress = true;
                ClearObjectiveMeasurements();

                Debug.Log("--------------------------------------Current Iteration (Final Design Round): " +
                          currentIteration.ToString(CultureInfo.InvariantCulture));
                simulationRunning = true;
                SetOptimizerStatePanelVisible(false);

                if (reloadSceneOnIterationAdvance)
                {
                    ReloadActiveScene();
                }
                else
                {
                    SetLoadingVisible(false);
                }
                return;
            }

            if (ShouldStopForPerfectRating())
            {
                Debug.Log(">>>>> Perfect Rating");
                // Python already proposed the next design; tell it the run ends here so it exits cleanly
                // (and logs why) instead of being killed while it waits for objectives. The final-design round
                // applies after an early stop exactly as after a full run.
                optimizationFinished = true;
                socketNetwork?.SendStop("perfect_rating");
                EnterFinalDesignRoundOrComplete();
                return;
            }

            if (optimizationFinished || currentIteration > totalIterations)
            {
                CompleteLoop();
                return;
            }

            // hide the panel as the next iteration starts after scene transition
            SetOptimizerStatePanelVisible(false);

            Debug.Log("--------------------------------------Current Iteration: " +
                      currentIteration.ToString(CultureInfo.InvariantCulture));

            ClearObjectiveMeasurements();
            simulationRunning = true; // waiting for the simulation to finish

            if (reloadSceneOnIterationAdvance)
            {
                ReloadActiveScene(); // reload scene
            }
            else
            {
                SetLoadingVisible(false);
            }
        }

        private void ReloadActiveScene()
        {
            // Reload by build index: loading by name picks the first Build Settings scene with that name (another
            // folder's scene of the same name), and a scene missing from Build Settings is never loaded by name,
            // which left the loading screen up forever.
            Scene scene = SceneManager.GetActiveScene();
            if (!CanReloadScene(scene))
            {
                TerminateWithError(GetSceneNotReloadableMessage(scene));
                return;
            }

            if (scene.buildIndex >= 0)
            {
                SceneManager.LoadScene(scene.buildIndex);
                return;
            }

#if UNITY_EDITOR
            // Not in the Build Profiles scene list (the example scenes are not): the Editor reloads it by asset path.
            EditorSceneManager.LoadSceneInPlayMode(scene.path, new LoadSceneParameters(LoadSceneMode.Single));
#else
            // A player scene without build index comes from an AssetBundle; those load by their asset path.
            SceneManager.LoadScene(scene.path);
#endif
        }

        private static bool CanReloadScene(Scene scene)
        {
            return scene.buildIndex >= 0 || !string.IsNullOrEmpty(scene.path);
        }

        private static string GetSceneNotReloadableMessage(Scene scene)
        {
            return
                $"Scene '{scene.name}' has no build index and no asset path, so it cannot be reloaded for the next " +
                "iteration.\nAdd it under File > Build Profiles (Scene List), or disable Reload Scene On Advance.";
        }

        private void UpdateProgressText()
        {
            if (progressText == null)
                return;

            progressText.text = _finalDesignRoundPrepared || _finalDesignRoundInProgress
                ? "Final design"
                : "Iteration " + currentIteration.ToString(CultureInfo.InvariantCulture) + " / " +
                  Mathf.Max(currentIteration, totalIterations).ToString(CultureInfo.InvariantCulture);
        }

        private void ClearObjectiveMeasurements()
        {
            if (objectives == null)
                return;

            foreach (var objective in objectives)
            {
                if (objective?.value?.values == null)
                    continue;

                objective.value.values.Clear();
            }
        }

        private bool ShouldStopForPerfectRating()
        {
            if (!perfectRatingActive)
                return false;
            int initialRounds = GetEffectiveSamplingIterations();
            // currentIteration was already advanced past the evaluation being judged.
            if (!perfectRatingInInitialRounds && currentIteration - 1 <= initialRounds)
                return false;
            return IsPerfectRating();
        }

        public static int ComputeRecommendedSamplingIterations(int designParameterCount)
        {
            return 2 * (Mathf.Max(0, designParameterCount) + 1);
        }

        private void SyncSamplingIterationDefaults()
        {
            if (warmStart)
            {
                return;
            }

            if (enableSamplingEdit)
                return;

            numSamplingIterations = ComputeRecommendedSamplingIterations(parameters?.Count ?? 0);
        }

        public int GetEffectiveSamplingIterations()
        {
            return warmStart ? 0 : Mathf.Max(0, numSamplingIterations);
        }

        private int GetConfiguredTotalIterations()
        {
            int samplingIterations = GetEffectiveSamplingIterations();
            int optimizationIterations = Mathf.Max(0, numOptimizationIterations);
            return samplingIterations + optimizationIterations;
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
                // Real time: a scene that pauses with Time.timeScale = 0 must not stall the study forever.
                yield return new WaitForSecondsRealtime(automaticAdvanceDelaySec);
            }

            _automaticAdvanceCoroutine = null;
            RequestNextIteration();
        }

        private void CompleteLoop(string statusText = null)
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
            hasNewDesignParameterValues = false;
            _waitingForPythonProcess = false;

            SetOptimizerStatePanelVisible(RequiresOptimizerPanelForOutputText());
            SetLoadingVisible(false);
            SetNextButtonVisible(false);
            SetOutputText(
                string.IsNullOrWhiteSpace(statusText)
                    ? "The simulation has finished!\nYou can now close the application."
                    : statusText
            );

            try
            {
                socketNetwork?.SocketQuit();
            }
            catch (Exception e)
            {
                Debug.LogWarning($"SocketQuit failed during loop termination: {e.Message}");
            }
        }

        private bool RequiresOptimizerPanelForReadyStateUi(bool forceManualAdvance = false)
        {
            if (RequiresOptimizerPanelForOutputText())
                return true;

            if (optimizerStatePanel == null)
                return false;

            var panelTransform = optimizerStatePanel.transform;
            if ((forceManualAdvance || iterationAdvanceMode == IterationAdvanceMode.NextButton) &&
                nextButton != null &&
                nextButton.transform.IsChildOf(panelTransform))
            {
                return true;
            }

            return false;
        }

        private bool RequiresOptimizerPanelForOutputText()
        {
            return optimizerStatePanel != null &&
                   outputText != null &&
                   outputText.transform.IsChildOf(optimizerStatePanel.transform);
        }

        private void EnsureNextButtonListener()
        {
            if (nextButton == null)
                return;

            Button button = nextButton.GetComponent<Button>();
            if (button == null)
                return;

            if (HasPersistentNextButtonListener(button))
                return;

            button.onClick.RemoveListener(RequestNextIteration);
            button.onClick.AddListener(RequestNextIteration);
        }

        private bool HasPersistentNextButtonListener(Button button)
        {
            if (button == null)
                return false;

            int count = button.onClick.GetPersistentEventCount();
            for (int i = 0; i < count; i++)
            {
                if (button.onClick.GetPersistentListenerState(i) == UnityEventCallState.Off)
                    continue;

                if (button.onClick.GetPersistentTarget(i) != this)
                    continue;

                string method = button.onClick.GetPersistentMethodName(i);
                if (method == nameof(ButtonNextIteration) || method == nameof(RequestNextIteration))
                    return true;
            }

            return false;
        }

        private void SetNextButtonText(string value)
        {
            if (nextButton == null)
                return;

            TMP_Text buttonText = nextButton.GetComponentInChildren<TMP_Text>(true);
            if (buttonText != null)
                buttonText.text = value;
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

        private void SetOutputText(string value)
        {
            if (outputText != null)
                outputText.text = value;
        }

        private static List<ParameterEntry> BuildEffectiveParameterEntries(
            IList<ParameterEntry> source,
            string context,
            bool logWarnings = true)
        {
            var result = new List<ParameterEntry>();
            if (source == null)
                return result;

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < source.Count; i++)
            {
                var entry = source[i];
                if (entry == null || entry.value == null || string.IsNullOrWhiteSpace(entry.key))
                {
                    if (logWarnings)
                    {
                        Debug.LogWarning($"Skipping invalid parameter entry at index {i} during {context}.");
                    }
                    continue;
                }

                string key = entry.key.Trim();
                if (!seen.Add(key))
                {
                    if (logWarnings)
                    {
                        Debug.LogWarning($"Skipping duplicate parameter key '{key}' during {context}.");
                    }
                    continue;
                }

                result.Add(new ParameterEntry(key, entry.value));
            }

            return result;
        }

        private static List<ObjectiveEntry> BuildEffectiveObjectiveEntries(
            IList<ObjectiveEntry> source,
            string context,
            bool logWarnings = true)
        {
            var result = new List<ObjectiveEntry>();
            if (source == null)
                return result;

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < source.Count; i++)
            {
                var entry = source[i];
                if (entry == null || entry.value == null || string.IsNullOrWhiteSpace(entry.key))
                {
                    if (logWarnings)
                    {
                        Debug.LogWarning($"Skipping invalid objective entry at index {i} during {context}.");
                    }
                    continue;
                }

                string key = entry.key.Trim();
                if (!seen.Add(key))
                {
                    if (logWarnings)
                    {
                        Debug.LogWarning($"Skipping duplicate objective key '{key}' during {context}.");
                    }
                    continue;
                }

                result.Add(new ObjectiveEntry(key, entry.value));
            }

            return result;
        }

        private bool TryPrepareFinalDesignRound(out string error)
        {
            FinalDesignLogIteration = -1;
            _finalDesignRoundPrepared = false;
            _finalDesignRoundInProgress = false;
            _finalDesignRoundLogged = false;
            _finalDesignObservationCsvPath = null;

            var effectiveParameters = BuildEffectiveParameterEntries(parameters, "final-design selection");
            var effectiveObjectives = BuildEffectiveObjectiveEntries(objectives, "final-design selection");
            if (effectiveParameters.Count == 0 || effectiveObjectives.Count == 0)
            {
                error = "No valid parameters or objectives are configured for final-design selection.";
                return false;
            }

            // The first candidate lies below the log root Python writes to (LogDataFolderUtility.LogDataRoot): this
            // condition's folder, or its CABOP/<mode> folder for CABOP. The rest are fallbacks (the root itself,
            // StreamingAssets/persistentDataPath roots of builds that fell back, legacy and CABOP layouts).
            string[] logRootCandidates = GetFinalDesignLogRootCandidates();
            if (!FinalDesignSelector.TrySelectFromLogRoots(
                    primaryLogRoot: logRootCandidates[0],
                    fallbackLogRoots: logRootCandidates.Skip(1),
                    userId: userId,
                    conditionId: conditionId,
                    groupId: groupId,
                    parameters: effectiveParameters,
                    objectives: effectiveObjectives,
                    distanceEpsilon: finalDesignDistanceEpsilon,
                    maximinEpsilon: finalDesignMaximinEpsilon,
                    aggressionEpsilon: finalDesignAggressionEpsilon,
                    selection: out FinalDesignSelector.SelectionResult selected,
                    selectedCsvPath: out string selectedCsvPath,
                    selectedLogRoot: out string selectedLogRoot,
                    error: out string selectionError))
            {
                error = selectionError;
                return false;
            }

            string primaryRoot = LogDataFolderUtility.NormalizeRoot(logRootCandidates[0]);
            if (!string.Equals(selectedLogRoot, primaryRoot, StringComparison.OrdinalIgnoreCase))
            {
                Debug.LogWarning(
                    $"Final design selector used fallback log root: {selectedLogRoot}. Primary path was: {primaryRoot}"
                );
            }

            if (selected == null || selected.ParameterRaw == null)
            {
                error = "Final design selector returned an invalid result.";
                return false;
            }

            if (selected.ParameterRaw.Length != effectiveParameters.Count)
            {
                error = "Selected final-design parameter count does not match current parameter list.";
                return false;
            }

            for (int i = 0; i < effectiveParameters.Count; i++)
            {
                var parameterEntry = effectiveParameters[i];
                if (parameterEntry == null || parameterEntry.value == null || string.IsNullOrWhiteSpace(parameterEntry.key))
                {
                    error = $"Parameter entry at index {i} is invalid.";
                    return false;
                }

                float selectedValue = selected.ParameterRaw[i];
                if (float.IsNaN(selectedValue) || float.IsInfinity(selectedValue))
                {
                    error = $"Selected parameter '{parameterEntry.key}' is non-finite.";
                    return false;
                }

                float lo = parameterEntry.value.lowerBound;
                float hi = parameterEntry.value.upperBound;
                float eps = 1e-4f;
                if (selectedValue < lo - eps || selectedValue > hi + eps)
                {
                    error = $"Selected parameter '{parameterEntry.key}'={selectedValue} is outside bounds [{lo}, {hi}].";
                    return false;
                }

                parameterEntry.value.Value = Mathf.Clamp(selectedValue, lo, hi);
            }

            currentIteration = totalIterations + 1;
            _finalDesignRoundPrepared = true;
            _finalDesignRoundInProgress = false;
            _finalDesignRoundLogged = false;
            _finalDesignObservationCsvPath = selectedCsvPath;
            // Fixed now so scripts can log the final round under the same Iteration before the row is written.
            FinalDesignLogIteration = ResolveFinalDesignLogIteration(selectedCsvPath);

            Debug.Log(
                "Selected final design for last evaluation round: " +
                $"iteration={selected.Iteration}, candidates={selected.CandidateSource}, " +
                $"utopiaDist={selected.UtopiaDistance}, maximin={selected.Maximin}, " +
                $"aggression={selected.Aggression}, logIteration={FinalDesignLogIteration}, csv={selectedCsvPath}"
            );
            if (selected.CandidateSource == "computed" || selected.CandidateSource == "all")
            {
                Debug.LogWarning(
                    $"The observation log has no IsBest/IsPareto flags for this run (candidates={selected.CandidateSource}: " +
                    "the best/non-dominated rows were computed in Unity, or all rows were used), so the final design " +
                    $"was not chosen from the backend's flags. Check that the backend finished writing {selectedCsvPath}."
                );
            }

            error = null;
            return true;
        }

        // Excel keeps an open CSV locked for writing on Windows; retry briefly before using a fallback file.
        private const int FinalDesignAppendAttempts = 5;
        private const int FinalDesignAppendRetryDelayMs = 200;
        private static readonly Encoding Utf8NoBom = new UTF8Encoding(false);

        private bool TryAppendFinalDesignObservationRow(out string error)
        {
            error = null;
            string csvPath = _finalDesignObservationCsvPath;

            if (string.IsNullOrWhiteSpace(csvPath))
            {
                error = "Final-design CSV path is empty.";
                return false;
            }
            if (!File.Exists(csvPath))
            {
                error = $"Final-design CSV does not exist: {csvPath}";
                return false;
            }

            List<string> lines;
            try
            {
                lines = ReadLinesShared(csvPath);
            }
            catch (Exception ex)
            {
                error = $"Could not read CSV: {ex.Message}";
                return false;
            }

            string headerLine = lines.Count > 0 ? lines[0].Trim('\uFEFF') : null;
            if (string.IsNullOrWhiteSpace(headerLine))
            {
                error = "Observation CSV has no header.";
                return false;
            }

            string[] header = SplitCsvLine(headerLine);
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

            SetColumn("UserID", LogIdToken(userId));
            SetColumn("ConditionID", LogIdToken(conditionId));
            SetColumn("GroupID", LogIdToken(groupId));
            SetColumn("Timestamp", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
            if (FinalDesignLogIteration < 0)
                FinalDesignLogIteration = ResolveFinalDesignLogIteration(lines, columnIndex);
            SetColumn("Iteration", FinalDesignLogIteration.ToString(CultureInfo.InvariantCulture));
            SetColumn("Phase", "finaldesign");
            if (contextualOptimization)
                SetColumn(BoConfigValidator.ContextLogColumn, GetCurrentContextKeyForLogging());
            SetColumn("IsPareto", "NULL");
            SetColumn("IsBest", "NULL");

            var effectiveObjectives = BuildEffectiveObjectiveEntries(objectives, "finaldesign logging");
            foreach (var objective in effectiveObjectives)
            {
                // The same reduction SendObjectives uses for the values Python receives.
                var arg = objective.value;
                var aggregate = BoObjectiveMath.Aggregate(arg.values, arg.numberOfSubMeasures, arg.lowerBound, arg.upperBound);
                if (aggregate.UsedMidpointFallback)
                {
                    Debug.LogWarning(
                        $"Objective '{objective.key}' has no numeric values during finaldesign logging. " +
                        $"Using midpoint fallback {BoObjectiveMath.FormatCsvFloat(aggregate.Value)}."
                    );
                }
                else
                {
                    if (aggregate.FiniteCount < aggregate.WindowCount)
                    {
                        Debug.LogWarning(
                            $"Objective '{objective.key}': ignoring {aggregate.WindowCount - aggregate.FiniteCount} " +
                            $"non-numeric sub-measure(s) during finaldesign logging and averaging the remaining {aggregate.FiniteCount}."
                        );
                    }
                    if (aggregate.Clamped)
                    {
                        Debug.LogWarning(
                            $"Objective '{objective.key}' value {BoObjectiveMath.FormatCsvFloat(aggregate.UnclampedValue)} is " +
                            "outside its configured bounds during finaldesign logging. " +
                            $"Clamped to {BoObjectiveMath.FormatCsvFloat(aggregate.Value)}."
                        );
                    }
                }
                if (aggregate.DroppedCount > 0)
                {
                    Debug.LogWarning(
                        $"Objective '{objective.key}': {aggregate.DroppedCount} older sub-measure(s) beyond Number Of " +
                        "Sub Measures were not part of the finaldesign value."
                    );
                }

                SetColumn(objective.key, BoObjectiveMath.FormatCsvFloat(aggregate.Value));
            }

            var effectiveParameters = BuildEffectiveParameterEntries(parameters, "finaldesign logging");
            foreach (var parameter in effectiveParameters)
            {
                if (parameter == null || string.IsNullOrWhiteSpace(parameter.key) || parameter.value == null)
                    continue;

                float rawValue = parameter.value.Value;
                if (!BoObjectiveMath.IsFinite(rawValue))
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

                SetColumn(parameter.key, BoObjectiveMath.FormatCsvFloat(rawValue));
            }

            string rowLine = string.Join(";", row.Select(EscapeCsvCell));
            string prefix;
            try
            {
                prefix = EndsWithLineBreak(csvPath) ? string.Empty : Environment.NewLine;
            }
            catch (Exception ex)
            {
                error = $"Could not inspect CSV newline state: {ex.Message}";
                return false;
            }

            if (TryAppendWithRetry(csvPath, prefix + rowLine + Environment.NewLine, out string appendError))
            {
                Debug.Log(
                    $"Final design evaluation appended to observations CSV (phase=finaldesign, marker=NULL): {csvPath}"
                );
                return true;
            }

            // Keep the measurement even when the log stays locked: write header + row next to it.
            string fallbackPath = Path.Combine(
                Path.GetDirectoryName(csvPath) ?? string.Empty,
                Path.GetFileNameWithoutExtension(csvPath) + "_finaldesign_" +
                DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture) + ".csv"
            );
            try
            {
                File.WriteAllText(fallbackPath, headerLine + Environment.NewLine + rowLine + Environment.NewLine, Utf8NoBom);
            }
            catch (Exception ex)
            {
                error =
                    $"Could not append finaldesign row ({appendError}) and could not write it to '{fallbackPath}' " +
                    $"either ({ex.Message}).";
                return false;
            }

            Debug.LogError(
                $"Could not append the finaldesign row to '{csvPath}' ({appendError}); is it open in another " +
                $"program such as Excel? The row was written to '{fallbackPath}' instead. Append it to " +
                "ObservationsPerEvaluation.csv before analysis."
            );
            return true;
        }

        private int ResolveFinalDesignLogIteration(string csvPath)
        {
            var lines = new List<string>();
            var columnIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            try
            {
                lines = ReadLinesShared(csvPath);
                string[] header = lines.Count > 0 ? SplitCsvLine(lines[0].Trim('\uFEFF')) : new string[0];
                for (int i = 0; i < header.Length; i++)
                {
                    string key = (header[i] ?? string.Empty).Trim();
                    if (!string.IsNullOrEmpty(key) && !columnIndex.ContainsKey(key))
                        columnIndex[key] = i;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Could not read '{csvPath}' to number the finaldesign row: {ex.Message}");
            }

            return ResolveFinalDesignLogIteration(lines, columnIndex);
        }

        /// <summary>
        /// The Iteration for the finaldesign row: one past the largest Iteration this context logged in the CSV
        /// (rows of other IDs only when none match), so the row never repeats a live row's Iteration, also not under
        /// warm start where live rows continue from the number of warm-start rows.
        /// </summary>
        private int ResolveFinalDesignLogIteration(List<string> lines, Dictionary<string, int> columnIndex)
        {
            if (columnIndex.TryGetValue("Iteration", out int iterationIndex))
            {
                columnIndex.TryGetValue("UserID", out int userIndex);
                columnIndex.TryGetValue("ConditionID", out int conditionIndex);
                columnIndex.TryGetValue("GroupID", out int groupIndex);
                bool hasUser = columnIndex.ContainsKey("UserID");
                bool hasCondition = columnIndex.ContainsKey("ConditionID");
                bool hasGroup = columnIndex.ContainsKey("GroupID");

                int maxAll = int.MinValue;
                int maxOwn = int.MinValue;
                for (int i = 1; i < lines.Count; i++)
                {
                    if (string.IsNullOrWhiteSpace(lines[i]))
                        continue;

                    string[] cells = SplitCsvLine(lines[i]);
                    if (iterationIndex >= cells.Length ||
                        !double.TryParse(cells[iterationIndex].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) ||
                        double.IsNaN(parsed) || double.IsInfinity(parsed) ||
                        parsed < int.MinValue || parsed >= int.MaxValue)
                    {
                        continue;
                    }

                    int iteration = (int)Math.Round(parsed);
                    maxAll = Math.Max(maxAll, iteration);
                    if ((!hasUser || CellEquals(cells, userIndex, userId)) &&
                        (!hasCondition || CellEquals(cells, conditionIndex, conditionId)) &&
                        (!hasGroup || CellEquals(cells, groupIndex, groupId)))
                    {
                        maxOwn = Math.Max(maxOwn, iteration);
                    }
                }

                if (maxOwn != int.MinValue)
                    return maxOwn + 1;
                if (maxAll != int.MinValue)
                    return maxAll + 1;
            }

            int fallbackIteration = LastSuggestionLogIteration >= 0 ? LastSuggestionLogIteration + 1 : currentIteration;
            Debug.LogWarning(
                "Could not read any Iteration from the observation CSV; logging the finaldesign row as Iteration " +
                fallbackIteration.ToString(CultureInfo.InvariantCulture) + "."
            );
            return fallbackIteration;
        }

        private static bool CellEquals(string[] cells, int index, string expected)
        {
            return index < cells.Length &&
                   string.Equals(LogIdToken(cells[index]), LogIdToken(expected), StringComparison.Ordinal);
        }

        // An ID as the backends log it (bo_protocol.normalize_user_token): trimmed, "-1" when empty.
        private static string LogIdToken(string value)
        {
            return string.IsNullOrWhiteSpace(value) ? "-1" : value.Trim();
        }

        // The context key as Python logs it: the configured entry's spelling (Python matches the current key
        // case-insensitively against the context list).
        private string GetCurrentContextKeyForLogging()
        {
            string key = (currentContextKey ?? string.Empty).Trim();
            if (contexts != null)
            {
                foreach (var entry in contexts)
                {
                    if (entry != null && !string.IsNullOrWhiteSpace(entry.key) &&
                        string.Equals(entry.key.Trim(), key, StringComparison.OrdinalIgnoreCase))
                    {
                        return entry.key.Trim();
                    }
                }
            }
            return key;
        }

        // Shared read: File.ReadLines denies other writers, so it fails while Excel has the CSV open.
        private static List<string> ReadLinesShared(string path)
        {
            var lines = new List<string>();
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            using (var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                    lines.Add(line);
            }
            return lines;
        }

        private static bool EndsWithLineBreak(string path)
        {
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            {
                if (stream.Length == 0)
                    return true;
                stream.Seek(-1, SeekOrigin.End);
                int last = stream.ReadByte();
                return last == '\n' || last == '\r';
            }
        }

        private static bool TryAppendWithRetry(string path, string text, out string error)
        {
            error = null;
            byte[] bytes = Utf8NoBom.GetBytes(text);
            for (int attempt = 1; ; attempt++)
            {
                try
                {
                    using (var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read))
                    {
                        stream.Write(bytes, 0, bytes.Length);
                    }
                    return true;
                }
                catch (IOException) when (attempt < FinalDesignAppendAttempts)
                {
                    System.Threading.Thread.Sleep(FinalDesignAppendRetryDelayMs);
                }
                catch (Exception ex)
                {
                    error = ex.Message;
                    return false;
                }
            }
        }

        // Splits one ';'-separated line, honouring "..." quoting with "" escapes (as Python's csv module writes).
        private static string[] SplitCsvLine(string line)
        {
            var cells = new List<string>();
            var cell = new StringBuilder();
            bool inQuotes = false;
            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (inQuotes)
                {
                    if (c != '"')
                        cell.Append(c);
                    else if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        cell.Append('"');
                        i++;
                    }
                    else
                        inQuotes = false;
                }
                else if (c == '"' && cell.Length == 0)
                    inQuotes = true;
                else if (c == ';')
                {
                    cells.Add(cell.ToString());
                    cell.Clear();
                }
                else
                    cell.Append(c);
            }
            cells.Add(cell.ToString());
            return cells.ToArray();
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

        /// <summary>
        /// Reserves this run's log folder: <c>userId</c> is kept unless <c>LogData/&lt;userId&gt;/&lt;conditionId&gt;</c>
        /// already exists, in which case a suffixed user folder (<c>_1</c>, <c>_2</c>, ...) is used and written back to
        /// <c>userId</c>. Idempotent; resolves again only when code changed the IDs since the last resolution.
        /// </summary>
        private void EnsureUniqueRuntimeUserFolder()
        {
            if (!Application.isPlaying)
                return;
            // A questionnaire may ask before Awake ran; remember the configuration before userId is rewritten.
            CaptureConfiguration();
            if (_runtimeUserFolderReserved &&
                string.Equals(userId, _resolvedFolderUserId, StringComparison.Ordinal) &&
                string.Equals(conditionId, _resolvedFolderConditionId, StringComparison.Ordinal))
            {
                return;
            }

            string requestedUserId = userId;
            string normalizedRequestedUserId = LogDataFolderUtility.NormalizeLogFolderToken(requestedUserId);
            // An existing user folder from another condition is reused (one participant, several condition
            // folders); only an existing folder for this same condition forces a suffix.
            userId = LogDataFolderUtility.GetOrCreateUserFolderTokenForCondition(
                LogDataFolderUtility.LogDataRoot,
                requestedUserId,
                conditionId,
                allowExistingRequestedUserFolder: true
            );
            _runtimeUserFolderReserved = true;
            _resolvedFolderUserId = userId;
            _resolvedFolderConditionId = conditionId;

            if (!string.Equals(normalizedRequestedUserId, userId, StringComparison.Ordinal))
            {
                Debug.Log(
                    $"BOforUnity: log folder '{normalizedRequestedUserId}/" +
                    $"{LogDataFolderUtility.NormalizeLogFolderToken(conditionId)}' already exists. " +
                    $"Using user folder '{userId}' for this run."
                );
            }
        }

        public void ClearPriorSliderRatingHints()
        {
            _priorSliderRatingHints.Clear();
        }

        public bool UsesExternalIterationSignal => iterationAdvanceMode == IterationAdvanceMode.ExternalSignal;

        public bool EnablePriorRatingHints => enablePriorSliderRatingHint;

        public float PriorRatingHintAlpha => priorSliderRatingHintAlpha;

        // Questionnaires read the IDs through IQuestionnaireOptimizationBridge, possibly before this manager's
        // Awake (Awake order is undefined), so the log folder is resolved on first use and every reader gets the
        // final user token. A duplicate in a reloaded scene that is not discarded yet answers for the running manager.
        public string UserId
        {
            get
            {
                if (_instance != null && _instance != this)
                    return _instance.UserId;
                EnsureUniqueRuntimeUserFolder();
                return userId;
            }
        }

        public string ConditionId => _instance != null && _instance != this ? _instance.conditionId : conditionId;

        public string GroupId => _instance != null && _instance != this ? _instance.groupId : groupId;

        public void SubmitQuestionnaireObjectiveValue(string headerName, string rawValue, string sourceName)
        {
            if (optimizer == null || !optimizer.HasObjectiveMatch(headerName))
            {
                return;
            }

            if (!TryParseQuestionnaireNumber(rawValue, out var value))
            {
                Debug.LogWarning(
                    $"Objective value for '{headerName}' from '{sourceName}' is not numeric ('{rawValue}'). " +
                    "Submitting NaN so this iteration uses BO fallback handling instead of reusing a stale value."
                );
                optimizer.AddObjectiveValue(headerName, float.NaN);
                return;
            }

            optimizer.AddObjectiveValue(headerName, value);
        }

        /// <summary>
        /// Parses a questionnaire answer: invariant culture first ("3.5"), then the participant's culture ("3,5" on a
        /// German system), then a single decimal comma ("3,5" on an English system). False when none applies, so
        /// the caller submits NaN.
        /// </summary>
        private static bool TryParseQuestionnaireNumber(string rawValue, out float value)
        {
            value = float.NaN;
            string text = rawValue?.Trim();
            if (string.IsNullOrEmpty(text))
                return false;

            if (float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value) ||
                float.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out value))
            {
                return true;
            }

            int comma = text.IndexOf(',');
            if (comma >= 0 && comma == text.LastIndexOf(',') && text.IndexOf('.') < 0 &&
                float.TryParse(text.Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out value))
            {
                return true;
            }

            value = float.NaN;
            return false;
        }

        public void SetPriorSliderRatingHint(string questionKey, float sliderValue)
        {
            if (string.IsNullOrWhiteSpace(questionKey) || !BoObjectiveMath.IsFinite(sliderValue))
                return;

            _priorSliderRatingHints[questionKey] = sliderValue;
        }

        public bool TryGetPriorSliderRatingHint(string questionKey, out float sliderValue)
        {
            sliderValue = 0f;
            if (string.IsNullOrWhiteSpace(questionKey))
                return false;

            return _priorSliderRatingHints.TryGetValue(questionKey, out sliderValue) && BoObjectiveMath.IsFinite(sliderValue);
        }

        public void RemovePriorSliderRatingHint(string questionKey)
        {
            if (string.IsNullOrWhiteSpace(questionKey))
                return;

            _priorSliderRatingHints.Remove(questionKey);
        }

        public void ClearPriorLinearScaleRatingHints()
        {
            _priorLinearScaleRatingHints.Clear();
        }

        public void SetPriorLinearScaleRatingHint(string questionKey, string answerValue)
        {
            if (string.IsNullOrWhiteSpace(questionKey) || string.IsNullOrWhiteSpace(answerValue))
                return;

            _priorLinearScaleRatingHints[questionKey] = answerValue.Trim();
        }

        public bool TryGetPriorLinearScaleRatingHint(string questionKey, out string answerValue)
        {
            answerValue = string.Empty;
            if (string.IsNullOrWhiteSpace(questionKey))
                return false;

            if (!_priorLinearScaleRatingHints.TryGetValue(questionKey, out string storedValue))
                return false;

            if (string.IsNullOrWhiteSpace(storedValue))
                return false;

            answerValue = storedValue;
            return true;
        }

        public void RemovePriorLinearScaleRatingHint(string questionKey)
        {
            if (string.IsNullOrWhiteSpace(questionKey))
                return;

            _priorLinearScaleRatingHints.Remove(questionKey);
        }

        private string[] GetFinalDesignLogRootCandidates()
        {
            // The resolved log root the Python process writes to comes first; the other roots cover builds that
            // fell back to persistentDataPath and logs from earlier versions.
            var roots = new List<string>
            {
                LogDataFolderUtility.LogDataRoot,
                LogDataFolderUtility.StreamingAssetsLogRoot,
                LogDataFolderUtility.PersistentLogRoot
            };
            roots = roots.Distinct().ToList();

            // Legacy location from earlier versions / docs.
            string legacy = Path.Combine(
                Application.dataPath,
                "StreamingAssets",
                "BOData",
                "BayesianOptimization",
                "LogData"
            );

            string userFolder = NormalizeLogFolderToken(userId);
            string conditionFolder = NormalizeLogFolderToken(conditionId);
            var conditionRoots = roots.Select(root => Path.Combine(root, userFolder, conditionFolder)).ToList();

            // CABOP stores runs under dedicated subfolders to keep metrics/logs separate.
            List<string> CabopFolders(string mode)
            {
                var folders = conditionRoots.Select(root => Path.Combine(root, "CABOP", mode)).ToList();
                folders.AddRange(roots.Select(root => Path.Combine(root, "CABOP", mode)));
                folders.Add(Path.Combine(legacy, "CABOP", mode));
                return folders;
            }

            var ordered = new List<string>();
            if (optimizerBackend == OptimizerBackend.CABOP)
            {
                ordered.AddRange(CabopFolders(
                    cabopObjectiveMode == CabopObjectiveMode.SingleObjective ? "single" : "multi"
                ));
            }

            ordered.AddRange(conditionRoots);
            ordered.AddRange(roots);
            ordered.Add(legacy);
            ordered.AddRange(CabopFolders("single"));
            ordered.AddRange(CabopFolders("multi"));

            return ordered.Distinct().ToArray();
        }

        private static string NormalizeLogFolderToken(string value)
        {
            return LogDataFolderUtility.NormalizeLogFolderToken(value);
        }
        
        private bool IsPerfectRating()
        {
            var effectiveObjectives = BuildEffectiveObjectiveEntries(
                objectives,
                "perfect-rating evaluation",
                logWarnings: false
            );
            if (effectiveObjectives.Count == 0)
            {
                return false;
            }

            var hasValidObjective = false;
            foreach (var ob in effectiveObjectives)
            {
                if (ob == null || ob.value == null || ob.value.values == null)
                {
                    return false;
                }

                hasValidObjective = true;
                // Judge the value Python received: the mean of the finite sub-measures in the window (one
                // unanswered item must not make a perfect rating imperfect), with swapped bounds normalized.
                var aggregate = BoObjectiveMath.Aggregate(
                    ob.value.values,
                    ob.value.numberOfSubMeasures,
                    ob.value.lowerBound,
                    ob.value.upperBound
                );
                if (aggregate.UsedMidpointFallback)
                {
                    return false;
                }

                float best = ob.value.smallerIsBetter
                    ? Mathf.Min(ob.value.lowerBound, ob.value.upperBound)
                    : Mathf.Max(ob.value.lowerBound, ob.value.upperBound);
                if (ob.value.smallerIsBetter ? aggregate.Value > best : aggregate.Value < best)
                {
                    return false; // the rating is imperfect!
                }
            }

            if (!hasValidObjective)
            {
                return false;
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
                    // The previous perfect rating was more than one iteration ago, so this one
                    // starts a new streak.
                    perfectRatingStart = true;
                    perfectRating = false;
                    perfectRatingIteration = currentIteration;
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

        [System.Serializable]
        public class CabopCostTriplet
        {
            public float unchanged = 1f;
            public float swapped = 10f;
            public float acquired = 100f;

            public CabopCostTriplet() { }

            public CabopCostTriplet(float unchanged, float swapped, float acquired)
            {
                this.unchanged = unchanged;
                this.swapped = swapped;
                this.acquired = acquired;
            }
        }

        // ------------------
        // the context entries (contextual optimization / LCE-M GP):
        // ------------------
        [System.Serializable]
        public class ContextEntry
        {
            [Tooltip("Unique context identifier, e.g. a user, device, or environment name. " +
                     "Used as the 'Context' value in warm-start CSVs and observation logs.")]
            public string key = "";

            [Tooltip("Pre-computed context embedding vector (Manual source only). " +
                     "All contexts must use vectors of the same length.")]
            public List<float> embedding = new List<float>();

            [Tooltip("Image describing this context (Image source only). Relative paths are " +
                     "resolved against StreamingAssets/BOData/InitData.")]
            public string imagePath = "";

            public ContextEntry() { }

            public ContextEntry(string key)
            {
                this.key = key;
            }

            public ContextEntry(string key, List<float> embedding)
            {
                this.key = key;
                this.embedding = embedding ?? new List<float>();
            }
        }

        [System.Serializable]
        public class CabopGroupCostEntry
        {
            public string group = "default";
            public CabopCostTriplet cost = new CabopCostTriplet();
            public CabopCostTriplet actualCost = new CabopCostTriplet();

            public CabopGroupCostEntry() { }

            public CabopGroupCostEntry(string group, CabopCostTriplet cost, CabopCostTriplet actualCost)
            {
                this.group = group;
                this.cost = cost ?? new CabopCostTriplet();
                this.actualCost = actualCost ?? new CabopCostTriplet();
            }
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
            [Min(0f)] public float cabopWeight = 1.0f;

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
            public string cabopGroup = "default";
            [Range(0f, 1f)]
            [Tooltip("CABOP reuse tolerance as a fraction of this parameter's range (0.05 = 5% of Upper - Lower). " +
                     "A proposal this close to an earlier design reuses it (unchanged/swapped cost) instead of " +
                     "acquiring a new one.")]
            public float cabopTolerance = 0.05f;
            public List<float> cabopPrefabricatedValues = new List<float>();

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
