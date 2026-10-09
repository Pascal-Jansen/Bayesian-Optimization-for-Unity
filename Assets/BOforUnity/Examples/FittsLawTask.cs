using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using BOforUnity.Scripts;
using QuestionnaireToolkit.Scripts;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace BOforUnity.Examples
{
    public class FittsLawTask : MonoBehaviour
    {
        public enum TargetSelectionMode
        {
            AcrossCircle,
            SequentialStep,
            Random,
            CustomSequence
        }

        public enum SelectionEvent
        {
            [Tooltip("Register the selection when the pointer is pressed (ISO 9241-9).")]
            PointerDown,
            [Tooltip("Register the selection on release over the same target (earlier versions).")]
            PointerClick
        }

        [Serializable]
        public class TrialResult
        {
            public int trialIndex;
            public int targetIndex;
            public Vector2 targetPosition;
            /// <summary>Center of the previously selected target (start of the movement); unset for the first trial.</summary>
            public Vector2 fromTargetPosition;
            public bool hasFromTarget;
            public Vector2 clickPosition;
            public float clickTimeMs;
            public float centerDistancePixels;
            public int wrongClicksBeforeHit;
            public int wrongTargetClicksBeforeHit;
            public int playAreaMissClicksBeforeHit;
        }

        [Header("Task")]
        [Min(2)] public int targetCount = 12;
        [Min(1)] public int trialCount = 10;
        public bool startOnAwake = true;
        [Min(0f)] public float startDelaySeconds = 0.25f;
        [Tooltip("Debug aid: restart the running task with the restart key. Ignored once the task is complete " +
                 "(restarting then would send 0 ms / 0 px to the optimizer). Needs the legacy Input Manager.")]
        public bool restartWithKey = false;
        public KeyCode restartKey = KeyCode.R;

        [Header("Ready-to-run BO Example")]
        public bool ensureBoManagerInScene = true;
        public bool configureBoManagerForFittsTask = false;
        public bool waitForBoEvaluationStart = true;
        public bool startBoOptimizationAfterResults = true;
        public bool queueNextExternalSignalIteration = true;
        [Min(0)] public int boOptimizationIterations = 5;

        [Header("Design Parameters")]
        [Min(1f)] public float circleSizePixels = 72f;
        [Min(1f)] public float circleDistancePixels = 520f;
        public float movementDirectionDegrees = 0f;
        [Min(1f)] public float xFontSizePixels = 42f;
        [Range(0f, 1f)] public float buttonHue = 0.53f;
        [Range(0f, 1f)] public float buttonSaturation = 0.88f;
        [Range(0f, 1f)] public float buttonColorBrightness = 0.5f;

        [Header("BO Design Parameter Keys")]
        [Tooltip("When enabled, Fitts law design values are read only from matching keys in BoForUnityManager.parameters.")]
        public bool readDesignParametersFromBo = true;
        public bool boParameterValuesAreNormalized = false;
        [Tooltip("Applied only when this exact key exists in the BoForUnityManager parameter list.")]
        public string xFontSizeParameterKey = "x_font_size";
        [Tooltip("Applied only when this exact key exists in the BoForUnityManager parameter list.")]
        public string buttonSizeParameterKey = "button_size";
        [Tooltip("Applied only when this exact key exists in the BoForUnityManager parameter list.")]
        public string buttonDistanceParameterKey = "button_distance";
        [Tooltip("Applied only when this exact key exists in the BoForUnityManager parameter list.")]
        public string buttonHueParameterKey = "button_hue";
        [Tooltip("Applied only when this exact key exists in the BoForUnityManager parameter list.")]
        public string buttonSaturationParameterKey = "button_saturation";

        [Header("BO Auto-configuration Ranges")]
        [Tooltip("Used only when Configure Bo Manager For Fitts Task is enabled.")]
        public Vector2 xFontSizeRangePixels = new Vector2(18f, 64f);
        [Tooltip("Used only when Configure Bo Manager For Fitts Task is enabled.")]
        public Vector2 buttonSizeRangePixels = new Vector2(40f, 120f);
        [Tooltip("Used only when Configure Bo Manager For Fitts Task is enabled.")]
        public Vector2 buttonDistanceRangePixels = new Vector2(464f, 760f);
        [Tooltip("Used only when Configure Bo Manager For Fitts Task is enabled.")]
        public Vector2 buttonHueRange = new Vector2(0f, 1f);
        [Tooltip("Used only when Configure Bo Manager For Fitts Task is enabled.")]
        public Vector2 buttonSaturationRange = new Vector2(0f, 1f);

        [Header("Condition Controls")]
        [Tooltip("When enabled, the task samples a new design from the configured parameter ranges whenever BeginTask starts.")]
        public bool randomizeDesignParametersOnBegin = false;
        public bool useDeterministicRandomDesignSeed = false;
        public int randomDesignSeed = 98765;

        [Header("Target Order")]
        public TargetSelectionMode targetSelectionMode = TargetSelectionMode.AcrossCircle;
        [Min(0)] public int firstTargetIndex = 0;
        [Min(1)] public int targetStep = 1;
        public bool randomUsesSeed = true;
        public int randomSeed = 12345;
        public bool preventRandomImmediateRepeats = true;
        public List<int> customTargetSequence = new List<int>();

        [Header("Layout")]
        public bool fitPlayAreaToScreen = true;
        public Vector2 fixedPlayAreaSize = new Vector2(960f, 720f);
        public Vector2 playAreaPadding = new Vector2(120f, 96f);
        public Vector2 taskCenter = Vector2.zero;
        public bool clampTargetsInsidePlayArea = true;

        [Header("Canvas")]
        public Vector2 referenceResolution = new Vector2(1920f, 1080f);
        public Color backgroundColor = new Color(0.06f, 0.07f, 0.09f, 1f);
        public Color playAreaColor = new Color(0.08f, 0.1f, 0.13f, 1f);

        [Header("Target Appearance")]
        public Color targetColor = new Color(0.35f, 0.42f, 0.5f, 1f);
        public Color highlightedTargetColor = new Color(0.12f, 0.78f, 1f, 1f);
        public Color completedTargetColor = new Color(0.2f, 0.9f, 0.45f, 1f);
        public Color wrongTargetFlashColor = new Color(1f, 0.24f, 0.16f, 1f);
        public Color targetOutlineColor = new Color(1f, 1f, 1f, 0.45f);
        [Min(0f)] public float targetOutlineWidth = 0f;
        public bool showTargetLabels = false;
        public Color targetLabelColor = Color.white;
        [Min(1)] public int targetLabelFontSize = 22;
        public bool showTargetX = true;
        public bool showTargetXOnlyOnCurrentTarget = true;
        public Color targetXColor = Color.white;

        [Header("Interaction")]
        [Tooltip("When a target selection is registered. ISO 9241-9 uses the button press (Pointer Down).")]
        public SelectionEvent selectionEvent = SelectionEvent.PointerDown;
        public bool countWrongTargetClicks = true;
        public bool countPlayAreaMissClicks = true;
        public bool hideCursorDuringTask = false;
        [Min(0f)] public float wrongTargetFlashSeconds = 0f;

        [Header("Status Text")]
        public bool showStatusText = true;
        public string instructionText = "Click the highlighted target";
        public string progressFormat = "Trial {0} / {1}";
        public string completedText = "Task complete";
        public Color statusTextColor = Color.white;
        [Min(1)] public int statusFontSize = 32;

        [Header("Questionnaire Toolkit")]
        public QTQuestionnaireManager questionnaireToolkitManager;

        [Header("Questionnaire CSV Logging")]
        public bool logPerformanceToQuestionnaireCsv = true;
        public string questionnaireSpeedCsvHeader = "speed";
        public string questionnaireAccuracyCsvHeader = "accuracy";

        [Header("Design Objectives")]
        public float taskCompletionTimeMs;
        public float accuracyDistancePixels;

        public string QuestionnaireSpeedCsvValue => FormatCsvFloat(taskCompletionTimeMs);

        public string QuestionnaireAccuracyCsvValue => FormatCsvFloat(accuracyDistancePixels);

        [Header("BO Design Objectives")]
        public bool writeObjectivesToBo = true;
        public string aestheticsObjectiveKey = "aesthetics";
        public string speedObjectiveKey = "speed";
        public string accuracyObjectiveKey = "accuracy";
        public string usabilityObjectiveKey = "usability";

        [Header("Result Logging")]
        public bool logResultsToConsole = true;
        public bool writeResultsCsv = false;
        public string resultsFileName = "fitts_law_results.csv";
        public bool writeDetailedAppLogCsv = true;
        public string appSummaryLogFileName = "FittsLawAppLog.csv";
        public string appTrialLogFileName = "FittsLawTrialLog.csv";
        public List<TrialResult> trialResults = new List<TrialResult>();

        private readonly List<Image> _targetImages = new List<Image>();
        private readonly List<Button> _targetButtons = new List<Button>();
        private readonly List<RectTransform> _targetRects = new List<RectTransform>();
        private readonly List<TextMeshProUGUI> _targetXLabels = new List<TextMeshProUGUI>();
        private readonly List<int> _targetSequence = new List<int>();

        private Canvas _canvas;
        private RectTransform _playArea;
        private TextMeshProUGUI _statusText;
        private Sprite _circleSprite;
        private Texture2D _circleTexture;
        private int _currentTrial;
        private int _currentTargetIndex = -1;
        private int _wrongClicksThisTrial;
        private int _wrongTargetClicksThisTrial;
        private int _playAreaMissClicksThisTrial;
        private int _wrongClicksTotal;
        private int _wrongTargetClicksTotal;
        private int _playAreaMissClicksTotal;
        private int _correctClicks;
        private float _centerDistanceSum;
        private int _centerDistanceSamples;
        private float _targetShownAt;
        private float _taskStartedAt;
        private bool _taskRunning;
        private bool _taskComplete;
        private bool _cursorWasVisible;
        private bool _cursorStateCaptured;
        private bool _resultsFinalized;
        private int _randomDesignSampleIndex;
        private bool _manualLogContextActive;
        private int _manualLogIteration;
        private string _manualLogPhase = "manual";
        private string _manualLogUserId = "-1";
        private string _manualLogConditionId = "-1";
        private string _manualLogGroupId = "-1";
        private string _manualLogDirectory;
        private BoForUnityManager _runtimeDesignParameterSource;
        private bool _warnedRestartKeyUnavailable;
        private DesignSnapshot _roundDesign;
        private float _appliedCircleSizePixels;
        private float _appliedCircleDistancePixels;
        private float _appliedRingDiameterPixels;
        private bool _appliedLayoutAdjusted;

        /// <summary>The design of the current round as suggested (BO), sampled (random) or configured (static).</summary>
        private struct DesignSnapshot
        {
            public float XFontSizePixels;
            public float ButtonSizePixels;
            public float ButtonDistancePixels;
            public float ButtonHue;
            public float ButtonSaturation;
        }

        /// <summary>
        /// Marks a manager this task has already set up, so scene reloads (which re-run Awake against the
        /// persistent manager) do not overwrite the parameters the optimizer has just applied.
        /// </summary>
        private sealed class FittsBoSetupMarker : MonoBehaviour
        {
        }

        private void Awake()
        {
            if (ensureBoManagerInScene)
                EnsureBoManagerForExample();
        }

        private IEnumerator Start()
        {
            if (!startOnAwake)
                yield break;

            if (waitForBoEvaluationStart)
                yield return WaitForBoEvaluationStart();

            if (startDelaySeconds > 0f)
                yield return new WaitForSecondsRealtime(startDelaySeconds);

            yield return null;
            BeginTask();
        }

        private void Update()
        {
            // Only a running task can be restarted: after completion the questionnaire is open and a restart
            // would zero the measured time/accuracy that are about to be sent to the optimizer.
            if (!restartWithKey || !_taskRunning || _taskComplete)
                return;

#if ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKeyDown(restartKey))
                RestartTask();
#else
            if (!_warnedRestartKeyUnavailable)
            {
                _warnedRestartKeyUnavailable = true;
                Debug.LogWarning(
                    "FittsLawTask: Restart With Key needs the legacy Input Manager (Project Settings > Player > " +
                    "Active Input Handling = Input Manager or Both); the restart key is unavailable."
                );
            }
#endif
        }

        private void OnDisable()
        {
            RestoreCursor();
        }

        private void OnDestroy()
        {
            RestoreCursor();
            DestroyGeneratedAssets();
        }

        private void OnValidate()
        {
            targetCount = Mathf.Max(2, targetCount);
            trialCount = Mathf.Max(1, trialCount);
            targetStep = Mathf.Max(1, targetStep);
            circleSizePixels = Mathf.Max(1f, circleSizePixels);
            circleDistancePixels = Mathf.Max(1f, circleDistancePixels);
            xFontSizePixels = Mathf.Max(1f, xFontSizePixels);
            buttonHue = Mathf.Clamp01(buttonHue);
            buttonSaturation = Mathf.Clamp01(buttonSaturation);
            buttonColorBrightness = Mathf.Clamp01(buttonColorBrightness);
            fixedPlayAreaSize = new Vector2(Mathf.Max(1f, fixedPlayAreaSize.x), Mathf.Max(1f, fixedPlayAreaSize.y));
            referenceResolution = new Vector2(Mathf.Max(1f, referenceResolution.x), Mathf.Max(1f, referenceResolution.y));
            statusFontSize = Mathf.Max(1, statusFontSize);
            targetLabelFontSize = Mathf.Max(1, targetLabelFontSize);
            boOptimizationIterations = Mathf.Max(0, boOptimizationIterations);
            EnsureButtonDistanceRangeSupportsNoOverlap();
            EnsureQuestionnairePerformanceCsvItems();
        }

        private IEnumerator WaitForBoEvaluationStart()
        {
            BoForUnityManager manager = GetRuntimeDesignParameterSource();
            if (manager == null)
                yield break;

            while (manager != null &&
                   (!manager.initialized ||
                    !manager.simulationRunning ||
                    manager.optimizationRunning ||
                    manager.hasNewDesignParameterValues))
            {
                yield return null;
            }
        }

        public void BeginTask()
        {
            ClearGeneratedUi();
            if (randomizeDesignParametersOnBegin)
                ApplyRandomDesignParameters();
            else
            {
                ApplyBoDesignParameters();
                if (!readDesignParametersFromBo)
                    SyncRuntimeDesignParameterSourceWithCurrentDesign();
            }
            highlightedTargetColor = GetOptimizedButtonColor();
            _roundDesign = CaptureDesign();

            EnsureEventSystem();
            CreateCanvas();
            CreateTargets();
            WarnIfAppliedLayoutDiffersFromDesign();
            BuildTargetSequence();

            _currentTrial = 0;
            _wrongClicksThisTrial = 0;
            _wrongTargetClicksThisTrial = 0;
            _playAreaMissClicksThisTrial = 0;
            _wrongClicksTotal = 0;
            _wrongTargetClicksTotal = 0;
            _playAreaMissClicksTotal = 0;
            _correctClicks = 0;
            _centerDistanceSum = 0f;
            _centerDistanceSamples = 0;
            _taskRunning = true;
            _taskComplete = false;
            taskCompletionTimeMs = 0f;
            accuracyDistancePixels = 0f;
            _resultsFinalized = false;
            trialResults.Clear();

            if (hideCursorDuringTask)
            {
                _cursorWasVisible = Cursor.visible;
                _cursorStateCaptured = true;
                Cursor.visible = false;
            }

            _taskStartedAt = Time.realtimeSinceStartup;
            ShowCurrentTarget();
        }

        public void RestartTask()
        {
            if (_taskComplete && !_resultsFinalized)
            {
                Debug.LogWarning(
                    "FittsLawTask: restart ignored because the completed round's results have not been submitted yet."
                );
                return;
            }

            RestoreCursor();
            BeginTask();
        }

        public void SetManualLogContext(
            int iteration,
            string phase,
            string userId,
            string conditionId,
            string groupId)
        {
            _manualLogContextActive = true;
            _manualLogIteration = Mathf.Max(1, iteration);
            _manualLogPhase = string.IsNullOrWhiteSpace(phase) ? "manual" : phase.Trim();
            _manualLogUserId = GetContextValue(userId);
            _manualLogConditionId = GetContextValue(conditionId);
            _manualLogGroupId = GetContextValue(groupId);
            _manualLogDirectory = null;
        }

        public void ClearManualLogContext()
        {
            _manualLogContextActive = false;
            _manualLogIteration = 0;
            _manualLogPhase = "manual";
            _manualLogUserId = "-1";
            _manualLogConditionId = "-1";
            _manualLogGroupId = "-1";
            _manualLogDirectory = null;
        }

        public void SetRuntimeDesignParameterSource(BoForUnityManager manager)
        {
            _runtimeDesignParameterSource = manager;
        }

        private void EnsureBoManagerForExample()
        {
            BoForUnityManager manager = FindPreferredBoManager();
            if (manager == null)
            {
                if (!configureBoManagerForFittsTask)
                {
                    Debug.LogWarning(
                        "FittsLawTask did not find a BoForUnityManager in the scene. " +
                        "Add and configure a BoForUnityManager in the inspector, or enable Configure Bo Manager For Fitts Task to create one at runtime."
                    );
                    return;
                }

                GameObject managerObject = new GameObject("BOforUnityManager");
                managerObject.AddComponent<MainThreadDispatcher>();
                managerObject.AddComponent<Optimizer>();
                managerObject.AddComponent<PythonStarter>();
                managerObject.AddComponent<SocketNetwork>();
                manager = managerObject.AddComponent<BoForUnityManager>();
            }

            EnsureBoComponents(manager);
            EnsureEventSystem();
            EnsureBoControlUi(manager);

            // Awake runs again after every scene reload, against the persistent manager. Its parameters then hold
            // the design the optimizer has just applied, so the parameter/ID setup runs once per manager only:
            // repeating it replaced that design with the inspector defaults and reset the user/condition/group
            // IDs, so every round showed the default design while Python logged its suggestion.
            if (manager.TryGetComponent(out FittsBoSetupMarker _))
                return;

            if (manager.initialized || manager.simulationRunning)
            {
                Debug.LogWarning(
                    "FittsLawTask: the BoForUnityManager is already running; its parameters were not reconfigured."
                );
            }
            else
            {
                if (configureBoManagerForFittsTask)
                    ConfigureBoManagerForFittsTask(manager);

                EnsureNoOverlapButtonDistanceParameterBounds(manager);
            }

            manager.gameObject.AddComponent<FittsBoSetupMarker>();
        }

        private void EnsureBoComponents(BoForUnityManager manager)
        {
            if (manager == null)
                return;

            // TryGetComponent instead of "GetComponent() ?? AddComponent()": Unity's fake-null component is not
            // C# null, so the null-coalescing form never added a missing component.
            GameObject managerObject = manager.gameObject;
            manager.mainThreadDispatcher = GetOrAddComponent<MainThreadDispatcher>(managerObject);
            manager.optimizer = GetOrAddComponent<Optimizer>(managerObject);
            manager.pythonStarter = GetOrAddComponent<PythonStarter>(managerObject);
            manager.socketNetwork = GetOrAddComponent<SocketNetwork>(managerObject);
        }

        private static T GetOrAddComponent<T>(GameObject target) where T : Component
        {
            return target.TryGetComponent(out T existing) ? existing : target.AddComponent<T>();
        }

        private void EnsureBoControlUi(BoForUnityManager manager)
        {
            if (manager == null ||
                (manager.optimizerStatePanel != null &&
                 manager.outputText != null &&
                 manager.loadingObj != null &&
                 manager.nextButton != null))
            {
                return;
            }

            GameObject canvasObject = new GameObject("Fitts BO Control Canvas", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            int uiLayer = LayerMask.NameToLayer("UI");
            if (uiLayer >= 0)
                canvasObject.layer = uiLayer;

            canvasObject.transform.SetParent(manager.transform, false);

            Canvas canvas = canvasObject.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 100;

            CanvasScaler scaler = canvasObject.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = referenceResolution;
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 0.5f;

            RectTransform canvasRect = canvasObject.GetComponent<RectTransform>();

            GameObject panelObject = CreateUiObject("Optimizer State Panel", canvasRect);
            RectTransform panelRect = panelObject.GetComponent<RectTransform>();
            StretchToParent(panelRect);
            Image panelImage = panelObject.AddComponent<Image>();
            panelImage.color = new Color(0f, 0f, 0f, 0.72f);

            GameObject outputObject = CreateUiObject("Output Text", panelRect);
            RectTransform outputRect = outputObject.GetComponent<RectTransform>();
            outputRect.anchorMin = new Vector2(0.5f, 0.5f);
            outputRect.anchorMax = new Vector2(0.5f, 0.5f);
            outputRect.pivot = new Vector2(0.5f, 0.5f);
            outputRect.anchoredPosition = new Vector2(0f, 80f);
            outputRect.sizeDelta = new Vector2(1000f, 180f);
            TextMeshProUGUI outputText = outputObject.AddComponent<TextMeshProUGUI>();
            outputText.alignment = TextAlignmentOptions.Center;
            outputText.color = Color.white;
            outputText.fontSize = 36;
            outputText.text = "Starting Bayesian optimization...";
            outputText.raycastTarget = false;

            GameObject loadingObject = CreateUiObject("Loading Text", panelRect);
            RectTransform loadingRect = loadingObject.GetComponent<RectTransform>();
            loadingRect.anchorMin = new Vector2(0.5f, 0.5f);
            loadingRect.anchorMax = new Vector2(0.5f, 0.5f);
            loadingRect.pivot = new Vector2(0.5f, 0.5f);
            loadingRect.anchoredPosition = new Vector2(0f, -50f);
            loadingRect.sizeDelta = new Vector2(900f, 80f);
            TextMeshProUGUI loadingText = loadingObject.AddComponent<TextMeshProUGUI>();
            loadingText.alignment = TextAlignmentOptions.Center;
            loadingText.color = new Color(0.72f, 0.88f, 1f, 1f);
            loadingText.fontSize = 28;
            loadingText.text = "Loading optimizer...";
            loadingText.raycastTarget = false;

            GameObject buttonObject = CreateUiObject("Next Iteration Button", panelRect);
            RectTransform buttonRect = buttonObject.GetComponent<RectTransform>();
            buttonRect.anchorMin = new Vector2(0.5f, 0.5f);
            buttonRect.anchorMax = new Vector2(0.5f, 0.5f);
            buttonRect.pivot = new Vector2(0.5f, 0.5f);
            buttonRect.anchoredPosition = new Vector2(0f, -150f);
            buttonRect.sizeDelta = new Vector2(360f, 72f);
            Image buttonImage = buttonObject.AddComponent<Image>();
            buttonImage.color = new Color(0.12f, 0.48f, 0.78f, 1f);
            Button button = buttonObject.AddComponent<Button>();
            button.transition = Selectable.Transition.ColorTint;
            button.targetGraphic = buttonImage;
            button.onClick.AddListener(manager.RequestNextIteration);

            TextMeshProUGUI buttonLabel = CreateButtonLabel(buttonRect);
            buttonLabel.text = "Start Next Trial";
            buttonLabel.color = Color.white;
            buttonLabel.fontSize = 24;

            manager.optimizerStatePanel = panelObject;
            manager.outputText = outputText;
            manager.loadingObj = loadingObject;
            manager.nextButton = buttonObject;
            manager.welcomePanel = panelObject;
        }

        private void ConfigureBoManagerForFittsTask(BoForUnityManager manager)
        {
            float xFontSizeMin = Mathf.Min(xFontSizeRangePixels.x, xFontSizeRangePixels.y);
            float xFontSizeMax = Mathf.Max(xFontSizeRangePixels.x, xFontSizeRangePixels.y);
            float buttonSizeMin = Mathf.Min(buttonSizeRangePixels.x, buttonSizeRangePixels.y);
            float buttonSizeMax = Mathf.Max(buttonSizeRangePixels.x, buttonSizeRangePixels.y);
            float buttonDistanceMin = Mathf.Min(buttonDistanceRangePixels.x, buttonDistanceRangePixels.y);
            float buttonDistanceMax = Mathf.Max(buttonDistanceRangePixels.x, buttonDistanceRangePixels.y);
            float buttonHueMin = Mathf.Min(buttonHueRange.x, buttonHueRange.y);
            float buttonHueMax = Mathf.Max(buttonHueRange.x, buttonHueRange.y);
            float buttonSaturationMin = Mathf.Min(buttonSaturationRange.x, buttonSaturationRange.y);
            float buttonSaturationMax = Mathf.Max(buttonSaturationRange.x, buttonSaturationRange.y);
            buttonDistanceMin = Mathf.Max(
                buttonDistanceMin,
                GetMinimumCircleDistanceForNoOverlap(buttonSizeMax, targetCount)
            );
            buttonDistanceMax = Mathf.Max(buttonDistanceMax, buttonDistanceMin);

            manager.parameters = new List<ParameterEntry>
            {
                new ParameterEntry(
                    xFontSizeParameterKey,
                    new ParameterArgs(xFontSizeMin, xFontSizeMax)
                    {
                        Value = Mathf.Clamp(xFontSizePixels, xFontSizeMin, xFontSizeMax),
                        cabopGroup = "x_font_size",
                        cabopTolerance = CabopToleranceFraction(2f, xFontSizeMin, xFontSizeMax)
                    }),
                new ParameterEntry(
                    buttonSizeParameterKey,
                    new ParameterArgs(buttonSizeMin, buttonSizeMax)
                    {
                        Value = Mathf.Clamp(circleSizePixels, buttonSizeMin, buttonSizeMax),
                        cabopGroup = "button_size",
                        cabopTolerance = CabopToleranceFraction(2f, buttonSizeMin, buttonSizeMax)
                    }),
                new ParameterEntry(
                    buttonDistanceParameterKey,
                    new ParameterArgs(buttonDistanceMin, buttonDistanceMax)
                    {
                        Value = Mathf.Clamp(circleDistancePixels, buttonDistanceMin, buttonDistanceMax),
                        cabopGroup = "button_distance",
                        cabopTolerance = CabopToleranceFraction(5f, buttonDistanceMin, buttonDistanceMax)
                    }),
                new ParameterEntry(
                    buttonHueParameterKey,
                    new ParameterArgs(buttonHueMin, buttonHueMax)
                    {
                        Value = Mathf.Clamp(buttonHue, buttonHueMin, buttonHueMax),
                        cabopGroup = "button_color",
                        cabopTolerance = 0.02f
                    }),
                new ParameterEntry(
                    buttonSaturationParameterKey,
                    new ParameterArgs(buttonSaturationMin, buttonSaturationMax)
                    {
                        Value = Mathf.Clamp(buttonSaturation, buttonSaturationMin, buttonSaturationMax),
                        cabopGroup = "button_color",
                        cabopTolerance = 0.02f
                    })
            };
            SyncDesignParameterSourceWithCurrentDesign(manager, true);

            manager.objectives = new List<ObjectiveEntry>
            {
                new ObjectiveEntry(
                    aestheticsObjectiveKey,
                    new ObjectiveArgs(0f, 100f, false, 1)),
                new ObjectiveEntry(
                    speedObjectiveKey,
                    new ObjectiveArgs(0f, 30000f, true, 1)),
                new ObjectiveEntry(
                    accuracyObjectiveKey,
                    new ObjectiveArgs(0f, 1300f, true, 1)),
                new ObjectiveEntry(
                    usabilityObjectiveKey,
                    new ObjectiveArgs(0f, 100f, false, 2))
            };

            manager.optimizerBackend = BoForUnityManager.OptimizerBackend.BoTorch;
            manager.iterationAdvanceMode = BoForUnityManager.IterationAdvanceMode.ExternalSignal;
            manager.reloadSceneOnIterationAdvance = true;
            manager.warmStart = false;
            manager.numSamplingIterations = BoForUnityManager.ComputeRecommendedSamplingIterations(manager.parameters.Count);
            manager.numOptimizationIterations = boOptimizationIterations;
            manager.totalIterations = manager.numSamplingIterations + manager.numOptimizationIterations;
            manager.localPython = false;
            manager.pythonPath = string.Empty;
            manager.userId = "-1";
            manager.conditionId = "-1";
            manager.groupId = "-1";
        }

        /// <summary>
        /// CABOP reads the reuse tolerance as a fraction of the parameter's range ([0, 1]); converts a tolerance
        /// in parameter units (pixels) to that fraction. A range without width gets 0.
        /// </summary>
        private static float CabopToleranceFraction(float toleranceUnits, float min, float max)
        {
            float width = max - min;
            if (!(width > 0f) || float.IsInfinity(width))
                return 0f;

            return Mathf.Clamp01(toleranceUnits / width);
        }

        private void ApplyRandomDesignParameters()
        {
            System.Random seededRandom = null;
            if (useDeterministicRandomDesignSeed)
                seededRandom = new System.Random(randomDesignSeed + _randomDesignSampleIndex);

            _randomDesignSampleIndex++;

            BoForUnityManager parameterSource = GetRuntimeDesignParameterSource();
            bool useConfiguredParameterList = HasConfiguredParameterList(parameterSource);
            bool buttonColorChanged = false;

            if (TrySampleRandomParameter(parameterSource, xFontSizeParameterKey, seededRandom, out float sampledXFontSize))
                xFontSizePixels = sampledXFontSize;
            else if (!useConfiguredParameterList)
                xFontSizePixels = SampleRange(xFontSizeRangePixels, seededRandom);

            if (TrySampleRandomParameter(parameterSource, buttonSizeParameterKey, seededRandom, out float sampledButtonSize))
                circleSizePixels = sampledButtonSize;
            else if (!useConfiguredParameterList)
                circleSizePixels = SampleRange(buttonSizeRangePixels, seededRandom);

            if (TrySampleRandomDistanceParameter(parameterSource, seededRandom, out float sampledButtonDistance))
                circleDistancePixels = sampledButtonDistance;
            else if (!useConfiguredParameterList)
                circleDistancePixels = SampleRange(
                    GetNoOverlapDistanceRangeForButtonSize(circleSizePixels),
                    seededRandom
                );

            // Hue is stored as sampled (1.0 stays 1.0); the color wraps it (GetOptimizedButtonColor).
            if (TrySampleRandomParameter(parameterSource, buttonHueParameterKey, seededRandom, out float sampledButtonHue))
            {
                buttonHue = sampledButtonHue;
                buttonColorChanged = true;
            }
            else if (!useConfiguredParameterList)
            {
                buttonHue = SampleRange(buttonHueRange, seededRandom);
                buttonColorChanged = true;
            }

            if (TrySampleRandomParameter(parameterSource, buttonSaturationParameterKey, seededRandom, out float sampledButtonSaturation))
            {
                buttonSaturation = Mathf.Clamp01(sampledButtonSaturation);
                buttonColorChanged = true;
            }
            else if (!useConfiguredParameterList)
            {
                buttonSaturation = Mathf.Clamp01(SampleRange(buttonSaturationRange, seededRandom));
                buttonColorChanged = true;
            }

            if (buttonColorChanged)
                highlightedTargetColor = GetOptimizedButtonColor();

            EnforceNoOverlapDistanceForCurrentDesign();
            SyncRuntimeDesignParameterSourceWithCurrentDesign();
        }

        private static float SampleRange(Vector2 range, System.Random seededRandom)
        {
            float min = Mathf.Min(range.x, range.y);
            float max = Mathf.Max(range.x, range.y);
            if (seededRandom != null)
                return Mathf.Lerp(min, max, (float)seededRandom.NextDouble());

            return UnityEngine.Random.Range(min, max);
        }

        private Vector2 GetNoOverlapDistanceRangeForButtonSize(float buttonSizePixels)
        {
            float min = Mathf.Min(buttonDistanceRangePixels.x, buttonDistanceRangePixels.y);
            float max = Mathf.Max(buttonDistanceRangePixels.x, buttonDistanceRangePixels.y);
            min = Mathf.Max(min, GetMinimumCircleDistanceForNoOverlap(buttonSizePixels, targetCount));
            max = Mathf.Max(max, min);
            return new Vector2(min, max);
        }

        private BoForUnityManager GetRuntimeDesignParameterSource()
        {
            return _runtimeDesignParameterSource != null
                ? _runtimeDesignParameterSource
                : FindPreferredBoManager();
        }

        private static bool HasConfiguredParameterList(BoForUnityManager manager)
        {
            return manager != null && manager.parameters != null && manager.parameters.Count > 0;
        }

        private bool TrySampleRandomParameter(
            BoForUnityManager manager,
            string key,
            System.Random seededRandom,
            out float sampledValue)
        {
            sampledValue = 0f;
            if (!TryFindBoParameter(manager, key, out ParameterArgs parameter))
                return false;

            sampledValue = SampleParameterBounds(parameter, seededRandom);
            SetBoParameterValue(parameter, sampledValue);
            return true;
        }

        private bool TrySampleRandomDistanceParameter(
            BoForUnityManager manager,
            System.Random seededRandom,
            out float sampledValue)
        {
            sampledValue = 0f;
            if (!TryFindBoParameter(manager, buttonDistanceParameterKey, out ParameterArgs parameter))
                return false;

            float min = Mathf.Min(parameter.lowerBound, parameter.upperBound);
            float max = Mathf.Max(parameter.lowerBound, parameter.upperBound);
            min = Mathf.Max(min, GetMinimumCircleDistanceForNoOverlap(circleSizePixels, targetCount));
            max = Mathf.Max(max, min);
            sampledValue = SampleRange(new Vector2(min, max), seededRandom);
            SetBoParameterValue(parameter, sampledValue);
            return true;
        }

        private static float SampleParameterBounds(ParameterArgs parameter, System.Random seededRandom)
        {
            return SampleRange(new Vector2(parameter.lowerBound, parameter.upperBound), seededRandom);
        }

        private void SyncRuntimeDesignParameterSourceWithCurrentDesign()
        {
            SyncDesignParameterSourceWithCurrentDesign(GetRuntimeDesignParameterSource());
        }

        private void SyncDesignParameterSourceWithCurrentDesign(BoForUnityManager parameterSource, bool clampToParameterBounds = false)
        {
            if (!HasConfiguredParameterList(parameterSource))
                return;

            TrySetBoParameterValue(parameterSource, xFontSizeParameterKey, xFontSizePixels, clampToParameterBounds);
            TrySetBoParameterValue(parameterSource, buttonSizeParameterKey, circleSizePixels, clampToParameterBounds);
            TrySetBoParameterValue(parameterSource, buttonDistanceParameterKey, circleDistancePixels, clampToParameterBounds);
            TrySetBoParameterValue(parameterSource, buttonHueParameterKey, buttonHue, clampToParameterBounds);
            TrySetBoParameterValue(parameterSource, buttonSaturationParameterKey, buttonSaturation, clampToParameterBounds);
        }

        private void TrySetBoParameterValue(
            BoForUnityManager manager,
            string key,
            float value,
            bool clampToParameterBounds = false)
        {
            if (!IsFinite(value) || !TryFindBoParameter(manager, key, out ParameterArgs parameter))
                return;

            SetBoParameterValue(parameter, value, clampToParameterBounds);
        }

        private void SetBoParameterValue(
            ParameterArgs parameter,
            float rawValue,
            bool clampToParameterBounds = false)
        {
            if (parameter == null)
                return;

            float min = Mathf.Min(parameter.lowerBound, parameter.upperBound);
            float max = Mathf.Max(parameter.lowerBound, parameter.upperBound);
            if (clampToParameterBounds)
                rawValue = Mathf.Clamp(rawValue, min, max);

            if (!boParameterValuesAreNormalized)
            {
                parameter.Value = rawValue;
                return;
            }

            parameter.Value = Mathf.Approximately(min, max)
                ? 0f
                : Mathf.InverseLerp(min, max, rawValue);
        }

        private void ApplyBoDesignParameters()
        {
            if (!readDesignParametersFromBo)
                return;

            if (TryFindBoParameter(xFontSizeParameterKey, out ParameterArgs xFontSizeParameter))
                xFontSizePixels = MapBoParameterValue(xFontSizeParameter);

            if (TryFindBoParameter(buttonSizeParameterKey, out ParameterArgs buttonSizeParameter))
                circleSizePixels = MapBoParameterValue(buttonSizeParameter);

            if (TryFindBoParameter(buttonDistanceParameterKey, out ParameterArgs distanceParameter))
                circleDistancePixels = MapBoParameterValue(distanceParameter);

            // Stored as suggested; GetOptimizedButtonColor wraps the hue and clamps the saturation. (Wrapping
            // here turned a suggested hue of 1.0 into 0.)
            bool buttonColorChanged = false;
            if (TryFindBoParameter(buttonHueParameterKey, out ParameterArgs hueParameter))
            {
                buttonHue = MapBoParameterValue(hueParameter);
                buttonColorChanged = true;
            }

            if (TryFindBoParameter(buttonSaturationParameterKey, out ParameterArgs saturationParameter))
            {
                buttonSaturation = MapBoParameterValue(saturationParameter);
                buttonColorChanged = true;
            }

            if (buttonColorChanged)
                highlightedTargetColor = GetOptimizedButtonColor();

            // Nothing is written back to the manager: its parameters must keep the design Python suggested
            // (and logs the observation under). Layout safety adjustments are applied to the rendered layout
            // only and recorded in the Applied* telemetry columns.
        }

        private float MapBoParameterValue(ParameterArgs parameter)
        {
            float min = Mathf.Min(parameter.lowerBound, parameter.upperBound);
            float max = Mathf.Max(parameter.lowerBound, parameter.upperBound);
            if (boParameterValuesAreNormalized)
                return Mathf.Lerp(min, max, Mathf.Clamp01(parameter.Value));

            return Mathf.Clamp(parameter.Value, min, max);
        }

        private bool TryFindBoParameter(string key, out ParameterArgs value)
        {
            value = null;
            BoForUnityManager manager = GetRuntimeDesignParameterSource();
            return TryFindBoParameter(manager, key, out value);
        }

        private static bool TryFindBoParameter(BoForUnityManager manager, string key, out ParameterArgs value)
        {
            value = null;
            if (manager == null || manager.parameters == null)
                return false;

            if (string.IsNullOrWhiteSpace(key))
                return false;

            for (int i = 0; i < manager.parameters.Count; i++)
            {
                ParameterEntry parameter = manager.parameters[i];
                if (parameter == null || parameter.value == null)
                    continue;

                if (string.Equals(parameter.key, key, StringComparison.Ordinal))
                {
                    value = parameter.value;
                    return true;
                }
            }

            return false;
        }

        private void EnsureButtonDistanceRangeSupportsNoOverlap()
        {
            float buttonSizeMax = Mathf.Max(
                1f,
                Mathf.Max(buttonSizeRangePixels.x, buttonSizeRangePixels.y)
            );
            float minRequiredDistance = GetMinimumCircleDistanceForNoOverlap(buttonSizeMax, targetCount);
            float min = Mathf.Min(buttonDistanceRangePixels.x, buttonDistanceRangePixels.y);
            float max = Mathf.Max(buttonDistanceRangePixels.x, buttonDistanceRangePixels.y);
            min = Mathf.Max(min, minRequiredDistance);
            max = Mathf.Max(max, min);
            buttonDistanceRangePixels = new Vector2(min, max);
        }

        private void EnsureNoOverlapButtonDistanceParameterBounds(BoForUnityManager manager)
        {
            if (manager == null)
                return;

            float maxButtonSize = Mathf.Max(1f, circleSizePixels);
            if (TryFindBoParameter(manager, buttonSizeParameterKey, out ParameterArgs sizeParameter))
            {
                maxButtonSize = Mathf.Max(
                    maxButtonSize,
                    Mathf.Max(sizeParameter.lowerBound, sizeParameter.upperBound)
                );
            }
            else
            {
                maxButtonSize = Mathf.Max(
                    maxButtonSize,
                    Mathf.Max(buttonSizeRangePixels.x, buttonSizeRangePixels.y)
                );
            }

            float minRequiredDistance = GetMinimumCircleDistanceForNoOverlap(maxButtonSize, targetCount);
            if (!TryFindBoParameter(manager, buttonDistanceParameterKey, out ParameterArgs distanceParameter))
                return;

            float lower = Mathf.Min(distanceParameter.lowerBound, distanceParameter.upperBound);
            float upper = Mathf.Max(distanceParameter.lowerBound, distanceParameter.upperBound);
            float currentDistance = MapBoParameterValue(distanceParameter);
            lower = Mathf.Max(lower, minRequiredDistance);
            upper = Mathf.Max(upper, lower);

            if (!Mathf.Approximately(distanceParameter.lowerBound, lower) ||
                !Mathf.Approximately(distanceParameter.upperBound, upper))
            {
                Debug.LogWarning(
                    "FittsLawTask: adjusted button_distance bounds to prevent target overlap. " +
                    "New bounds: [" +
                    lower.ToString("0.###", CultureInfo.InvariantCulture) + ", " +
                    upper.ToString("0.###", CultureInfo.InvariantCulture) + "]."
                );
            }

            distanceParameter.lowerBound = lower;
            distanceParameter.upperBound = upper;
            SetBoParameterValue(distanceParameter, Mathf.Clamp(currentDistance, lower, upper));
        }

        private void EnforceNoOverlapDistanceForCurrentDesign()
        {
            float minDistance = GetMinimumCircleDistanceForNoOverlap(circleSizePixels, targetCount);
            if (circleDistancePixels >= minDistance)
                return;

            Debug.LogWarning(
                "FittsLawTask: increased button_distance from " +
                circleDistancePixels.ToString("0.###", CultureInfo.InvariantCulture) + " to " +
                minDistance.ToString("0.###", CultureInfo.InvariantCulture) +
                " to prevent target overlap."
            );
            circleDistancePixels = minDistance;
        }

        private void ApplyLayoutSafetyConstraints()
        {
            float safeSize = Mathf.Max(1f, circleSizePixels);
            float safeDistance = Mathf.Max(
                Mathf.Max(1f, circleDistancePixels),
                GetMinimumCircleDistanceForNoOverlap(safeSize, targetCount)
            );

            if (clampTargetsInsidePlayArea && _playArea != null)
            {
                Vector2 halfSize = _playArea.rect.size * 0.5f;
                float centerRadiusLimit = Mathf.Min(
                    Mathf.Max(0f, halfSize.x - Mathf.Abs(taskCenter.x)),
                    Mathf.Max(0f, halfSize.y - Mathf.Abs(taskCenter.y))
                );

                if (centerRadiusLimit > 0f)
                {
                    float maxFeasibleSize = GetMaximumCircleSizeForNoOverlap(centerRadiusLimit, targetCount);
                    if (safeSize > maxFeasibleSize)
                    {
                        safeSize = Mathf.Max(1f, maxFeasibleSize);
                        safeDistance = Mathf.Max(
                            Mathf.Max(1f, circleDistancePixels),
                            GetMinimumCircleDistanceForNoOverlap(safeSize, targetCount)
                        );
                    }

                    float minDistance = GetMinimumCircleDistanceForNoOverlap(safeSize, targetCount);
                    float maxDistance = Mathf.Max(0f, 2f * (centerRadiusLimit - safeSize * 0.5f));
                    if (maxDistance > 0f)
                    {
                        safeDistance = minDistance <= maxDistance
                            ? Mathf.Clamp(safeDistance, minDistance, maxDistance)
                            : maxDistance;
                    }
                }
            }

            // The design fields (and the manager's parameters) keep the suggested design; only the rendered
            // layout uses the safe values.
            _appliedCircleSizePixels = safeSize;
            _appliedCircleDistancePixels = safeDistance;
        }

        private DesignSnapshot CaptureDesign()
        {
            return new DesignSnapshot
            {
                XFontSizePixels = xFontSizePixels,
                ButtonSizePixels = circleSizePixels,
                ButtonDistancePixels = circleDistancePixels,
                ButtonHue = buttonHue,
                ButtonSaturation = buttonSaturation
            };
        }

        private void WarnIfAppliedLayoutDiffersFromDesign()
        {
            _appliedLayoutAdjusted =
                !NearlyEqual(_appliedCircleSizePixels, _roundDesign.ButtonSizePixels) ||
                !NearlyEqual(_appliedRingDiameterPixels, _roundDesign.ButtonDistancePixels);
            if (!_appliedLayoutAdjusted)
                return;

            Debug.LogWarning(
                "FittsLawTask: the play area cannot show this design as specified, so the rendered layout differs " +
                "from it: button_size " + FormatCsvFloat(_roundDesign.ButtonSizePixels) + " -> " +
                FormatCsvFloat(_appliedCircleSizePixels) + " px, button_distance " +
                FormatCsvFloat(_roundDesign.ButtonDistancePixels) + " -> " + FormatCsvFloat(_appliedRingDiameterPixels) +
                " px. The optimizer attributes this round to the specified design; the rendered values are logged " +
                "as AppliedButtonSizePixels/AppliedButtonDistancePixels. Use a larger window or narrower bounds."
            );
        }

        private static bool NearlyEqual(float a, float b)
        {
            return Mathf.Abs(a - b) <= 1e-3f * Mathf.Max(1f, Mathf.Max(Mathf.Abs(a), Mathf.Abs(b)));
        }

        private Color GetOptimizedButtonColor()
        {
            return Color.HSVToRGB(Mathf.Repeat(buttonHue, 1f), Mathf.Clamp01(buttonSaturation), Mathf.Clamp01(buttonColorBrightness));
        }

        private void CreateCanvas()
        {
            GameObject canvasObject = new GameObject("Fitts Law Task Canvas", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            int uiLayer = LayerMask.NameToLayer("UI");
            if (uiLayer >= 0)
                canvasObject.layer = uiLayer;

            canvasObject.transform.SetParent(transform, false);

            _canvas = canvasObject.GetComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;

            CanvasScaler scaler = canvasObject.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = referenceResolution;
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 0.5f;

            RectTransform canvasRect = canvasObject.GetComponent<RectTransform>();

            GameObject background = CreateUiObject("Background", canvasRect);
            RectTransform backgroundRect = background.GetComponent<RectTransform>();
            StretchToParent(backgroundRect);
            Image backgroundImage = background.AddComponent<Image>();
            backgroundImage.color = backgroundColor;
            backgroundImage.raycastTarget = false;

            GameObject playAreaObject = CreateUiObject("Play Area", canvasRect);
            _playArea = playAreaObject.GetComponent<RectTransform>();
            if (fitPlayAreaToScreen)
            {
                StretchToParent(_playArea);
                _playArea.offsetMin = playAreaPadding;
                _playArea.offsetMax = -playAreaPadding;
            }
            else
            {
                _playArea.anchorMin = new Vector2(0.5f, 0.5f);
                _playArea.anchorMax = new Vector2(0.5f, 0.5f);
                _playArea.pivot = new Vector2(0.5f, 0.5f);
                _playArea.anchoredPosition = Vector2.zero;
                _playArea.sizeDelta = fixedPlayAreaSize;
            }

            Image playAreaImage = playAreaObject.AddComponent<Image>();
            playAreaImage.color = playAreaColor;
            playAreaImage.raycastTarget = countPlayAreaMissClicks;
            if (countPlayAreaMissClicks)
            {
                PlayAreaPointerClickRelay playAreaRelay = playAreaObject.AddComponent<PlayAreaPointerClickRelay>();
                playAreaRelay.Initialize(this);
            }

            if (showStatusText)
                CreateStatusText(canvasRect);

            Canvas.ForceUpdateCanvases();
        }

        private void CreateStatusText(RectTransform canvasRect)
        {
            GameObject statusObject = CreateUiObject("Status Text", canvasRect);
            RectTransform statusRect = statusObject.GetComponent<RectTransform>();
            statusRect.anchorMin = new Vector2(0.5f, 1f);
            statusRect.anchorMax = new Vector2(0.5f, 1f);
            statusRect.pivot = new Vector2(0.5f, 1f);
            statusRect.anchoredPosition = new Vector2(0f, -28f);
            statusRect.sizeDelta = new Vector2(900f, 92f);

            _statusText = statusObject.AddComponent<TextMeshProUGUI>();
            _statusText.alignment = TextAlignmentOptions.Center;
            _statusText.color = statusTextColor;
            _statusText.fontSize = statusFontSize;
            _statusText.raycastTarget = false;
        }

        private void CreateTargets()
        {
            _targetImages.Clear();
            _targetButtons.Clear();
            _targetRects.Clear();
            _targetXLabels.Clear();

            ApplyLayoutSafetyConstraints();
            float effectiveRadius = GetEffectiveRingRadius();
            _appliedRingDiameterPixels = effectiveRadius * 2f;
            for (int i = 0; i < targetCount; i++)
            {
                GameObject targetObject = CreateUiObject("Target " + (i + 1).ToString(CultureInfo.InvariantCulture), _playArea);
                RectTransform targetRect = targetObject.GetComponent<RectTransform>();
                targetRect.anchorMin = new Vector2(0.5f, 0.5f);
                targetRect.anchorMax = new Vector2(0.5f, 0.5f);
                targetRect.pivot = new Vector2(0.5f, 0.5f);
                targetRect.sizeDelta = new Vector2(_appliedCircleSizePixels, _appliedCircleSizePixels);

                float angle = (movementDirectionDegrees + (360f * i / targetCount)) * Mathf.Deg2Rad;
                Vector2 position = taskCenter + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * effectiveRadius;
                targetRect.anchoredPosition = position;

                Image image = targetObject.AddComponent<Image>();
                image.sprite = GetCircleSprite();
                image.color = targetColor;
                image.raycastTarget = true;
                image.alphaHitTestMinimumThreshold = 0.1f;

                if (targetOutlineWidth > 0f)
                {
                    Outline outline = targetObject.AddComponent<Outline>();
                    outline.effectColor = targetOutlineColor;
                    outline.effectDistance = new Vector2(targetOutlineWidth, -targetOutlineWidth);
                }

                Button button = targetObject.AddComponent<Button>();
                button.transition = Selectable.Transition.None;
                TargetPointerClickRelay clickRelay = targetObject.AddComponent<TargetPointerClickRelay>();
                clickRelay.Initialize(this, i);

                if (showTargetLabels)
                    CreateTargetLabel(targetObject.transform, i + 1);

                if (showTargetX)
                    _targetXLabels.Add(CreateTargetXLabel(targetObject.transform));

                _targetImages.Add(image);
                _targetButtons.Add(button);
                _targetRects.Add(targetRect);
            }
        }

        private TextMeshProUGUI CreateTargetXLabel(Transform parent)
        {
            GameObject labelObject = CreateUiObject("Target X", parent as RectTransform);
            RectTransform labelRect = labelObject.GetComponent<RectTransform>();
            StretchToParent(labelRect);

            TextMeshProUGUI label = labelObject.AddComponent<TextMeshProUGUI>();
            label.text = "X";
            label.alignment = TextAlignmentOptions.Center;
            label.color = targetXColor;
            label.fontSize = xFontSizePixels;
            label.fontStyle = FontStyles.Bold;
            label.enableWordWrapping = false;
            label.raycastTarget = false;
            return label;
        }

        private void CreateTargetLabel(Transform parent, int labelIndex)
        {
            GameObject labelObject = CreateUiObject("Label", parent as RectTransform);
            RectTransform labelRect = labelObject.GetComponent<RectTransform>();
            StretchToParent(labelRect);

            TextMeshProUGUI label = labelObject.AddComponent<TextMeshProUGUI>();
            label.text = labelIndex.ToString(CultureInfo.InvariantCulture);
            label.alignment = TextAlignmentOptions.Center;
            label.color = targetLabelColor;
            label.fontSize = targetLabelFontSize;
            label.raycastTarget = false;
        }

        private void BuildTargetSequence()
        {
            _targetSequence.Clear();
            int start = PositiveModulo(firstTargetIndex, targetCount);

            if (targetSelectionMode == TargetSelectionMode.CustomSequence && customTargetSequence.Count > 0)
            {
                for (int i = 0; i < trialCount; i++)
                {
                    int sequenceValue = customTargetSequence[i % customTargetSequence.Count];
                    _targetSequence.Add(PositiveModulo(sequenceValue, targetCount));
                }
                return;
            }

            if (targetSelectionMode == TargetSelectionMode.Random)
            {
                System.Random random = randomUsesSeed ? new System.Random(randomSeed) : new System.Random();
                int previous = -1;
                for (int i = 0; i < trialCount; i++)
                {
                    int next = random.Next(0, targetCount);
                    if (preventRandomImmediateRepeats && targetCount > 1)
                    {
                        int guard = 0;
                        while (next == previous && guard < 12)
                        {
                            next = random.Next(0, targetCount);
                            guard++;
                        }
                    }

                    _targetSequence.Add(next);
                    previous = next;
                }
                return;
            }

            if (targetSelectionMode == TargetSelectionMode.AcrossCircle)
            {
                // Odd target counts step by (n + 1) / 2 (ISO 9241-9) so every movement has the same amplitude;
                // the previous pairing alternated, e.g. 0.993 · D and 0.935 · D for n = 13.
                FittsLawMath.BuildAcrossCircleSequence(targetCount, start, trialCount, _targetSequence);
                return;
            }

            for (int i = 0; i < trialCount; i++)
                _targetSequence.Add(PositiveModulo(start + i * targetStep, targetCount));
        }

        private void ShowCurrentTarget()
        {
            if (_currentTrial >= trialCount)
            {
                CompleteTask();
                return;
            }

            _currentTargetIndex = _targetSequence[_currentTrial];
            _wrongClicksThisTrial = 0;
            _wrongTargetClicksThisTrial = 0;
            _playAreaMissClicksThisTrial = 0;
            _targetShownAt = Time.realtimeSinceStartup;
            UpdateTargetVisuals();
            UpdateStatusText();
        }

        private void HandleTargetPointerEvent(int clickedTargetIndex, PointerEventData eventData, SelectionEvent eventType)
        {
            if (eventType != selectionEvent || !_taskRunning || _taskComplete)
                return;

            if (clickedTargetIndex != _currentTargetIndex)
            {
                if (countWrongTargetClicks)
                {
                    // A wrong-target click enters the accuracy objective only when it is counted.
                    RecordCenterDistanceFromPointer(eventData);
                    _wrongClicksThisTrial++;
                    _wrongTargetClicksThisTrial++;
                    _wrongClicksTotal++;
                    _wrongTargetClicksTotal++;
                }

                if (wrongTargetFlashSeconds > 0f)
                    StartCoroutine(FlashWrongTarget(clickedTargetIndex));

                return;
            }

            float centerDistancePixels = RecordCenterDistanceFromPointer(eventData);
            _correctClicks++;
            AdvanceAfterHit(clickedTargetIndex, eventData, centerDistancePixels);
        }

        private void HandlePlayAreaPointerEvent(PointerEventData eventData, SelectionEvent eventType)
        {
            if (eventType != selectionEvent || !_taskRunning || _taskComplete || !countPlayAreaMissClicks)
                return;

            RecordCenterDistanceFromPointer(eventData);
            _wrongClicksThisTrial++;
            _playAreaMissClicksThisTrial++;
            _wrongClicksTotal++;
            _playAreaMissClicksTotal++;
        }

        private void AdvanceAfterHit(int clickedTargetIndex, PointerEventData eventData, float centerDistancePixels)
        {
            float clickTimeMs = (Time.realtimeSinceStartup - _targetShownAt) * 1000f;
            RectTransform targetRect = _targetRects[clickedTargetIndex];
            Vector2 clickPosition = targetRect.anchoredPosition;
            if (TryGetPlayAreaLocalPoint(eventData, out Vector2 localClickPosition))
                clickPosition = localClickPosition;

            bool hasFromTarget = _currentTrial > 0 && _currentTrial - 1 < _targetSequence.Count;
            Vector2 fromTargetPosition = Vector2.zero;
            if (hasFromTarget)
            {
                int fromTargetIndex = _targetSequence[_currentTrial - 1];
                hasFromTarget = fromTargetIndex >= 0 && fromTargetIndex < _targetRects.Count;
                if (hasFromTarget)
                    fromTargetPosition = _targetRects[fromTargetIndex].anchoredPosition;
            }

            TrialResult result = new TrialResult
            {
                trialIndex = _currentTrial + 1,
                targetIndex = clickedTargetIndex,
                targetPosition = targetRect.anchoredPosition,
                fromTargetPosition = fromTargetPosition,
                hasFromTarget = hasFromTarget,
                clickPosition = clickPosition,
                clickTimeMs = clickTimeMs,
                centerDistancePixels = float.IsNaN(centerDistancePixels) ? 0f : centerDistancePixels,
                wrongClicksBeforeHit = _wrongClicksThisTrial,
                wrongTargetClicksBeforeHit = _wrongTargetClicksThisTrial,
                playAreaMissClicksBeforeHit = _playAreaMissClicksThisTrial
            };
            trialResults.Add(result);

            _currentTrial++;
            ShowCurrentTarget();
        }

        private float RecordCenterDistanceFromPointer(PointerEventData eventData)
        {
            if (!TryGetPlayAreaLocalPoint(eventData, out Vector2 localClickPosition) ||
                _currentTargetIndex < 0 ||
                _currentTargetIndex >= _targetRects.Count)
            {
                return float.NaN;
            }

            float distancePixels = Vector2.Distance(localClickPosition, _targetRects[_currentTargetIndex].anchoredPosition);
            _centerDistanceSum += distancePixels;
            _centerDistanceSamples++;
            return distancePixels;
        }

        private bool TryGetPlayAreaLocalPoint(PointerEventData eventData, out Vector2 localPoint)
        {
            localPoint = Vector2.zero;
            if (eventData == null || _playArea == null)
                return false;

            Camera eventCamera = null;
            if (_canvas != null && _canvas.renderMode != RenderMode.ScreenSpaceOverlay)
                eventCamera = _canvas.worldCamera;

            return RectTransformUtility.ScreenPointToLocalPointInRectangle(
                _playArea,
                eventData.position,
                eventCamera,
                out localPoint
            );
        }

        private IEnumerator FlashWrongTarget(int targetIndex)
        {
            if (targetIndex < 0 || targetIndex >= _targetImages.Count)
                yield break;

            Image image = _targetImages[targetIndex];
            Color previousColor = image.color;
            image.color = wrongTargetFlashColor;
            yield return new WaitForSecondsRealtime(wrongTargetFlashSeconds);

            if (image != null && !_taskComplete && targetIndex != _currentTargetIndex)
                image.color = previousColor;
        }

        private void CompleteTask()
        {
            _taskRunning = false;
            _taskComplete = true;
            RestoreCursor();
            ComputeDesignObjectives();

            for (int i = 0; i < _targetImages.Count; i++)
            {
                _targetImages[i].color = completedTargetColor;
                _targetButtons[i].interactable = false;
                if (i < _targetXLabels.Count)
                    _targetXLabels[i].enabled = false;
            }

            UpdateStatusText();
            ShowSubjectiveRatingQuestions();
        }

        private void ComputeDesignObjectives()
        {
            taskCompletionTimeMs = (Time.realtimeSinceStartup - _taskStartedAt) * 1000f;
            accuracyDistancePixels = _centerDistanceSamples > 0 ? _centerDistanceSum / _centerDistanceSamples : 0f;
        }

        private void ShowSubjectiveRatingQuestions()
        {
            if (TryShowToolkitSubjectiveRatingQuestions())
                return;

            if (writeObjectivesToBo || startBoOptimizationAfterResults)
            {
                Debug.LogError(
                    "FittsLawTask: Subjective questionnaire is not configured correctly. " +
                    "Create the required QuestionnaireToolkit items in the scene before running BO."
                );
                return;
            }

            FinalizeTaskResults();
        }

        private bool TryShowToolkitSubjectiveRatingQuestions()
        {
            QTQuestionnaireManager manager = ResolveQuestionnaireToolkitManager();
            if (manager == null)
            {
                Debug.LogWarning("FittsLawTask: QuestionnaireToolkit manager was not found in the scene. Finalizing without a questionnaire rating.");
                return false;
            }

            if (!ValidateSubjectiveQuestionnaireItems(manager))
                return false;

            EnsureQuestionnairePerformanceCsvItems(manager);

            bool started = manager.StartQuestionnaire();
            if (!started)
            {
                Debug.LogWarning("FittsLawTask: QuestionnaireToolkit questionnaire could not be started. Finalizing without a questionnaire rating.");
                return false;
            }

            return true;
        }

        public void SendFittsResultsToOptimizerFromQuestionnaire()
        {
            FinalizeTaskResults(false);
            ClearGeneratedUi();
        }

        private QTQuestionnaireManager ResolveQuestionnaireToolkitManager()
        {
            if (questionnaireToolkitManager != null)
                return questionnaireToolkitManager;

            return FindAnyObjectByType<QTQuestionnaireManager>();
        }

        private void EnsureQuestionnairePerformanceCsvItems(QTQuestionnaireManager manager = null)
        {
            if (!logPerformanceToQuestionnaireCsv)
                return;

            manager = manager != null ? manager : questionnaireToolkitManager;
            if (manager == null)
                return;

            manager.EnsureAdditionalCsvItem(
                string.IsNullOrWhiteSpace(questionnaireSpeedCsvHeader) ? speedObjectiveKey : questionnaireSpeedCsvHeader.Trim(),
                gameObject,
                nameof(FittsLawTask),
                nameof(QuestionnaireSpeedCsvValue),
                3
            );
            manager.EnsureAdditionalCsvItem(
                string.IsNullOrWhiteSpace(questionnaireAccuracyCsvHeader) ? accuracyObjectiveKey : questionnaireAccuracyCsvHeader.Trim(),
                gameObject,
                nameof(FittsLawTask),
                nameof(QuestionnaireAccuracyCsvValue),
                4
            );
        }

        private bool ValidateSubjectiveQuestionnaireItems(QTQuestionnaireManager manager)
        {
            if (manager == null || manager.questionPages == null || manager.questionPages.Count == 0)
                return false;

            int aestheticsCount = 0;
            int usabilityCount = 0;

            for (int i = 0; i < manager.questionPages.Count; i++)
            {
                GameObject page = manager.questionPages[i];
                if (page == null)
                    continue;

                QTQuestionPageManager pageManager = page.GetComponent<QTQuestionPageManager>();
                if (pageManager == null || pageManager.questionItems == null)
                    continue;

                aestheticsCount += CountSliderQuestionsForObjective(pageManager, aestheticsObjectiveKey);
                usabilityCount += CountSliderQuestionsForObjective(pageManager, usabilityObjectiveKey);
            }

            int requiredAestheticsCount = GetRequiredObjectiveSubMeasureCount(aestheticsObjectiveKey);
            int requiredUsabilityCount = GetRequiredObjectiveSubMeasureCount(usabilityObjectiveKey);
            if (aestheticsCount < requiredAestheticsCount || usabilityCount < requiredUsabilityCount)
            {
                Debug.LogError(
                    "FittsLawTask: Missing required QuestionnaireToolkit slider item(s). " +
                    $"Expected at least {requiredAestheticsCount} slider(s) for objective '{aestheticsObjectiveKey}' " +
                    $"and {requiredUsabilityCount} slider(s) for objective '{usabilityObjectiveKey}'. " +
                    $"Found {aestheticsCount} and {usabilityCount}. " +
                    "Create them manually in the scene. Submeasure headers such as 'usability1' and 'usability2' are accepted."
                );
                return false;
            }

            manager.BuildHeaderItems();
            return true;
        }

        private int GetRequiredObjectiveSubMeasureCount(string objectiveKey)
        {
            BoForUnityManager manager = GetRuntimeDesignParameterSource();
            if (TryFindBoObjective(manager, objectiveKey, out ObjectiveEntry objective) &&
                objective?.value != null)
            {
                return Mathf.Max(1, objective.value.numberOfSubMeasures);
            }

            return 1;
        }

        private static int CountSliderQuestionsForObjective(QTQuestionPageManager pageManager, string objectiveKey)
        {
            if (pageManager == null || pageManager.questionItems == null || string.IsNullOrWhiteSpace(objectiveKey))
                return 0;

            int count = 0;

            for (int i = 0; i < pageManager.questionItems.Count; i++)
            {
                GameObject item = pageManager.questionItems[i];
                if (item == null)
                    continue;

                QTSlider slider = item.GetComponent<QTSlider>();
                if (slider != null && MatchesObjectiveHeader(slider.headerName, objectiveKey))
                    count++;
            }

            return count;
        }

        private static bool MatchesObjectiveHeader(string headerName, string objectiveKey)
        {
            if (string.IsNullOrWhiteSpace(headerName) || string.IsNullOrWhiteSpace(objectiveKey))
                return false;

            string source = headerName.Trim();
            string key = objectiveKey.Trim();
            int start = 0;
            while (start < source.Length)
            {
                int idx = source.IndexOf(key, start, StringComparison.OrdinalIgnoreCase);
                if (idx < 0)
                    return false;

                int end = idx + key.Length;
                if (IsObjectiveHeaderBoundary(source, idx) && IsObjectiveHeaderBoundary(source, end))
                    return true;

                start = idx + 1;
            }

            return false;
        }

        private static bool IsObjectiveHeaderBoundary(string text, int boundaryIndex)
        {
            if (boundaryIndex <= 0 || boundaryIndex >= text.Length)
                return true;

            char left = text[boundaryIndex - 1];
            char right = text[boundaryIndex];
            if (!char.IsLetterOrDigit(left) || !char.IsLetterOrDigit(right))
                return true;

            if (char.IsLetter(left) && char.IsLetter(right) && char.IsLower(left) && char.IsUpper(right))
                return true;

            if (char.IsLetter(left) && char.IsDigit(right))
                return true;
            if (char.IsDigit(left) && char.IsLetter(right))
                return true;

            return false;
        }

        private TextMeshProUGUI CreateButtonLabel(RectTransform parent)
        {
            GameObject labelObject = CreateUiObject("Label", parent);
            RectTransform labelRect = labelObject.GetComponent<RectTransform>();
            StretchToParent(labelRect);

            TextMeshProUGUI label = labelObject.AddComponent<TextMeshProUGUI>();
            label.alignment = TextAlignmentOptions.Center;
            label.raycastTarget = false;
            return label;
        }

        private void FinalizeTaskResults(bool startOptimization = true)
        {
            if (_resultsFinalized)
                return;

            _resultsFinalized = true;

            if (logResultsToConsole)
                Debug.Log(BuildResultsSummary());

            if (writeObjectivesToBo)
                WriteBoObjectiveValues();

            if (writeResultsCsv || writeDetailedAppLogCsv)
            {
                // This runs from the questionnaire's onQuestionnaireFinished, before the questionnaire starts the
                // optimization step: a log error (file open in Excel, read-only folder) must not abort it.
                try
                {
                    string logRoot = ResolveFittsAppLogDirectory(FindPreferredBoManager());
                    Directory.CreateDirectory(logRoot);

                    if (writeResultsCsv)
                        WriteResultsCsv(logRoot);

                    if (writeDetailedAppLogCsv)
                        WriteDetailedAppLogs(logRoot);
                }
                catch (Exception e)
                {
                    Debug.LogException(e, this);
                    Debug.LogError(
                        "FittsLawTask: could not write the Fitts law logs for this round (see above). " +
                        "The objective values were still handed to the optimizer."
                    );
                }
            }

            if (startOptimization && startBoOptimizationAfterResults)
                StartBoOptimization();
        }

        private void StartBoOptimization()
        {
            BoForUnityManager manager = GetRuntimeDesignParameterSource();
            if (manager == null)
            {
                Debug.LogWarning("FittsLawTask: Could not start BO optimization because BoForUnityManager was not found.");
                return;
            }

            manager.OptimizationStart();
            if (manager.optimizationRunning)
                ClearGeneratedUi();

            if (queueNextExternalSignalIteration &&
                manager.optimizationRunning &&
                manager.iterationAdvanceMode == BoForUnityManager.IterationAdvanceMode.ExternalSignal)
            {
                manager.RequestNextIteration();
            }
        }

        private void UpdateTargetVisuals()
        {
            for (int i = 0; i < _targetImages.Count; i++)
            {
                bool isCurrentTarget = i == _currentTargetIndex;
                _targetImages[i].color = isCurrentTarget ? highlightedTargetColor : targetColor;
                _targetButtons[i].interactable = true;
                if (i < _targetXLabels.Count)
                    _targetXLabels[i].enabled = !showTargetXOnlyOnCurrentTarget || isCurrentTarget;
            }
        }

        private void UpdateStatusText()
        {
            if (!_statusText)
                return;

            if (_taskComplete)
            {
                _statusText.text = completedText;
                return;
            }

            _statusText.text = instructionText + Environment.NewLine + FormatProgressText();
        }

        private string FormatProgressText()
        {
            int visibleTrial = Mathf.Min(_currentTrial + 1, trialCount);
            try
            {
                return string.Format(CultureInfo.InvariantCulture, progressFormat, visibleTrial, trialCount);
            }
            catch (FormatException)
            {
                return "Trial " + visibleTrial.ToString(CultureInfo.InvariantCulture) +
                       " / " + trialCount.ToString(CultureInfo.InvariantCulture);
            }
        }

        private string BuildResultsSummary()
        {
            if (trialResults.Count == 0)
                return "FittsLawTask: complete with no trials recorded.";

            float sum = 0f;
            int wrongClicks = 0;
            for (int i = 0; i < trialResults.Count; i++)
            {
                sum += trialResults[i].clickTimeMs;
                wrongClicks += trialResults[i].wrongClicksBeforeHit;
            }

            float mean = sum / trialResults.Count;
            return "FittsLawTask: " + trialResults.Count.ToString(CultureInfo.InvariantCulture) +
               " trials complete. Completion time: " +
                   taskCompletionTimeMs.ToString("0.0", CultureInfo.InvariantCulture) +
                   " ms. Mean click time: " +
                   mean.ToString("0.0", CultureInfo.InvariantCulture) +
                   " ms. Mean center distance: " +
                   accuracyDistancePixels.ToString("0.0", CultureInfo.InvariantCulture) +
                   ". Wrong clicks: " + wrongClicks.ToString(CultureInfo.InvariantCulture) + ".";
        }

        private void WriteResultsCsv(string logRoot = null)
        {
            string fileName = string.IsNullOrWhiteSpace(resultsFileName) ? "fitts_law_results.csv" : resultsFileName;
            string directory = string.IsNullOrWhiteSpace(logRoot)
                ? ResolveFittsAppLogDirectory(FindPreferredBoManager())
                : logRoot;
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, GetCsvFileName(fileName, "fitts_law_results.csv"));

            StringBuilder builder = new StringBuilder();
            builder.AppendLine("trialIndex,targetIndex,targetX,targetY,clickX,clickY,clickTimeMs,centerDistancePixels,wrongClicksBeforeHit,wrongTargetClicksBeforeHit,playAreaMissClicksBeforeHit,targetCount,buttonSizePixels,buttonDistancePixels,movementDirectionDegrees,xFontSizePixels,buttonHue,buttonSaturation,buttonColorBrightness,taskCompletionTimeMs,accuracyDistancePixels");
            for (int i = 0; i < trialResults.Count; i++)
            {
                TrialResult result = trialResults[i];
                builder.Append(result.trialIndex.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.targetIndex.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.targetPosition.x.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.targetPosition.y.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.clickPosition.x.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.clickPosition.y.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.clickTimeMs.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.centerDistancePixels.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.wrongClicksBeforeHit.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.wrongTargetClicksBeforeHit.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.playAreaMissClicksBeforeHit.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(targetCount.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(circleSizePixels.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(circleDistancePixels.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(movementDirectionDegrees.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(xFontSizePixels.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(buttonHue.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(buttonSaturation.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(buttonColorBrightness.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(taskCompletionTimeMs.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.AppendLine(accuracyDistancePixels.ToString(CultureInfo.InvariantCulture));
            }

            File.WriteAllText(path, builder.ToString());
            Debug.Log("FittsLawTask: wrote results to " + path);
        }

        private void WriteDetailedAppLogs(string logRoot = null)
        {
            BoForUnityManager manager = FindPreferredBoManager();
            if (string.IsNullOrWhiteSpace(logRoot))
                logRoot = ResolveFittsAppLogDirectory(manager);

            Directory.CreateDirectory(logRoot);

            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
            ResolveFittsLogContext(manager, out string userId, out string conditionId, out string groupId);
            string iteration = GetLogIteration(manager);
            string phase = GetLogPhase(manager);

            string summaryPath = Path.Combine(logRoot, GetCsvFileName(appSummaryLogFileName, "FittsLawAppLog.csv"));
            string trialPath = Path.Combine(logRoot, GetCsvFileName(appTrialLogFileName, "FittsLawTrialLog.csv"));

            float meanClickTimeMs = ComputeMeanClickTimeMs();
            float minCenterDistancePixels = ComputeMinCorrectCenterDistancePixels();
            float maxCenterDistancePixels = ComputeMaxCorrectCenterDistancePixels();
            int totalClicks = _correctClicks + _wrongClicksTotal;
            DesignSnapshot design = _roundDesign;
            List<FittsTrialSample> isoSamples = BuildIsoTrialSamples();
            FittsBlockMetrics iso = FittsLawMath.ComputeBlockMetrics(isoSamples, _appliedCircleSizePixels);

            AppendSemicolonCsvRow(
                summaryPath,
                new[]
                {
                    "UserID", "ConditionID", "GroupID", "Timestamp", "Iteration", "Phase", "LogType",
                    "TargetCount", "ConfiguredTrialCount", "CompletedTrials",
                    "CorrectClicks", "WrongClicks", "WrongTargetClicks", "PlayAreaMissClicks", "TotalClicks",
                    "CenterDistanceSampleCount", "TaskCompletionTimeMs", "MeanClickTimeMs",
                    "MeanCenterDistancePixels", "MinCorrectCenterDistancePixels", "MaxCorrectCenterDistancePixels",
                    "ButtonSizePixels", "ButtonDistancePixels", "MovementDirectionDegrees", "XFontSizePixels",
                    "ButtonHue", "ButtonSaturation", "ButtonColorBrightness",
                    "AestheticsObjective", "SpeedObjective", "AccuracyObjective", "UsabilityObjective",
                    "AppliedButtonSizePixels", "AppliedButtonDistancePixels", "AppliedLayoutAdjusted",
                    "SelectionEvent", "IsoTrialCount", "IsoWidthPixels", "IsoAmplitudePixels", "IsoID",
                    "IsoEffectiveAmplitudePixels", "IsoEffectiveWidthPixels", "IsoIDe", "IsoMovementTimeMs",
                    "IsoThroughputBitsPerSecond"
                },
                new[]
                {
                    userId, conditionId, groupId, timestamp, iteration, phase, "summary",
                    FormatInt(targetCount), FormatInt(trialCount), FormatInt(trialResults.Count),
                    FormatInt(_correctClicks), FormatInt(_wrongClicksTotal), FormatInt(_wrongTargetClicksTotal),
                    FormatInt(_playAreaMissClicksTotal), FormatInt(totalClicks),
                    FormatInt(_centerDistanceSamples), FormatCsvFloat(taskCompletionTimeMs), FormatCsvFloat(meanClickTimeMs),
                    FormatCsvFloat(accuracyDistancePixels), FormatCsvFloat(minCenterDistancePixels),
                    FormatCsvFloat(maxCenterDistancePixels), FormatCsvFloat(design.ButtonSizePixels),
                    FormatCsvFloat(design.ButtonDistancePixels), FormatCsvFloat(movementDirectionDegrees),
                    FormatCsvFloat(design.XFontSizePixels), FormatCsvFloat(design.ButtonHue),
                    FormatCsvFloat(design.ButtonSaturation),
                    FormatCsvFloat(buttonColorBrightness), FormatObjectiveAverage(manager, aestheticsObjectiveKey),
                    FormatObjectiveAverage(manager, speedObjectiveKey), FormatObjectiveAverage(manager, accuracyObjectiveKey),
                    FormatObjectiveAverage(manager, usabilityObjectiveKey),
                    FormatCsvFloat(_appliedCircleSizePixels), FormatCsvFloat(_appliedRingDiameterPixels),
                    _appliedLayoutAdjusted ? "TRUE" : "FALSE",
                    selectionEvent.ToString(), FormatInt(iso.TrialCount), FormatCsvFloat(iso.WidthPixels),
                    FormatCsvFloat(iso.AmplitudePixels), FormatCsvFloat(iso.IndexOfDifficulty),
                    FormatCsvFloat(iso.EffectiveAmplitudePixels), FormatCsvFloat(iso.EffectiveWidthPixels),
                    FormatCsvFloat(iso.EffectiveIndexOfDifficulty), FormatCsvFloat(iso.MovementTimeMs),
                    FormatCsvFloat(iso.ThroughputBitsPerSecond)
                }
            );

            string[] trialHeaders =
            {
                "UserID", "ConditionID", "GroupID", "Timestamp", "Iteration", "Phase", "LogType",
                "TrialIndex", "TargetIndex", "TargetX", "TargetY", "ClickX", "ClickY", "ClickTimeMs",
                "CenterDistancePixels", "WrongClicksBeforeHit", "WrongTargetClicksBeforeHit",
                "PlayAreaMissClicksBeforeHit", "ButtonSizePixels", "ButtonDistancePixels",
                "MovementDirectionDegrees", "XFontSizePixels", "ButtonHue", "ButtonSaturation",
                "ButtonColorBrightness", "AppliedButtonSizePixels", "AppliedButtonDistancePixels",
                "FromTargetX", "FromTargetY", "AmplitudePixels", "EndpointDeviationPixels",
                "EffectiveAmplitudePixels", "IncludedInIsoMetrics"
            };

            // One header check and one file open for the whole round (not one per trial row).
            var trialRows = new List<string[]>(trialResults.Count);
            for (int i = 0; i < trialResults.Count; i++)
            {
                TrialResult result = trialResults[i];
                FittsTrialSample sample = isoSamples[i];
                float amplitude = sample.HasFrom ? Vector2.Distance(sample.From, sample.To) : float.NaN;
                float deviation = sample.HasFrom
                    ? FittsLawMath.EndpointDeviationOnTaskAxis(sample.From, sample.To, sample.Endpoint)
                    : float.NaN;
                trialRows.Add(
                    new[]
                    {
                        userId, conditionId, groupId, timestamp, iteration, phase, "trial",
                        FormatInt(result.trialIndex), FormatInt(result.targetIndex),
                        FormatCsvFloat(result.targetPosition.x), FormatCsvFloat(result.targetPosition.y),
                        FormatCsvFloat(result.clickPosition.x), FormatCsvFloat(result.clickPosition.y),
                        FormatCsvFloat(result.clickTimeMs), FormatCsvFloat(result.centerDistancePixels),
                        FormatInt(result.wrongClicksBeforeHit), FormatInt(result.wrongTargetClicksBeforeHit),
                        FormatInt(result.playAreaMissClicksBeforeHit), FormatCsvFloat(design.ButtonSizePixels),
                        FormatCsvFloat(design.ButtonDistancePixels), FormatCsvFloat(movementDirectionDegrees),
                        FormatCsvFloat(design.XFontSizePixels), FormatCsvFloat(design.ButtonHue),
                        FormatCsvFloat(design.ButtonSaturation), FormatCsvFloat(buttonColorBrightness),
                        FormatCsvFloat(_appliedCircleSizePixels), FormatCsvFloat(_appliedRingDiameterPixels),
                        sample.HasFrom ? FormatCsvFloat(sample.From.x) : string.Empty,
                        sample.HasFrom ? FormatCsvFloat(sample.From.y) : string.Empty,
                        FormatCsvFloat(amplitude), FormatCsvFloat(deviation),
                        FormatCsvFloat(amplitude + deviation),
                        IsFinite(deviation) ? "TRUE" : "FALSE"
                    }
                );
            }

            if (trialRows.Count > 0)
                AppendSemicolonCsvRows(trialPath, trialHeaders, trialRows);

            Debug.Log("FittsLawTask: wrote detailed app logs to " + logRoot);
        }

        /// <summary>
        /// The completed trials as ISO 9241-9 samples: movement from the previously selected target's center to
        /// this target, ending at the registered selection point. The first trial of the round has no start.
        /// </summary>
        private List<FittsTrialSample> BuildIsoTrialSamples()
        {
            var samples = new List<FittsTrialSample>(trialResults.Count);
            for (int i = 0; i < trialResults.Count; i++)
            {
                TrialResult result = trialResults[i];
                samples.Add(new FittsTrialSample
                {
                    From = result.fromTargetPosition,
                    To = result.targetPosition,
                    Endpoint = result.clickPosition,
                    MovementTimeMs = result.clickTimeMs,
                    HasFrom = result.hasFromTarget
                });
            }

            return samples;
        }

        private string ResolveFittsAppLogDirectory(BoForUnityManager manager)
        {
            if (_manualLogContextActive && !string.IsNullOrWhiteSpace(_manualLogDirectory))
                return _manualLogDirectory;

            ResolveFittsLogContext(manager, out string userId, out string conditionId, out string ignoredGroupId);
            // Same root as the optimizer and the questionnaires (persistentDataPath where StreamingAssets is read-only).
            string logRoot = LogDataFolderUtility.LogDataRoot;
            string userLogId = LogDataFolderUtility.GetOrCreateUserFolderTokenForCondition(
                logRoot,
                userId,
                conditionId,
                _manualLogContextActive || manager != null,
                _manualLogContextActive || manager != null
            );
            string conditionLogId = NormalizeLogFolderToken(conditionId);

            string runDirectory = Path.Combine(
                logRoot,
                userLogId,
                conditionLogId
            );
            if (_manualLogContextActive)
                _manualLogDirectory = runDirectory;

            return runDirectory;
        }

        private void ResolveFittsLogContext(
            BoForUnityManager manager,
            out string userId,
            out string conditionId,
            out string groupId)
        {
            if (_manualLogContextActive)
            {
                userId = _manualLogUserId;
                conditionId = _manualLogConditionId;
                groupId = _manualLogGroupId;
                return;
            }

            if (manager != null)
            {
                userId = GetContextValue(manager.userId);
                conditionId = GetContextValue(manager.conditionId);
                groupId = GetContextValue(manager.groupId);
                return;
            }

            QTQuestionnaireManager questionnaireManager = ResolveQuestionnaireToolkitManager();
            if (questionnaireManager != null)
            {
                questionnaireManager.ResolveBoContextForLogging(out userId, out conditionId, out groupId);
                return;
            }

            userId = "-1";
            conditionId = "-1";
            groupId = "-1";
        }

        private string GetLogIteration(BoForUnityManager manager)
        {
            if (_manualLogContextActive)
                return _manualLogIteration.ToString(CultureInfo.InvariantCulture);

            if (manager == null)
                return "0";

            // Python numbers evaluations globally (warm-start rows and sampling included), so use the Iteration
            // it logs this design under; the rows then join with ObservationsPerEvaluation.csv. The final-design
            // row is logged by the manager under FinalDesignLogIteration.
            if (string.Equals(GetBoPhase(manager), "finaldesign", StringComparison.Ordinal))
            {
                if (manager.FinalDesignLogIteration >= 0)
                    return manager.FinalDesignLogIteration.ToString(CultureInfo.InvariantCulture);
                if (manager.LastSuggestionLogIteration >= 0)
                    return (manager.LastSuggestionLogIteration + 1).ToString(CultureInfo.InvariantCulture);
            }
            else if (manager.LastSuggestionLogIteration >= 0)
            {
                return manager.LastSuggestionLogIteration.ToString(CultureInfo.InvariantCulture);
            }

            return manager.currentIteration.ToString(CultureInfo.InvariantCulture);
        }

        private string GetLogPhase(BoForUnityManager manager)
        {
            if (_manualLogContextActive)
                return _manualLogPhase;

            return GetBoPhase(manager);
        }

        private string FormatObjectiveAverage(BoForUnityManager manager, string objectiveKey)
        {
            if (TryFindBoObjective(manager, objectiveKey, out ObjectiveEntry objective) &&
                objective.value != null &&
                objective.value.values != null &&
                objective.value.values.Count > 0)
            {
                // The same reduction the manager sends to Python (finite sub-measures of the last window,
                // midpoint when none is finite, clamped to the bounds), so this column matches the observation.
                ObjectiveAggregate aggregate = BoObjectiveMath.Aggregate(
                    objective.value.values,
                    Mathf.Max(1, objective.value.numberOfSubMeasures),
                    objective.value.lowerBound,
                    objective.value.upperBound
                );
                return FormatCsvFloat(aggregate.Value);
            }

            if (string.Equals(objectiveKey, speedObjectiveKey, StringComparison.Ordinal))
                return FormatCsvFloat(taskCompletionTimeMs);

            if (string.Equals(objectiveKey, accuracyObjectiveKey, StringComparison.Ordinal))
                return FormatCsvFloat(accuracyDistancePixels);

            FittsLawConditionManager conditionManager = FindAnyObjectByType<FittsLawConditionManager>();
            if (conditionManager != null && conditionManager.TryGetObjectiveAverage(objectiveKey, out float conditionObjectiveValue))
                return FormatCsvFloat(conditionObjectiveValue);

            return string.Empty;
        }

        private float ComputeMeanClickTimeMs()
        {
            if (trialResults.Count == 0)
                return 0f;

            double sum = 0.0;
            for (int i = 0; i < trialResults.Count; i++)
                sum += trialResults[i].clickTimeMs;

            return (float)(sum / trialResults.Count);
        }

        private float ComputeMinCorrectCenterDistancePixels()
        {
            if (trialResults.Count == 0)
                return 0f;

            float min = float.MaxValue;
            for (int i = 0; i < trialResults.Count; i++)
                min = Mathf.Min(min, trialResults[i].centerDistancePixels);

            return min == float.MaxValue ? 0f : min;
        }

        private float ComputeMaxCorrectCenterDistancePixels()
        {
            if (trialResults.Count == 0)
                return 0f;

            float max = 0f;
            for (int i = 0; i < trialResults.Count; i++)
                max = Mathf.Max(max, trialResults[i].centerDistancePixels);

            return max;
        }

        private static string GetBoPhase(BoForUnityManager manager)
        {
            if (manager == null)
                return "manual";

            if (manager.totalIterations > 0 && manager.currentIteration > manager.totalIterations)
                return "finaldesign";

            int samplingIterations = manager.GetEffectiveSamplingIterations();
            return manager.currentIteration <= samplingIterations ? "sampling" : "optimization";
        }

        private static string GetContextValue(string value)
        {
            return string.IsNullOrWhiteSpace(value) ? "-1" : value.Trim();
        }

        private static string NormalizeLogFolderToken(string value)
        {
            return LogDataFolderUtility.NormalizeLogFolderToken(value);
        }

        private static string GetCsvFileName(string configuredFileName, string defaultFileName)
        {
            string fileName = string.IsNullOrWhiteSpace(configuredFileName)
                ? defaultFileName
                : configuredFileName.Trim();
            fileName = fileName.Replace('\\', '/');
            int slashIndex = fileName.LastIndexOf('/');
            if (slashIndex >= 0)
                fileName = fileName.Substring(slashIndex + 1);

            char[] invalidChars = Path.GetInvalidFileNameChars();
            StringBuilder builder = new StringBuilder(fileName.Length);
            for (int i = 0; i < fileName.Length; i++)
            {
                char c = fileName[i];
                builder.Append(Array.IndexOf(invalidChars, c) >= 0 ? '_' : c);
            }

            fileName = builder.ToString().Trim();
            if (string.IsNullOrWhiteSpace(fileName))
                fileName = defaultFileName;

            if (!fileName.EndsWith(".csv", StringComparison.OrdinalIgnoreCase))
                fileName += ".csv";

            return fileName;
        }

        private static void AppendSemicolonCsvRow(string path, string[] headers, string[] values)
        {
            AppendSemicolonCsvRows(path, headers, new List<string[]> { values });
        }

        private static void AppendSemicolonCsvRows(string path, string[] headers, List<string[]> rows)
        {
            string headerLine = BuildSemicolonCsvLine(headers);
            path = ResolveCsvPathForHeader(path, headerLine);
            bool writeHeader = !File.Exists(path) || new FileInfo(path).Length == 0;
            using (var writer = new StreamWriter(path, true, Encoding.UTF8))
            {
                if (writeHeader)
                    writer.WriteLine(headerLine);

                for (int i = 0; i < rows.Count; i++)
                    writer.WriteLine(BuildSemicolonCsvLine(rows[i]));
            }
        }

        /// <summary>
        /// A log written by an earlier version has other columns; appending to it would misalign every new row.
        /// Such a file is left untouched and the rows go to the first "name_N.csv" whose header matches.
        /// </summary>
        private static string ResolveCsvPathForHeader(string path, string headerLine)
        {
            string directory = Path.GetDirectoryName(path) ?? string.Empty;
            string stem = Path.GetFileNameWithoutExtension(path);
            string extension = Path.GetExtension(path);
            string candidate = path;
            for (int suffix = 1; suffix < 1000; suffix++)
            {
                if (!File.Exists(candidate) || new FileInfo(candidate).Length == 0)
                    return candidate;

                string existingHeader;
                using (var reader = new StreamReader(candidate, Encoding.UTF8, true))
                    existingHeader = reader.ReadLine();

                if (string.Equals(existingHeader, headerLine, StringComparison.Ordinal))
                    return candidate;

                if (suffix == 1)
                {
                    Debug.LogWarning(
                        "FittsLawTask: '" + path + "' has different columns (written by an earlier version); " +
                        "new rows go to a separate file next to it."
                    );
                }

                candidate = Path.Combine(directory, stem + "_" + suffix.ToString(CultureInfo.InvariantCulture) + extension);
            }

            return candidate;
        }

        private static string BuildSemicolonCsvLine(string[] cells)
        {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < cells.Length; i++)
            {
                if (i > 0)
                    builder.Append(';');
                builder.Append(EscapeSemicolonCsvCell(cells[i]));
            }

            return builder.ToString();
        }

        private static string EscapeSemicolonCsvCell(string value)
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

        private static string FormatInt(int value)
        {
            return value.ToString(CultureInfo.InvariantCulture);
        }

        private static string FormatCsvFloat(float value)
        {
            // Round-trip, culture-invariant (same as the optimizer logs); the old 3-decimal rounding erased small
            // values and made the objective columns disagree with ObservationsPerEvaluation.csv.
            return BoObjectiveMath.FormatCsvFloat(value);
        }

        private static bool IsFinite(float value)
        {
            return BoObjectiveMath.IsFinite(value);
        }

        private void WriteBoObjectiveValues()
        {
            BoForUnityManager manager = GetRuntimeDesignParameterSource();
            if (manager == null || manager.objectives == null)
            {
                Debug.LogWarning("FittsLawTask: BO objective output requested, but BoForUnityManager objectives were not found.");
                return;
            }

            WriteSingleBoObjective(manager, speedObjectiveKey, taskCompletionTimeMs);
            WriteSingleBoObjective(manager, accuracyObjectiveKey, accuracyDistancePixels);
        }

        private void WriteSingleBoObjective(BoForUnityManager manager, string key, float value)
        {
            if (!TryFindBoObjective(manager, key, out ObjectiveEntry objective))
            {
                Debug.LogWarning("FittsLawTask: Could not find BO objective '" + key + "'.");
                return;
            }

            objective.value.values = new List<float> { value };
        }

        private bool TryFindBoObjective(BoForUnityManager manager, string key, out ObjectiveEntry objective)
        {
            objective = null;
            if (manager == null || manager.objectives == null)
                return false;

            if (string.IsNullOrWhiteSpace(key))
                return false;

            for (int i = 0; i < manager.objectives.Count; i++)
            {
                ObjectiveEntry candidate = manager.objectives[i];
                if (candidate == null || candidate.value == null)
                    continue;

                if (string.Equals(candidate.key, key, StringComparison.Ordinal))
                {
                    objective = candidate;
                    return true;
                }
            }

            return false;
        }

        private static BoForUnityManager FindPreferredBoManager()
        {
            // The persistent manager that runs the study; scene copies are discarded on reload.
            if (BoForUnityManager.Instance != null)
                return BoForUnityManager.Instance;

            BoForUnityManager[] managers = FindObjectsByType<BoForUnityManager>(FindObjectsSortMode.None);
            if (managers == null || managers.Length == 0)
                return null;

            BoForUnityManager best = null;
            int bestScore = int.MinValue;
            for (int i = 0; i < managers.Length; i++)
            {
                BoForUnityManager manager = managers[i];
                if (manager == null)
                    continue;

                int score = GetBoManagerRuntimeScore(manager);
                if (best == null || score > bestScore)
                {
                    best = manager;
                    bestScore = score;
                }
            }

            return best;
        }

        private static int GetBoManagerRuntimeScore(BoForUnityManager manager)
        {
            int score = Mathf.Max(0, manager.currentIteration);
            if (manager.optimizationRunning)
                score += 1000;
            if (manager.simulationRunning)
                score += 500;
            if (manager.initialized)
                score += 250;
            if (manager.hasNewDesignParameterValues)
                score += 100;
            if (manager.gameObject != null && manager.gameObject.activeInHierarchy)
                score += 50;

            return score;
        }

        private float GetEffectiveRingRadius()
        {
            float requestedRadius = Mathf.Max(
                _appliedCircleDistancePixels * 0.5f,
                GetMinimumRingRadiusForNoOverlap(_appliedCircleSizePixels, targetCount)
            );
            if (!clampTargetsInsidePlayArea || _playArea == null)
                return requestedRadius;

            Vector2 halfSize = _playArea.rect.size * 0.5f;
            float maxX = Mathf.Max(0f, halfSize.x - Mathf.Abs(taskCenter.x) - _appliedCircleSizePixels * 0.5f);
            float maxY = Mathf.Max(0f, halfSize.y - Mathf.Abs(taskCenter.y) - _appliedCircleSizePixels * 0.5f);
            return Mathf.Min(requestedRadius, maxX, maxY);
        }

        private static float GetMinimumCircleDistanceForNoOverlap(float circleDiameterPixels, int count)
        {
            return GetMinimumRingRadiusForNoOverlap(circleDiameterPixels, count) * 2f;
        }

        private static float GetMinimumRingRadiusForNoOverlap(float circleDiameterPixels, int count)
        {
            float sine = Mathf.Sin(Mathf.PI / Mathf.Max(2, count));
            if (sine <= 0.0001f)
                return Mathf.Max(0.5f, circleDiameterPixels * 0.5f);

            return Mathf.Max(1f, circleDiameterPixels) / (2f * sine);
        }

        private static float GetMaximumCircleSizeForNoOverlap(float centerRadiusLimit, int count)
        {
            float sine = Mathf.Sin(Mathf.PI / Mathf.Max(2, count));
            if (sine <= 0.0001f)
                return Mathf.Max(1f, centerRadiusLimit * 2f);

            return Mathf.Max(1f, 2f * centerRadiusLimit * sine / (1f + sine));
        }

        private Sprite GetCircleSprite()
        {
            if (_circleSprite != null)
                return _circleSprite;

            const int textureSize = 128;
            _circleTexture = new Texture2D(textureSize, textureSize, TextureFormat.RGBA32, false);
            _circleTexture.name = "Generated Fitts Law Circle";
            _circleTexture.hideFlags = HideFlags.DontSave;

            Color[] pixels = new Color[textureSize * textureSize];
            Vector2 center = new Vector2((textureSize - 1) * 0.5f, (textureSize - 1) * 0.5f);
            float radius = (textureSize - 2) * 0.5f;

            for (int y = 0; y < textureSize; y++)
            {
                for (int x = 0; x < textureSize; x++)
                {
                    float distance = Vector2.Distance(new Vector2(x, y), center);
                    float alpha = Mathf.Clamp01(radius - distance + 1f);
                    pixels[y * textureSize + x] = new Color(1f, 1f, 1f, alpha);
                }
            }

            _circleTexture.SetPixels(pixels);
            _circleTexture.Apply();
            _circleSprite = Sprite.Create(_circleTexture, new Rect(0f, 0f, textureSize, textureSize), new Vector2(0.5f, 0.5f), textureSize);
            _circleSprite.name = "Generated Fitts Law Circle";
            _circleSprite.hideFlags = HideFlags.DontSave;
            return _circleSprite;
        }

        private static GameObject CreateUiObject(string objectName, RectTransform parent)
        {
            GameObject uiObject = new GameObject(objectName, typeof(RectTransform));
            int uiLayer = LayerMask.NameToLayer("UI");
            if (uiLayer >= 0)
                uiObject.layer = uiLayer;

            uiObject.transform.SetParent(parent, false);
            return uiObject;
        }

        private static void StretchToParent(RectTransform rectTransform)
        {
            rectTransform.anchorMin = Vector2.zero;
            rectTransform.anchorMax = Vector2.one;
            rectTransform.pivot = new Vector2(0.5f, 0.5f);
            rectTransform.anchoredPosition = Vector2.zero;
            rectTransform.sizeDelta = Vector2.zero;
        }

        private static int PositiveModulo(int value, int modulo)
        {
            if (modulo <= 0)
                return 0;

            int result = value % modulo;
            return result < 0 ? result + modulo : result;
        }

        private static void EnsureEventSystem()
        {
            // Adds StandaloneInputModule only when the legacy Input Manager is active; in Input-System-only
            // projects it would throw every frame, so the Input System's UI module is used there.
            QTEventSystemUtility.EnsureEventSystem();
        }

        private void ClearGeneratedUi()
        {
            if (_canvas != null)
            {
                _canvas.gameObject.SetActive(false);
                Destroy(_canvas.gameObject);
            }

            _canvas = null;
            _playArea = null;
            _statusText = null;
            _targetImages.Clear();
            _targetButtons.Clear();
            _targetRects.Clear();
            _targetXLabels.Clear();
        }

        private void RestoreCursor()
        {
            if (hideCursorDuringTask && _cursorStateCaptured)
            {
                Cursor.visible = _cursorWasVisible;
                _cursorStateCaptured = false;
            }
        }

        private void DestroyGeneratedAssets()
        {
            if (_circleSprite != null)
            {
                Destroy(_circleSprite);
                _circleSprite = null;
            }

            if (_circleTexture != null)
            {
                Destroy(_circleTexture);
                _circleTexture = null;
            }
        }

        private sealed class TargetPointerClickRelay : MonoBehaviour, IPointerDownHandler, IPointerClickHandler
        {
            private FittsLawTask _owner;
            private int _targetIndex;

            public void Initialize(FittsLawTask owner, int targetIndex)
            {
                _owner = owner;
                _targetIndex = targetIndex;
            }

            public void OnPointerDown(PointerEventData eventData)
            {
                if (_owner != null)
                    _owner.HandleTargetPointerEvent(_targetIndex, eventData, SelectionEvent.PointerDown);
            }

            public void OnPointerClick(PointerEventData eventData)
            {
                if (_owner != null)
                    _owner.HandleTargetPointerEvent(_targetIndex, eventData, SelectionEvent.PointerClick);
            }
        }

        private sealed class PlayAreaPointerClickRelay : MonoBehaviour, IPointerDownHandler, IPointerClickHandler
        {
            private FittsLawTask _owner;

            public void Initialize(FittsLawTask owner)
            {
                _owner = owner;
            }

            public void OnPointerDown(PointerEventData eventData)
            {
                if (_owner != null)
                    _owner.HandlePlayAreaPointerEvent(eventData, SelectionEvent.PointerDown);
            }

            public void OnPointerClick(PointerEventData eventData)
            {
                if (_owner != null)
                    _owner.HandlePlayAreaPointerEvent(eventData, SelectionEvent.PointerClick);
            }
        }
    }
}
