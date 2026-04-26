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

        [Serializable]
        public class TrialResult
        {
            public int trialIndex;
            public int targetIndex;
            public Vector2 targetPosition;
            public float clickTimeMs;
            public int wrongClicksBeforeHit;
        }

        [Header("Task")]
        [Min(2)] public int targetCount = 12;
        [Min(1)] public int trialCount = 10;
        public bool startOnAwake = true;
        [Min(0f)] public float startDelaySeconds = 0.25f;
        public bool restartWithKey = true;
        public KeyCode restartKey = KeyCode.R;

        [Header("Ready-to-run BO Example")]
        public bool ensureBoManagerInScene = true;
        public bool configureBoManagerForFittsTask = true;
        public bool waitForBoEvaluationStart = true;
        public bool startBoOptimizationAfterResults = true;
        public bool queueNextExternalSignalIteration = true;
        [Min(0)] public int boOptimizationIterations = 5;

        [Header("Design Parameters")]
        [Min(1f)] public float circleSizePixels = 72f;
        [Min(1f)] public float circleDistancePixels = 520f;
        public float movementDirectionDegrees = 0f;

        [Header("BO Design Parameters")]
        public bool readDesignParametersFromBo = true;
        public bool boParameterValuesAreNormalized = false;
        public string circleSizeParameterKey = "circle_size";
        [Min(0)] public int circleSizeParameterIndex = 0;
        public Vector2 circleSizeRangePixels = new Vector2(40f, 120f);
        public string circleDistanceParameterKey = "circle_distance";
        [Min(0)] public int circleDistanceParameterIndex = 1;
        public Vector2 circleDistanceRangePixels = new Vector2(220f, 760f);
        public string movementDirectionParameterKey = "movement_direction";
        [Min(0)] public int movementDirectionParameterIndex = 2;
        public Vector2 movementDirectionRangeDegrees = new Vector2(0f, 180f);

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
        [Min(0f)] public float targetOutlineWidth = 2f;
        public bool showTargetLabels = false;
        public Color targetLabelColor = Color.white;
        [Min(1)] public int targetLabelFontSize = 22;

        [Header("Interaction")]
        public bool countWrongTargetClicks = true;
        public bool countPlayAreaMissClicks = true;
        public bool hideCursorDuringTask = false;
        [Min(0f)] public float wrongTargetFlashSeconds = 0.08f;

        [Header("Status Text")]
        public bool showStatusText = true;
        public string instructionText = "Click the highlighted target";
        public string progressFormat = "Trial {0} / {1}";
        public string completedText = "Task complete";
        public Color statusTextColor = Color.white;
        [Min(1)] public int statusFontSize = 32;

        [Header("Questionnaire Toolkit")]
        public QTQuestionnaireManager questionnaireToolkitManager;

        [Header("Design Objectives")]
        public float taskCompletionTimeMs;
        public float accuracyPercent;

        [Header("BO Design Objectives")]
        public bool writeObjectivesToBo = true;
        public string taskCompletionObjectiveKey = "task_completion_time";
        [Min(0)] public int taskCompletionObjectiveIndex = 0;
        public string accuracyObjectiveKey = "accuracy";
        [Min(0)] public int accuracyObjectiveIndex = 1;
        public string mentalDemandObjectiveKey = "mental_demand";

        [Header("Result Logging")]
        public bool logResultsToConsole = true;
        public bool writeResultsCsv = false;
        public string resultsFileName = "fitts_law_results.csv";
        public List<TrialResult> trialResults = new List<TrialResult>();

        private readonly List<Image> _targetImages = new List<Image>();
        private readonly List<Button> _targetButtons = new List<Button>();
        private readonly List<RectTransform> _targetRects = new List<RectTransform>();
        private readonly List<int> _targetSequence = new List<int>();

        private Canvas _canvas;
        private RectTransform _playArea;
        private TextMeshProUGUI _statusText;
        private Sprite _circleSprite;
        private Texture2D _circleTexture;
        private int _currentTrial;
        private int _currentTargetIndex = -1;
        private int _wrongClicksThisTrial;
        private int _wrongClicksTotal;
        private int _correctClicks;
        private float _targetShownAt;
        private float _taskStartedAt;
        private bool _taskRunning;
        private bool _taskComplete;
        private bool _cursorWasVisible;
        private bool _cursorStateCaptured;
        private bool _resultsFinalized;

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
            if (restartWithKey && Input.GetKeyDown(restartKey))
                RestartTask();
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
            fixedPlayAreaSize = new Vector2(Mathf.Max(1f, fixedPlayAreaSize.x), Mathf.Max(1f, fixedPlayAreaSize.y));
            referenceResolution = new Vector2(Mathf.Max(1f, referenceResolution.x), Mathf.Max(1f, referenceResolution.y));
            statusFontSize = Mathf.Max(1, statusFontSize);
            targetLabelFontSize = Mathf.Max(1, targetLabelFontSize);
            circleSizeParameterIndex = Mathf.Max(0, circleSizeParameterIndex);
            circleDistanceParameterIndex = Mathf.Max(0, circleDistanceParameterIndex);
            movementDirectionParameterIndex = Mathf.Max(0, movementDirectionParameterIndex);
            taskCompletionObjectiveIndex = Mathf.Max(0, taskCompletionObjectiveIndex);
            accuracyObjectiveIndex = Mathf.Max(0, accuracyObjectiveIndex);
            boOptimizationIterations = Mathf.Max(0, boOptimizationIterations);
        }

        private IEnumerator WaitForBoEvaluationStart()
        {
            BoForUnityManager manager = FindPreferredBoManager();
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
            ApplyBoDesignParameters();
            EnsureEventSystem();
            CreateCanvas();
            CreateTargets();
            BuildTargetSequence();

            _currentTrial = 0;
            _wrongClicksThisTrial = 0;
            _wrongClicksTotal = 0;
            _correctClicks = 0;
            _taskRunning = true;
            _taskComplete = false;
            taskCompletionTimeMs = 0f;
            accuracyPercent = 0f;
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
            RestoreCursor();
            BeginTask();
        }

        private void EnsureBoManagerForExample()
        {
            BoForUnityManager manager = FindPreferredBoManager();
            bool createdManager = false;
            if (manager == null)
            {
                GameObject managerObject = new GameObject("BOforUnityManager");
                managerObject.AddComponent<MainThreadDispatcher>();
                managerObject.AddComponent<Optimizer>();
                managerObject.AddComponent<PythonStarter>();
                managerObject.AddComponent<SocketNetwork>();
                manager = managerObject.AddComponent<BoForUnityManager>();
                createdManager = true;
            }

            EnsureBoComponents(manager);
            EnsureEventSystem();
            EnsureBoControlUi(manager);

            if (configureBoManagerForFittsTask || (createdManager && !HasFittsBoConfiguration(manager)))
                ConfigureBoManagerForFittsTask(manager);
        }

        private void EnsureBoComponents(BoForUnityManager manager)
        {
            if (manager == null)
                return;

            GameObject managerObject = manager.gameObject;
            manager.mainThreadDispatcher = managerObject.GetComponent<MainThreadDispatcher>() ??
                                           managerObject.AddComponent<MainThreadDispatcher>();
            manager.optimizer = managerObject.GetComponent<Optimizer>() ??
                                managerObject.AddComponent<Optimizer>();
            manager.pythonStarter = managerObject.GetComponent<PythonStarter>() ??
                                    managerObject.AddComponent<PythonStarter>();
            manager.socketNetwork = managerObject.GetComponent<SocketNetwork>() ??
                                    managerObject.AddComponent<SocketNetwork>();
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

        private bool HasFittsBoConfiguration(BoForUnityManager manager)
        {
            return manager != null &&
                   HasParameterKey(manager, circleSizeParameterKey) &&
                   HasParameterKey(manager, circleDistanceParameterKey) &&
                   HasParameterKey(manager, movementDirectionParameterKey) &&
                   HasObjectiveKey(manager, taskCompletionObjectiveKey) &&
                   HasObjectiveKey(manager, accuracyObjectiveKey) &&
                   HasObjectiveKey(manager, mentalDemandObjectiveKey);
        }

        private static bool HasParameterKey(BoForUnityManager manager, string key)
        {
            if (manager == null || manager.parameters == null || string.IsNullOrWhiteSpace(key))
                return false;

            for (int i = 0; i < manager.parameters.Count; i++)
            {
                ParameterEntry parameter = manager.parameters[i];
                if (parameter != null && string.Equals(parameter.key, key, StringComparison.Ordinal))
                    return true;
            }

            return false;
        }

        private static bool HasObjectiveKey(BoForUnityManager manager, string key)
        {
            if (manager == null || manager.objectives == null || string.IsNullOrWhiteSpace(key))
                return false;

            for (int i = 0; i < manager.objectives.Count; i++)
            {
                ObjectiveEntry objective = manager.objectives[i];
                if (objective != null && string.Equals(objective.key, key, StringComparison.Ordinal))
                    return true;
            }

            return false;
        }

        private void ConfigureBoManagerForFittsTask(BoForUnityManager manager)
        {
            float circleSizeMin = Mathf.Min(circleSizeRangePixels.x, circleSizeRangePixels.y);
            float circleSizeMax = Mathf.Max(circleSizeRangePixels.x, circleSizeRangePixels.y);
            float circleDistanceMin = Mathf.Min(circleDistanceRangePixels.x, circleDistanceRangePixels.y);
            float circleDistanceMax = Mathf.Max(circleDistanceRangePixels.x, circleDistanceRangePixels.y);
            float movementDirectionMin = Mathf.Min(movementDirectionRangeDegrees.x, movementDirectionRangeDegrees.y);
            float movementDirectionMax = Mathf.Max(movementDirectionRangeDegrees.x, movementDirectionRangeDegrees.y);

            manager.parameters = new List<ParameterEntry>
            {
                new ParameterEntry(
                    circleSizeParameterKey,
                    new ParameterArgs(circleSizeMin, circleSizeMax)
                    {
                        Value = Mathf.Clamp(circleSizePixels, circleSizeMin, circleSizeMax)
                    }),
                new ParameterEntry(
                    circleDistanceParameterKey,
                    new ParameterArgs(circleDistanceMin, circleDistanceMax)
                    {
                        Value = Mathf.Clamp(circleDistancePixels, circleDistanceMin, circleDistanceMax)
                    }),
                new ParameterEntry(
                    movementDirectionParameterKey,
                    new ParameterArgs(movementDirectionMin, movementDirectionMax)
                    {
                        Value = Mathf.Clamp(movementDirectionDegrees, movementDirectionMin, movementDirectionMax)
                    })
            };

            manager.objectives = new List<ObjectiveEntry>
            {
                new ObjectiveEntry(
                    taskCompletionObjectiveKey,
                    new ObjectiveArgs(0f, 120000f, true, 1)),
                new ObjectiveEntry(
                    accuracyObjectiveKey,
                    new ObjectiveArgs(0f, 100f, false, 1)),
                new ObjectiveEntry(
                    mentalDemandObjectiveKey,
                    new ObjectiveArgs(0f, 100f, true, 1))
            };

            manager.optimizerBackend = BoForUnityManager.OptimizerBackend.BoTorch;
            manager.iterationAdvanceMode = BoForUnityManager.IterationAdvanceMode.ExternalSignal;
            manager.reloadSceneOnIterationAdvance = true;
            manager.enableFinalDesignRound = false;
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

        private void ApplyBoDesignParameters()
        {
            if (!readDesignParametersFromBo)
                return;

            if (TryReadBoParameter(circleSizeParameterKey, circleSizeParameterIndex, out float circleSizeValue))
                circleSizePixels = MapBoParameterValue(circleSizeValue, circleSizeRangePixels);

            if (TryReadBoParameter(circleDistanceParameterKey, circleDistanceParameterIndex, out float distanceValue))
                circleDistancePixels = MapBoParameterValue(distanceValue, circleDistanceRangePixels);

            if (TryReadBoParameter(movementDirectionParameterKey, movementDirectionParameterIndex, out float directionValue))
                movementDirectionDegrees = MapBoParameterValue(directionValue, movementDirectionRangeDegrees);
        }

        private float MapBoParameterValue(float value, Vector2 range)
        {
            if (!boParameterValuesAreNormalized)
                return value;

            float min = Mathf.Min(range.x, range.y);
            float max = Mathf.Max(range.x, range.y);
            return Mathf.Lerp(min, max, Mathf.Clamp01(value));
        }

        private bool TryReadBoParameter(string key, int fallbackIndex, out float value)
        {
            value = 0f;
            BoForUnityManager manager = FindPreferredBoManager();
            if (manager == null || manager.parameters == null)
                return false;

            if (!string.IsNullOrWhiteSpace(key))
            {
                for (int i = 0; i < manager.parameters.Count; i++)
                {
                    ParameterEntry parameter = manager.parameters[i];
                    if (parameter == null || parameter.value == null)
                        continue;

                    if (string.Equals(parameter.key, key, StringComparison.Ordinal))
                    {
                        value = parameter.value.Value;
                        return true;
                    }
                }
            }

            if (fallbackIndex >= 0 && fallbackIndex < manager.parameters.Count)
            {
                ParameterEntry parameter = manager.parameters[fallbackIndex];
                if (parameter != null && parameter.value != null)
                {
                    value = parameter.value.Value;
                    return true;
                }
            }

            return false;
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
                Button playAreaButton = playAreaObject.AddComponent<Button>();
                playAreaButton.transition = Selectable.Transition.None;
                playAreaButton.onClick.AddListener(HandleMissClick);
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

            float effectiveRadius = GetEffectiveRingRadius();
            for (int i = 0; i < targetCount; i++)
            {
                GameObject targetObject = CreateUiObject("Target " + (i + 1).ToString(CultureInfo.InvariantCulture), _playArea);
                RectTransform targetRect = targetObject.GetComponent<RectTransform>();
                targetRect.anchorMin = new Vector2(0.5f, 0.5f);
                targetRect.anchorMax = new Vector2(0.5f, 0.5f);
                targetRect.pivot = new Vector2(0.5f, 0.5f);
                targetRect.sizeDelta = new Vector2(circleSizePixels, circleSizePixels);

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
                int clickedTargetIndex = i;
                button.onClick.AddListener(() => HandleTargetClick(clickedTargetIndex));

                if (showTargetLabels)
                    CreateTargetLabel(targetObject.transform, i + 1);

                _targetImages.Add(image);
                _targetButtons.Add(button);
                _targetRects.Add(targetRect);
            }
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
                int halfTargetCount = Mathf.Max(1, targetCount / 2);
                int pairCount = targetCount % 2 == 0 ? halfTargetCount : targetCount;
                for (int i = 0; i < trialCount; i++)
                {
                    int pairIndex = (i / 2) % pairCount;
                    int sideOffset = i % 2 == 0 ? 0 : halfTargetCount;
                    _targetSequence.Add(PositiveModulo(start + pairIndex + sideOffset, targetCount));
                }
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
            _targetShownAt = Time.realtimeSinceStartup;
            UpdateTargetVisuals();
            UpdateStatusText();
        }

        private void HandleTargetClick(int clickedTargetIndex)
        {
            if (!_taskRunning || _taskComplete)
                return;

            if (clickedTargetIndex != _currentTargetIndex)
            {
                if (countWrongTargetClicks)
                {
                    _wrongClicksThisTrial++;
                    _wrongClicksTotal++;
                }

                if (wrongTargetFlashSeconds > 0f)
                    StartCoroutine(FlashWrongTarget(clickedTargetIndex));

                return;
            }

            _correctClicks++;
            AdvanceAfterHit(clickedTargetIndex);
        }

        private void HandleMissClick()
        {
            if (!_taskRunning || _taskComplete || !countPlayAreaMissClicks)
                return;

            _wrongClicksThisTrial++;
            _wrongClicksTotal++;
        }

        private void AdvanceAfterHit(int clickedTargetIndex)
        {
            float clickTimeMs = (Time.realtimeSinceStartup - _targetShownAt) * 1000f;
            RectTransform targetRect = _targetRects[clickedTargetIndex];

            TrialResult result = new TrialResult
            {
                trialIndex = _currentTrial + 1,
                targetIndex = clickedTargetIndex,
                targetPosition = targetRect.anchoredPosition,
                clickTimeMs = clickTimeMs,
                wrongClicksBeforeHit = _wrongClicksThisTrial
            };
            trialResults.Add(result);

            _currentTrial++;
            ShowCurrentTarget();
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
            }

            UpdateStatusText();
            ShowMentalDemandQuestion();
        }

        private void ComputeDesignObjectives()
        {
            taskCompletionTimeMs = (Time.realtimeSinceStartup - _taskStartedAt) * 1000f;
            int totalClicks = Mathf.Max(1, _correctClicks + _wrongClicksTotal);
            accuracyPercent = 100f * _correctClicks / totalClicks;
        }

        private void ShowMentalDemandQuestion()
        {
            if (TryShowToolkitMentalDemandQuestion())
                return;

            FinalizeTaskResults();
        }

        private bool TryShowToolkitMentalDemandQuestion()
        {
            QTQuestionnaireManager manager = ResolveQuestionnaireToolkitManager();
            if (manager == null)
            {
                Debug.LogWarning("FittsLawTask: QuestionnaireToolkit manager was not found in the scene. Finalizing without a questionnaire rating.");
                return false;
            }

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

            QTQuestionnaireManager[] managers = FindObjectsOfType<QTQuestionnaireManager>();
            if (managers == null || managers.Length == 0)
                return null;

            return managers[0];
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

            if (writeResultsCsv)
                WriteResultsCsv();

            if (writeObjectivesToBo)
                WriteBoObjectiveValues();

            if (startOptimization && startBoOptimizationAfterResults)
                StartBoOptimization();
        }

        private void StartBoOptimization()
        {
            BoForUnityManager manager = FindPreferredBoManager();
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
                   " ms. Accuracy: " +
                   accuracyPercent.ToString("0.0", CultureInfo.InvariantCulture) +
                   "%. Wrong clicks: " + wrongClicks.ToString(CultureInfo.InvariantCulture) + ".";
        }

        private void WriteResultsCsv()
        {
            string fileName = string.IsNullOrWhiteSpace(resultsFileName) ? "fitts_law_results.csv" : resultsFileName;
            string path = Path.Combine(Application.persistentDataPath, fileName);

            StringBuilder builder = new StringBuilder();
            builder.AppendLine("trialIndex,targetIndex,targetX,targetY,clickTimeMs,wrongClicksBeforeHit,targetCount,circleSizePixels,circleDistancePixels,movementDirectionDegrees,taskCompletionTimeMs,accuracyPercent");
            for (int i = 0; i < trialResults.Count; i++)
            {
                TrialResult result = trialResults[i];
                builder.Append(result.trialIndex.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.targetIndex.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.targetPosition.x.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.targetPosition.y.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.clickTimeMs.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(result.wrongClicksBeforeHit.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(targetCount.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(circleSizePixels.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(circleDistancePixels.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(movementDirectionDegrees.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.Append(taskCompletionTimeMs.ToString(CultureInfo.InvariantCulture)).Append(',');
                builder.AppendLine(accuracyPercent.ToString(CultureInfo.InvariantCulture));
            }

            File.WriteAllText(path, builder.ToString());
            Debug.Log("FittsLawTask: wrote results to " + path);
        }

        private void WriteBoObjectiveValues()
        {
            BoForUnityManager manager = FindPreferredBoManager();
            if (manager == null || manager.objectives == null)
            {
                Debug.LogWarning("FittsLawTask: BO objective output requested, but BoForUnityManager objectives were not found.");
                return;
            }

            WriteSingleBoObjective(manager, taskCompletionObjectiveKey, taskCompletionObjectiveIndex, taskCompletionTimeMs);
            WriteSingleBoObjective(manager, accuracyObjectiveKey, accuracyObjectiveIndex, accuracyPercent);
        }

        private void WriteSingleBoObjective(BoForUnityManager manager, string key, int fallbackIndex, float value)
        {
            if (!TryFindBoObjective(manager, key, fallbackIndex, out ObjectiveEntry objective))
            {
                Debug.LogWarning("FittsLawTask: Could not find BO objective '" + key + "'.");
                return;
            }

            objective.value.values = new List<float> { value };
        }

        private bool TryFindBoObjective(BoForUnityManager manager, string key, int fallbackIndex, out ObjectiveEntry objective)
        {
            objective = null;
            if (manager == null || manager.objectives == null)
                return false;

            if (!string.IsNullOrWhiteSpace(key))
            {
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
            }

            if (fallbackIndex >= 0 && fallbackIndex < manager.objectives.Count)
            {
                ObjectiveEntry candidate = manager.objectives[fallbackIndex];
                if (candidate != null && candidate.value != null)
                {
                    objective = candidate;
                    return true;
                }
            }

            return false;
        }

        private static BoForUnityManager FindPreferredBoManager()
        {
            BoForUnityManager[] managers = FindObjectsOfType<BoForUnityManager>();
            if (managers == null || managers.Length == 0)
                return null;

            BoForUnityManager fallback = managers[0];
            for (int i = 0; i < managers.Length; i++)
            {
                BoForUnityManager manager = managers[i];
                if (manager == null)
                    continue;

                if (manager.initialized ||
                    manager.simulationRunning ||
                    manager.optimizationRunning ||
                    manager.currentIteration > 0)
                {
                    return manager;
                }

                fallback = manager;
            }

            return fallback;
        }

        private float GetEffectiveRingRadius()
        {
            float requestedRadius = circleDistancePixels * 0.5f;
            if (!clampTargetsInsidePlayArea || _playArea == null)
                return requestedRadius;

            Vector2 halfSize = _playArea.rect.size * 0.5f;
            float maxX = Mathf.Max(0f, halfSize.x - Mathf.Abs(taskCenter.x) - circleSizePixels * 0.5f);
            float maxY = Mathf.Max(0f, halfSize.y - Mathf.Abs(taskCenter.y) - circleSizePixels * 0.5f);
            return Mathf.Min(requestedRadius, maxX, maxY);
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
            if (FindObjectOfType<EventSystem>() != null)
                return;

            GameObject eventSystemObject = new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
            eventSystemObject.hideFlags = HideFlags.None;
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
    }
}
