using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using BOforUnity.Scripts;
using QuestionnaireToolkit.Scripts;
using UnityEngine;

namespace BOforUnity.Examples
{
    public class FittsLawConditionManager : MonoBehaviour, IQuestionnaireOptimizationBridge
    {
        public enum ConditionMode
        {
            [InspectorName("HITL MOBO")]
            AdaptiveBo,
            Static,
            Random
        }

        [Header("References")]
        public FittsLawTask fittsLawTask;
        public QTQuestionnaireManager questionnaireManager;
        public BoForUnityManager iterationSettingsSource;

        [Header("Condition")]
        public ConditionMode conditionMode = ConditionMode.AdaptiveBo;
        public bool setConditionIdFromMode = true;
        public bool startOnAwake = true;
        public string userId = "-1";
        public string conditionId = "HITL MOBO";
        public string groupId = "-1";

        [Header("Iterations")]
        public bool readIterationsFromSource = true;
        [Min(0)] public int samplingIterations = 3;
        [Min(0)] public int optimizationIterations = 2;
        public bool includeFinalDesignRound = true;
        [Min(0f)] public float nextRoundDelaySeconds = 0.25f;

        private readonly Dictionary<string, List<float>> _currentObjectiveValues =
            new Dictionary<string, List<float>>(StringComparer.OrdinalIgnoreCase);

        private int _currentRound;
        private int _baseRoundCount;
        private int _totalRoundCount;
        private bool _started;
        private bool _advanceQueued;
        private bool _runtimeUserFolderReserved;

        public bool UsesExternalIterationSignal =>
            conditionMode == ConditionMode.AdaptiveBo &&
            iterationSettingsSource != null &&
            iterationSettingsSource.UsesExternalIterationSignal;

        public bool EnablePriorRatingHints =>
            conditionMode == ConditionMode.AdaptiveBo &&
            iterationSettingsSource != null &&
            iterationSettingsSource.EnablePriorRatingHints;

        public float PriorRatingHintAlpha =>
            conditionMode == ConditionMode.AdaptiveBo && iterationSettingsSource != null
                ? iterationSettingsSource.PriorRatingHintAlpha
                : 0f;

        public string UserId =>
            conditionMode == ConditionMode.AdaptiveBo && iterationSettingsSource != null
                ? iterationSettingsSource.UserId
                : ResolveContextValue(userId);

        public string ConditionId =>
            conditionMode == ConditionMode.AdaptiveBo && iterationSettingsSource != null
                ? iterationSettingsSource.ConditionId
                : ResolveContextValue(GetConfiguredConditionId());

        public string GroupId =>
            conditionMode == ConditionMode.AdaptiveBo && iterationSettingsSource != null
                ? iterationSettingsSource.GroupId
                : ResolveContextValue(groupId);

        private void Awake()
        {
            ResolveReferences();
            ApplyConditionConfiguration();
        }

        private IEnumerator Start()
        {
            if (!startOnAwake || conditionMode == ConditionMode.AdaptiveBo)
                yield break;

            yield return null;
            if (nextRoundDelaySeconds > 0f)
                yield return new WaitForSecondsRealtime(nextRoundDelaySeconds);

            BeginNextRound();
        }

        private void OnValidate()
        {
            samplingIterations = Mathf.Max(0, samplingIterations);
            optimizationIterations = Mathf.Max(0, optimizationIterations);
            nextRoundDelaySeconds = Mathf.Max(0f, nextRoundDelaySeconds);

            if (setConditionIdFromMode || string.IsNullOrWhiteSpace(conditionId))
            {
                conditionId = GetDefaultConditionIdForMode();
            }
        }

        public void OptimizationStart()
        {
            if (conditionMode == ConditionMode.AdaptiveBo)
            {
                iterationSettingsSource?.OptimizationStart();
                return;
            }

            QueueNextRound();
        }

        public void RequestNextIteration()
        {
            if (conditionMode == ConditionMode.AdaptiveBo)
            {
                iterationSettingsSource?.RequestNextIteration();
                return;
            }

            QueueNextRound();
        }

        public void SubmitQuestionnaireObjectiveValue(string headerName, string rawValue, string sourceName)
        {
            if (conditionMode == ConditionMode.AdaptiveBo)
            {
                iterationSettingsSource?.SubmitQuestionnaireObjectiveValue(headerName, rawValue, sourceName);
                return;
            }

            if (!float.TryParse(rawValue, NumberStyles.Float, CultureInfo.InvariantCulture, out float value))
                return;

            string objectiveKey = ResolveObjectiveKey(headerName);
            if (string.IsNullOrWhiteSpace(objectiveKey))
                return;

            if (!_currentObjectiveValues.TryGetValue(objectiveKey, out List<float> values))
            {
                values = new List<float>();
                _currentObjectiveValues[objectiveKey] = values;
            }

            values.Add(value);
        }

        public bool TryGetObjectiveAverage(string objectiveKey, out float value)
        {
            value = 0f;
            if (string.IsNullOrWhiteSpace(objectiveKey) ||
                !_currentObjectiveValues.TryGetValue(objectiveKey.Trim(), out List<float> values) ||
                values == null ||
                values.Count == 0)
            {
                return false;
            }

            double sum = 0.0;
            int count = 0;
            for (int i = 0; i < values.Count; i++)
            {
                if (!IsFinite(values[i]))
                    continue;

                sum += values[i];
                count++;
            }

            if (count == 0)
                return false;

            value = (float)(sum / count);
            return true;
        }

        public void SetPriorSliderRatingHint(string questionKey, float sliderValue)
        {
            if (conditionMode == ConditionMode.AdaptiveBo)
                iterationSettingsSource?.SetPriorSliderRatingHint(questionKey, sliderValue);
        }

        public bool TryGetPriorSliderRatingHint(string questionKey, out float sliderValue)
        {
            sliderValue = 0f;
            return conditionMode == ConditionMode.AdaptiveBo &&
                   iterationSettingsSource != null &&
                   iterationSettingsSource.TryGetPriorSliderRatingHint(questionKey, out sliderValue);
        }

        public void RemovePriorSliderRatingHint(string questionKey)
        {
            if (conditionMode == ConditionMode.AdaptiveBo)
                iterationSettingsSource?.RemovePriorSliderRatingHint(questionKey);
        }

        public void SetPriorLinearScaleRatingHint(string questionKey, string answerValue)
        {
            if (conditionMode == ConditionMode.AdaptiveBo)
                iterationSettingsSource?.SetPriorLinearScaleRatingHint(questionKey, answerValue);
        }

        public bool TryGetPriorLinearScaleRatingHint(string questionKey, out string answerValue)
        {
            answerValue = null;
            return conditionMode == ConditionMode.AdaptiveBo &&
                   iterationSettingsSource != null &&
                   iterationSettingsSource.TryGetPriorLinearScaleRatingHint(questionKey, out answerValue);
        }

        public void RemovePriorLinearScaleRatingHint(string questionKey)
        {
            if (conditionMode == ConditionMode.AdaptiveBo)
                iterationSettingsSource?.RemovePriorLinearScaleRatingHint(questionKey);
        }

        private void ResolveReferences()
        {
            if (fittsLawTask == null)
                fittsLawTask = FindSceneObject<FittsLawTask>();

            if (questionnaireManager == null)
                questionnaireManager = FindSceneObject<QTQuestionnaireManager>();

            if (iterationSettingsSource == null)
                iterationSettingsSource = FindSceneObject<BoForUnityManager>();
        }

        private void ApplyConditionConfiguration()
        {
            if (conditionMode == ConditionMode.AdaptiveBo)
            {
                SyncAdaptiveConditionIdToSource();
                return;
            }

            RefreshIterationCounts();
            SyncContextToReferencedComponents();
            DisableBoRuntimeForBaseline();

            if (fittsLawTask == null)
                return;

            fittsLawTask.startOnAwake = false;
            fittsLawTask.ensureBoManagerInScene = false;
            fittsLawTask.configureBoManagerForFittsTask = false;
            fittsLawTask.waitForBoEvaluationStart = false;
            fittsLawTask.startBoOptimizationAfterResults = false;
            fittsLawTask.queueNextExternalSignalIteration = false;
            fittsLawTask.readDesignParametersFromBo = false;
            fittsLawTask.writeObjectivesToBo = false;
            fittsLawTask.writeDetailedAppLogCsv = true;
            fittsLawTask.randomizeDesignParametersOnBegin = conditionMode == ConditionMode.Random;
        }

        private void SyncAdaptiveConditionIdToSource()
        {
            if (!setConditionIdFromMode)
                return;

            conditionId = ResolveContextValue(GetDefaultConditionIdForMode());

            if (iterationSettingsSource != null)
                iterationSettingsSource.conditionId = conditionId;

            if (questionnaireManager != null)
                questionnaireManager.contextConditionId = conditionId;
        }

        private void DisableBoRuntimeForBaseline()
        {
            if (iterationSettingsSource == null || iterationSettingsSource.gameObject == gameObject)
                return;

            iterationSettingsSource.gameObject.SetActive(false);
        }

        private void RefreshIterationCounts()
        {
            int sourceSampling = samplingIterations;
            int sourceOptimization = optimizationIterations;

            if (readIterationsFromSource && iterationSettingsSource != null)
            {
                sourceSampling = Mathf.Max(0, iterationSettingsSource.numSamplingIterations);
                sourceOptimization = Mathf.Max(0, iterationSettingsSource.numOptimizationIterations);
            }

            samplingIterations = sourceSampling;
            optimizationIterations = sourceOptimization;
            _baseRoundCount = Mathf.Max(1, samplingIterations + optimizationIterations);
            _totalRoundCount = _baseRoundCount + (includeFinalDesignRound ? 1 : 0);

            if (iterationSettingsSource != null)
            {
                iterationSettingsSource.numSamplingIterations = samplingIterations;
                iterationSettingsSource.numOptimizationIterations = optimizationIterations;
                iterationSettingsSource.totalIterations = _baseRoundCount;
            }
        }

        private void SyncContextToReferencedComponents()
        {
            conditionId = ResolveContextValue(GetConfiguredConditionId());
            groupId = ResolveContextValue(groupId);
            EnsureUniqueRuntimeUserFolder();

            if (iterationSettingsSource != null)
            {
                iterationSettingsSource.userId = userId;
                iterationSettingsSource.conditionId = conditionId;
                iterationSettingsSource.groupId = groupId;
            }

            if (questionnaireManager != null)
            {
                questionnaireManager.contextUserId = userId;
                questionnaireManager.contextConditionId = conditionId;
                questionnaireManager.contextGroupId = groupId;
            }
        }

        private void EnsureUniqueRuntimeUserFolder()
        {
            if (_runtimeUserFolderReserved)
                return;

            string requestedUserId = ResolveContextValue(userId);
            string normalizedRequestedUserId = LogDataFolderUtility.NormalizeLogFolderToken(requestedUserId);
            userId = LogDataFolderUtility.GetOrCreateUserFolderTokenForCondition(
                LogDataFolderUtility.StreamingAssetsLogRoot,
                requestedUserId,
                conditionId
            );
            _runtimeUserFolderReserved = true;

            if (!string.Equals(normalizedRequestedUserId, userId, StringComparison.Ordinal))
            {
                Debug.Log(
                    $"FittsLawConditionManager: user log folder '{normalizedRequestedUserId}' already exists. " +
                    $"Using '{userId}' for this condition run."
                );
            }
        }

        private void QueueNextRound()
        {
            if (!_started || _advanceQueued)
                return;

            _advanceQueued = true;
            StartCoroutine(BeginNextRoundAfterDelay());
        }

        private IEnumerator BeginNextRoundAfterDelay()
        {
            if (nextRoundDelaySeconds > 0f)
                yield return new WaitForSecondsRealtime(nextRoundDelaySeconds);
            else
                yield return null;

            _advanceQueued = false;
            BeginNextRound();
        }

        private void BeginNextRound()
        {
            RefreshIterationCounts();
            SyncContextToReferencedComponents();

            if (_currentRound >= _totalRoundCount)
            {
                Debug.Log(
                    $"FittsLawConditionManager: condition '{conditionId}' completed {_totalRoundCount} rounds."
                );
                return;
            }

            _started = true;
            _currentRound++;
            _currentObjectiveValues.Clear();

            string phase = GetPhaseForRound(_currentRound);
            if (iterationSettingsSource != null)
                iterationSettingsSource.currentIteration = _currentRound;

            if (fittsLawTask == null)
            {
                Debug.LogError("FittsLawConditionManager: FittsLawTask reference is missing.");
                return;
            }

            fittsLawTask.SetManualLogContext(_currentRound, phase, userId, conditionId, groupId);
            fittsLawTask.BeginTask();
        }

        private string GetPhaseForRound(int round)
        {
            if (includeFinalDesignRound && round > _baseRoundCount)
                return "finaldesign";

            return round <= samplingIterations ? "sampling" : "optimization";
        }

        private string GetConfiguredConditionId()
        {
            return setConditionIdFromMode ? GetDefaultConditionIdForMode() : conditionId;
        }

        private string GetDefaultConditionIdForMode()
        {
            if (conditionMode == ConditionMode.Random)
                return "random";

            if (conditionMode == ConditionMode.Static)
                return "static";

            return "HITL MOBO";
        }

        private string ResolveObjectiveKey(string headerName)
        {
            if (fittsLawTask == null || string.IsNullOrWhiteSpace(headerName))
                return null;

            if (MatchesObjectiveHeader(headerName, fittsLawTask.aestheticsObjectiveKey))
                return fittsLawTask.aestheticsObjectiveKey;

            if (MatchesObjectiveHeader(headerName, fittsLawTask.usabilityObjectiveKey))
                return fittsLawTask.usabilityObjectiveKey;

            return null;
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

        private static T FindSceneObject<T>() where T : UnityEngine.Object
        {
            foreach (T candidate in Resources.FindObjectsOfTypeAll<T>())
            {
                if (candidate is Component component)
                {
                    if (component.gameObject != null && component.gameObject.scene.IsValid())
                        return candidate;
                }
            }

            return null;
        }

        private static string ResolveContextValue(string value)
        {
            return string.IsNullOrWhiteSpace(value) ? "-1" : value.Trim();
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }
    }
}
