using System;
using System.Collections;
using System.Collections.Generic;
using QuestionnaireToolkit.Scripts;
using UnityEngine;
using UnityEngine.UI;

namespace BOforUnity.Scripts
{
    /// Fraction BO interface (the parameter value itself is the fraction, so the configured bounds narrow
    /// the range, e.g. TargetSize in [0.3, 0.4] keeps sizes within 30-40% of the size range):
    ///   u_size = value of the parameter named sizeParameterKey (else the first parameter), clamped to [0,1]
    ///   u_ecc  = value of the parameter named eccentricityParameterKey (else the second parameter), clamped to [0,1]
    /// Runtime mapping (with tighter size range for higher difficulty):
    ///   size = lerp(sizeMinAbs, sizeMaxTight, u_size)
    ///   ecc  = lerp(0, EccMaxForSize(size), u_ecc)
    /// where sizeMaxTight = lerp(sizeMinAbs, GlobalSizeMax(), sizeRangeFactor) and sizeRangeFactor∈[0,1].
    public class TargetClickerEvaluator : MonoBehaviour
    {
        [Header("UI")]
        public RectTransform playArea;
        public Button targetButton;
        public RectTransform targetRect;

        [Header("Difficulty")]
        [Min(0.01f)] public float sizeMinAbs = 0.05f;     // absolute minimum UI scale
        [Range(0f,1f)] public float sizeRangeFactor = 0.25f; // 0=all sizes ≈ sizeMinAbs, 1=full range

        [Header("Design Parameters (runtime)")]
        [Min(0.01f)] public float size = 1.0f;            // derived from u_size
        [Min(0f)]    public float eccentricity = 100f;    // derived from u_ecc

        [Header("BO Keys")]
        [Tooltip("Parameter that controls the target size. Empty: the first parameter in the manager's list.")]
        public string sizeParameterKey = "TargetSize";
        [Tooltip("Parameter that controls the eccentricity. Empty: the second parameter in the manager's list.")]
        public string eccentricityParameterKey = "TargetEccentricity";
        [Tooltip("Objective that receives the click times. Empty: the second objective in the manager's list.")]
        public string clickTimeObjectiveKey = "AverageClickTimeMS";

        [Header("Outputs")]
        public List<float> clickTimes;

        public int maxIterations = 3;
        public int currRound = 1;

        public QTQuestionnaireManager qtManager;

        private BoForUnityManager boManager;
        private bool _clicked;
        private float _t0;

        private void Awake()
        {
            if (targetButton && !targetRect) targetRect = targetButton.GetComponent<RectTransform>();
            if (targetButton) targetButton.gameObject.SetActive(false);
            if (clickTimes == null) clickTimes = new List<float>();

            boManager = BoParameterReader.FindManager();

            StartCoroutine(StartGame());
        }

        private IEnumerator StartGame()
        {
            // Read normalized params from BO
            float u_size = 0.5f, u_ecc = 0.5f;
            if (BoParameterReader.TryGetFraction(boManager, sizeParameterKey, 0, out var normalizedSize))
            {
                u_size = normalizedSize;
            }
            if (BoParameterReader.TryGetFraction(boManager, eccentricityParameterKey, 1, out var normalizedEccentricity))
            {
                u_ecc = normalizedEccentricity;
            }

            // Wait one frame so RectTransforms have valid geometry
            yield return null;

            // Compute tight upper bound for size
            float fullSizeMax   = GlobalSizeMax();
            float sizeMaxTight  = Mathf.Lerp(Mathf.Min(sizeMinAbs, fullSizeMax), fullSizeMax, Mathf.Clamp01(sizeRangeFactor));
            float sizeMinClamped= Mathf.Min(sizeMinAbs, sizeMaxTight);

            // Map normalized → feasible scene parameters (tighter range)
            size = Mathf.Lerp(sizeMinClamped, sizeMaxTight, u_size);

            float eccMax = EccMaxForSize(size);
            eccentricity = Mathf.Lerp(0f, eccMax, u_ecc);

            yield return RunGame();
        }

        private IEnumerator RunGame()
        {
            currRound = 1;
            while (currRound <= maxIterations)
            {
                _clicked = false;

                if (!targetButton || !targetRect)
                {
                    Debug.LogError("TargetClickerEvaluator: targetButton/targetRect missing.");
                    yield break;
                }

                targetRect.localScale = Vector3.one * size;
                PlaceTarget(eccentricity);

                targetButton.onClick.RemoveAllListeners();
                targetButton.onClick.AddListener(() => _clicked = true);
                targetButton.gameObject.SetActive(true);

                _t0 = Time.realtimeSinceStartup;
                while (!_clicked) yield return null;

                targetButton.gameObject.SetActive(false);
                clickTimes.Add((Time.realtimeSinceStartup - _t0) * 1000f);

                currRound++;
            }

            // set the click times as f2 (example)
            if (!TrySetSecondObjectiveValues(clickTimes))
            {
                Debug.LogWarning("TargetClickerEvaluator: Could not assign click times to the second valid objective.");
            }

            // call the questionnaire to receive the user feedback for perceived difficulty as f1
            if (qtManager) qtManager.StartQuestionnaire();
        }

        private bool TrySetSecondObjectiveValues(List<float> values)
        {
            if (boManager == null || boManager.objectives == null)
                return false;

            if (!string.IsNullOrWhiteSpace(clickTimeObjectiveKey))
            {
                string key = clickTimeObjectiveKey.Trim();
                foreach (var candidate in boManager.objectives)
                {
                    if (candidate != null && candidate.value != null &&
                        string.Equals(candidate.key?.Trim(), key, StringComparison.OrdinalIgnoreCase))
                    {
                        candidate.value.values = values ?? new List<float>();
                        return true;
                    }
                }

                Debug.LogWarning($"TargetClickerEvaluator: objective '{key}' not found; using the second objective.");
            }

            int seenValid = 0;
            for (int i = 0; i < boManager.objectives.Count; i++)
            {
                var objective = boManager.objectives[i];
                if (objective == null || objective.value == null || string.IsNullOrWhiteSpace(objective.key))
                    continue;

                if (seenValid == 1)
                {
                    objective.value.values = values ?? new List<float>();
                    return true;
                }
                seenValid++;
            }

            return false;
        }

        private void PlaceTarget(float eccPx)
        {
            RectTransform area = playArea ? playArea : (targetRect.parent as RectTransform);
            if (!area)
            {
                targetRect.anchoredPosition = Vector2.zero;
                return;
            }

            Vector2 dir = UnityEngine.Random.insideUnitCircle.normalized;
            if (dir.sqrMagnitude < 1e-6f) dir = Vector2.right;
            Vector2 pos = dir * eccPx;

            Vector2 half = area.rect.size * 0.5f;
            Vector2 sizeHalf = targetRect.rect.size * (0.5f * targetRect.localScale.x);
            Vector2 min = -half + sizeHalf;
            Vector2 max =  half - sizeHalf;
            pos = new Vector2(Mathf.Clamp(pos.x, min.x, max.x), Mathf.Clamp(pos.y, min.y, max.y));

            targetRect.anchoredPosition = pos;
        }

        // ── Geometry helpers ─────────────────────────────────────────────────

        private float GlobalSizeMax()
        {
            RectTransform area = playArea ? playArea : (targetRect ? targetRect.parent as RectTransform : null);
            if (!area || !targetRect) return 1f;

            Vector2 half = area.rect.size * 0.5f;
            float w = targetRect.rect.width;
            float h = targetRect.rect.height;

            float sx = 2f * half.x / w;
            float sy = 2f * half.y / h;
            return Mathf.Max(0.01f, Mathf.Min(sx, sy));
        }

        private float EccMaxForSize(float sizeVal)
        {
            RectTransform area = playArea ? playArea : (targetRect ? targetRect.parent as RectTransform : null);
            if (!area || !targetRect) return 0f;

            Vector2 half = area.rect.size * 0.5f;
            float w = targetRect.rect.width;
            float h = targetRect.rect.height;

            float x = half.x - 0.5f * w * sizeVal;
            float y = half.y - 0.5f * h * sizeVal;
            return Mathf.Max(0f, Mathf.Min(x, y));
        }
    }
    /// <summary>
    /// Reads design parameters for the demo scripts: by key when one is configured (falling back to the
    /// position in the manager's list), normalized to [0,1] with the parameter's configured bounds.
    /// </summary>
    public static class BoParameterReader
    {
        /// <summary>The persistent manager, or (before its Awake) any manager in the scene.</summary>
        public static BoForUnityManager FindManager(BoForUnityManager preferred = null)
        {
            if (BoForUnityManager.Instance != null)
                return BoForUnityManager.Instance;

            return preferred != null ? preferred : UnityEngine.Object.FindAnyObjectByType<BoForUnityManager>();
        }

        public static bool TryGetParameter(
            BoForUnityManager manager,
            string key,
            int fallbackIndex,
            out ParameterEntry entry)
        {
            entry = null;
            if (manager == null || manager.parameters == null)
                return false;

            if (!string.IsNullOrWhiteSpace(key))
            {
                string trimmedKey = key.Trim();
                foreach (StringComparison comparison in new[] { StringComparison.Ordinal, StringComparison.OrdinalIgnoreCase })
                {
                    foreach (var candidate in manager.parameters)
                    {
                        if (candidate != null && candidate.value != null &&
                            string.Equals(candidate.key?.Trim(), trimmedKey, comparison))
                        {
                            entry = candidate;
                            return true;
                        }
                    }
                }

                Debug.LogWarning(
                    $"BoParameterReader: parameter '{trimmedKey}' not found; using parameter #{fallbackIndex + 1} in the list."
                );
            }

            if (fallbackIndex < 0)
                return false;

            int seenValid = 0;
            foreach (var candidate in manager.parameters)
            {
                if (candidate == null || candidate.value == null || string.IsNullOrWhiteSpace(candidate.key))
                    continue;

                if (seenValid == fallbackIndex)
                {
                    entry = candidate;
                    return true;
                }

                seenValid++;
            }

            return false;
        }

        /// <summary>
        /// The value mapped to [0,1] by the configured bounds (clamped). Bounds that coincide leave the value
        /// as it is (clamped to [0,1]).
        /// </summary>
        public static float Normalize(ParameterArgs parameter)
        {
            if (parameter == null)
                return 0.5f;

            float lo = Mathf.Min(parameter.lowerBound, parameter.upperBound);
            float hi = Mathf.Max(parameter.lowerBound, parameter.upperBound);
            if (!(hi - lo > 1e-12f))
                return Mathf.Clamp01(parameter.Value);

            return Mathf.Clamp01((parameter.Value - lo) / (hi - lo));
        }

        /// <summary>The raw value clamped to [0,1], for parameters whose value is itself a fraction.</summary>
        public static bool TryGetFraction(BoForUnityManager manager, string key, int fallbackIndex, out float fraction)
        {
            fraction = 0.5f;
            if (!TryGetParameter(manager, key, fallbackIndex, out var entry))
                return false;

            fraction = Mathf.Clamp01(entry.value.Value);
            return true;
        }

        public static bool TryGetNormalized(BoForUnityManager manager, string key, int fallbackIndex, out float normalized)
        {
            normalized = 0.5f;
            if (!TryGetParameter(manager, key, fallbackIndex, out var entry))
                return false;

            normalized = Normalize(entry.value);
            return true;
        }
    }
}
