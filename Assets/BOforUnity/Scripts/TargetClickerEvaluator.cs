using System.Collections;
using System.Collections.Generic;
using QuestionnaireToolkit.Scripts;
using UnityEngine;
using UnityEngine.UI;

namespace BOforUnity.Scripts
{
    /// Normalized BO interface:
    ///   param[0] = u_size ∈ [0,1]
    ///   param[1] = u_ecc  ∈ [0,1]
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

        [Header("Outputs")]
        public List<float> clickTimes;

        public int maxRounds = 3;
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

            boManager = GameObject.FindWithTag("BOforUnityManager")?.GetComponent<BoForUnityManager>();

            StartCoroutine(StartGame());
        }

        private IEnumerator StartGame()
        {
            // Read normalized params from BO
            float u_size = 0.5f, u_ecc = 0.5f;
            if (boManager && boManager.parameters.Count >= 2)
            {
                u_size = Mathf.Clamp01(boManager.parameters[0].value.Value);
                u_ecc  = Mathf.Clamp01(boManager.parameters[1].value.Value);
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
            while (currRound <= maxRounds)
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
            if (boManager && boManager.objectives.Count > 1)
                boManager.objectives[1].value.values = clickTimes;

            // call the questionnaire to receive the user feedback for perceived difficulty as f1
            if (qtManager) qtManager.StartQuestionnaire();
        }

        private void PlaceTarget(float eccPx)
        {
            RectTransform area = playArea ? playArea : (targetRect.parent as RectTransform);
            if (!area)
            {
                targetRect.anchoredPosition = Vector2.zero;
                return;
            }

            Vector2 dir = Random.insideUnitCircle.normalized;
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
}