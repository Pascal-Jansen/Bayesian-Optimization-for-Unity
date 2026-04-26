using System;
using System.Windows.Forms;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using Application = UnityEngine.Application;

namespace QuestionnaireToolkit.Scripts
{
    /// <summary>
    /// This class is used to manage (add, remove, edit) rows/columns of a Slider question item and to edit its general properties.
    /// </summary>
    [ExecuteInEditMode]
    public class QTSlider : MonoBehaviour
    {
        // bool to determine if this question items must be answered
        public bool answerRequired = true;
        // the name of the item which appears in the results file as header entry
        public string headerName = "";
        private string _oldHeaderName = "";
        
        // the displayed question of this question item
        public string question = "";

        [Header("Slider Settings")] 
        public int minValue = 0;
        public int maxValue = 100;
        public bool wholeNumbers = true;
        
        [Header("Slider Visuals")]
        public bool showPanels = true;
        public bool showIntermediatePanels = true;
        public bool automaticLabelNames = true;
        public string labelZero = "0";
        public string labelQuarter = "25";
        public string labelHalf = "50";
        public string labelThreeQuarters = "75";
        public string labelFull = "100";
        
        private QTQuestionnaireManager _questionnaireManager;
        private UnityEngine.UI.Slider sliderScript;
        private GameObject panels;
        private GameObject panelsIntermediate;
        private TextMeshProUGUI zero;
        private TextMeshProUGUI quarter;
        private TextMeshProUGUI half;
        private TextMeshProUGUI threeQuarter;
        private TextMeshProUGUI full;
        private RectTransform priorRatingHintRect;
        private Image priorRatingHintImage;
        private const string PriorRatingHintObjectName = "PriorRatingHint";
        private bool runtimeAnswered;
        private QTSliderPointerRelay pointerRelay;

//#if UNITY_EDITOR
        private void Start()
        {
            _questionnaireManager = transform.parent.parent.parent.parent.parent.GetComponent<QTQuestionnaireManager>();

            try
            {
                _oldHeaderName = headerName;
                transform.GetChild(0).GetComponent<TextMeshProUGUI>().text = question;
                
                if (!Application.isPlaying)
                {
                    _questionnaireManager.BuildHeaderItems();
                }
            }
            catch (Exception)
            {
                // ignored
            }

            GetReferences();
            HidePriorRatingHint();
            if (Application.isPlaying)
            {
                RegisterRuntimeTracking();
                ResetRuntimeAnswerState();
            }
        }

        private void GetReferences()
        {
            sliderScript = transform.GetChild(1).GetComponent<UnityEngine.UI.Slider>();
            
            panels = transform.GetChild(1).GetChild(1).gameObject;
            panelsIntermediate = transform.GetChild(1).GetChild(2).gameObject;
            
            zero = transform.GetChild(1).GetChild(5).GetComponent<TextMeshProUGUI>();
            quarter = transform.GetChild(1).GetChild(6).GetComponent<TextMeshProUGUI>();
            half = transform.GetChild(1).GetChild(7).GetComponent<TextMeshProUGUI>();
            threeQuarter = transform.GetChild(1).GetChild(8).GetComponent<TextMeshProUGUI>();
            full = transform.GetChild(1).GetChild(9).GetComponent<TextMeshProUGUI>();
        }
        
        private void OnValidate()
        {
            try
            {
                if (!Application.isPlaying)
                {
                    UpdateSlider();
                }
            }
            catch (Exception)
            {
                // ignored
            }
        }

        private void UpdateSlider()
        {
            // update headerName field
            if (!_oldHeaderName.Equals(headerName))
            {
                _oldHeaderName = headerName;
                name = name.Split('_')[0] + "_" + headerName;
                _questionnaireManager.BuildHeaderItems();
            }
            
            // update question field
            transform.GetChild(0).GetComponent<TextMeshProUGUI>().text = question;
            
            // update required bool
            transform.GetChild(0).GetChild(0).gameObject.SetActive(answerRequired);

            // update slider settings
            sliderScript.minValue = minValue;
            sliderScript.maxValue = maxValue;
            sliderScript.wholeNumbers = wholeNumbers;
                    
            // update slider visuals
            panels.SetActive(showPanels);
            panelsIntermediate.SetActive(showIntermediatePanels);
            if (!automaticLabelNames) // manual labels
            {
                zero.text = labelZero;
                quarter.text = labelQuarter;
                half.text = labelHalf;
                threeQuarter.text = labelThreeQuarters;
                full.text = labelFull;
            }
            else // automatic labels
            {
                var range = maxValue - minValue;
                zero.text = minValue.ToString();
                quarter.text = (minValue + (range * 0.25F)).ToString();
                half.text = (minValue + (range * 0.5F)).ToString();
                threeQuarter.text = (minValue + (range * 0.75F)).ToString();
                full.text = maxValue.ToString();
            }
        }
        
        public void ImportSliderValues( bool mandatory, int minVal, int maxVal, bool wN, bool sP, bool siP, bool autoLabels, string lzero, string lquarter, string lhalf, string lthreequarters, string lfull)
        {
            GetReferences();
            
            answerRequired = mandatory;
            
            minValue = minVal;
            maxValue = maxVal;
            wholeNumbers = wN;
            
            showPanels = sP;
            showIntermediatePanels = siP;

            automaticLabelNames = autoLabels;
            
            labelZero = lzero;
            labelQuarter = lquarter;
            labelHalf = lhalf;
            labelThreeQuarters = lthreequarters;
            labelFull = lfull;
            
            UpdateSlider();
        }

        public void ApplyPriorRatingHint(bool enabled, float priorValue, float alpha = 0.16f)
        {
            if (!enabled || sliderScript == null || maxValue <= minValue)
            {
                HidePriorRatingHint();
                return;
            }

            if (!EnsurePriorRatingHint())
                return;

            float clampedValue = Mathf.Clamp(priorValue, minValue, maxValue);
            float normalized = Mathf.InverseLerp(minValue, maxValue, clampedValue);
            bool horizontal =
                sliderScript.direction == UnityEngine.UI.Slider.Direction.LeftToRight ||
                sliderScript.direction == UnityEngine.UI.Slider.Direction.RightToLeft;

            if (sliderScript.direction == UnityEngine.UI.Slider.Direction.RightToLeft ||
                sliderScript.direction == UnityEngine.UI.Slider.Direction.TopToBottom)
            {
                normalized = 1f - normalized;
            }

            if (horizontal)
            {
                priorRatingHintRect.anchorMin = new Vector2(normalized, 0f);
                priorRatingHintRect.anchorMax = new Vector2(normalized, 1f);
                priorRatingHintRect.sizeDelta = new Vector2(2f, 0f);
            }
            else
            {
                priorRatingHintRect.anchorMin = new Vector2(0f, normalized);
                priorRatingHintRect.anchorMax = new Vector2(1f, normalized);
                priorRatingHintRect.sizeDelta = new Vector2(0f, 2f);
            }

            priorRatingHintRect.anchoredPosition = Vector2.zero;
            priorRatingHintRect.gameObject.SetActive(true);

            float subtleAlpha = Mathf.Clamp(alpha, 0.05f, 0.45f);
            Color markerColor = new Color(0f, 0f, 0f, subtleAlpha);
            if (sliderScript.targetGraphic is Graphic targetGraphic)
            {
                Color baseColor = targetGraphic.color;
                float luminance = (0.2126f * baseColor.r) + (0.7152f * baseColor.g) + (0.0722f * baseColor.b);
                markerColor = luminance < 0.5f
                    ? new Color(1f, 1f, 1f, subtleAlpha)
                    : new Color(0f, 0f, 0f, subtleAlpha);
            }
            priorRatingHintImage.color = markerColor;
        }

        public bool HasRuntimeAnswer()
        {
            return runtimeAnswered;
        }

        public void ResetRuntimeAnswerState()
        {
            runtimeAnswered = false;
        }

        public void HidePriorRatingHint()
        {
            if (priorRatingHintRect != null)
                priorRatingHintRect.gameObject.SetActive(false);
        }

        private void RegisterRuntimeTracking()
        {
            if (sliderScript == null)
                return;

            sliderScript.onValueChanged.RemoveListener(HandleSliderValueChanged);
            sliderScript.onValueChanged.AddListener(HandleSliderValueChanged);

            pointerRelay = sliderScript.GetComponent<QTSliderPointerRelay>();
            if (pointerRelay == null)
            {
                pointerRelay = sliderScript.gameObject.AddComponent<QTSliderPointerRelay>();
            }
            pointerRelay.owner = this;
        }

        private void OnDestroy()
        {
            if (sliderScript != null)
            {
                sliderScript.onValueChanged.RemoveListener(HandleSliderValueChanged);
            }
            if (pointerRelay != null && pointerRelay.owner == this)
            {
                pointerRelay.owner = null;
            }
        }

        private void HandleSliderValueChanged(float _)
        {
            runtimeAnswered = true;
        }

        private bool EnsurePriorRatingHint()
        {
            if (priorRatingHintRect != null && priorRatingHintImage != null)
                return true;

            RectTransform parent = null;
            if (sliderScript != null && sliderScript.handleRect != null)
                parent = sliderScript.handleRect.parent as RectTransform;
            if (parent == null && sliderScript != null && sliderScript.fillRect != null)
                parent = sliderScript.fillRect.parent as RectTransform;
            if (parent == null && sliderScript != null)
                parent = sliderScript.GetComponent<RectTransform>();
            if (parent == null)
                return false;

            var existing = parent.Find(PriorRatingHintObjectName) as RectTransform;
            if (existing != null)
            {
                priorRatingHintRect = existing;
                priorRatingHintImage = priorRatingHintRect.GetComponent<Image>();
                if (priorRatingHintImage == null)
                    priorRatingHintImage = priorRatingHintRect.gameObject.AddComponent<Image>();
            }
            else
            {
                var go = new GameObject(PriorRatingHintObjectName, typeof(RectTransform), typeof(Image));
                priorRatingHintRect = go.GetComponent<RectTransform>();
                priorRatingHintRect.SetParent(parent, false);
                priorRatingHintImage = go.GetComponent<Image>();
            }

            priorRatingHintImage.raycastTarget = false;
            priorRatingHintImage.maskable = true;
            return true;
        }

        private sealed class QTSliderPointerRelay : MonoBehaviour, IPointerDownHandler
        {
            public QTSlider owner;

            public void OnPointerDown(PointerEventData eventData)
            {
                if (owner != null)
                {
                    owner.runtimeAnswered = true;
                }
            }
        }
//#endif
    }
}
