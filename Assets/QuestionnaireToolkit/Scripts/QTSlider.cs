using System;
using System.Windows.Forms;
using TMPro;
using UnityEditor;
using UnityEngine;
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

//#if UNITY_EDITOR
        private void Start()
        {
            _questionnaireManager = transform.parent.parent.parent.parent.parent.GetComponent<QTQuestionnaireManager>();

            try
            {
                //Editor mode
                headerName = name.Split('_')[1];
                _oldHeaderName = headerName;
                question = name.Split('_')[2];
                transform.GetChild(0).GetComponent<TextMeshProUGUI>().text = question;
                name = name.Split('_')[0] + "_" + name.Split('_')[1];
                
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
//#endif
    }
}

