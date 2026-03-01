using System;
using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace QuestionnaireToolkit.Scripts
{
    /// <summary>
    /// This class is used to manage (add, remove, edit) options of a LinearScale question item and to edit its general properties.
    /// </summary>
    [ExecuteInEditMode]
    public class QTLinearScale : MonoBehaviour
    {
        // bool to determine if this question items must be answered
        public bool answerRequired = true;
        // the name of the item which appears in the results file as header entry
        public string headerName = "";
        private string _oldHeaderName = "";
        
        // the displayed question of this question item
        public string question = "";
        // list which contains the radio button options of this item
        public List<GameObject> options = new List<GameObject>();

        // the visible field to edit the displayed text below an option
        public string answerOption = "";
        // the visible field to edit the csv value of an answer option
        public string answerValue = "1";

        [HideInInspector]
        public int selectedIndex;
        private const string PriorRatingHintObjectName = "PriorRatingHint";
        private readonly Dictionary<GameObject, Image> _priorHintImagesByOption = new Dictionary<GameObject, Image>();

        private QTQuestionnaireManager _questionnaireManager;
        private QTQuestionPageManager _questionPageManager;
        
//#if UNITY_EDITOR
        private void Start()
        {
            _questionnaireManager = transform.parent.parent.parent.parent.parent.GetComponent<QTQuestionnaireManager>();
            try
            {
                headerName = name.Split('_')[1];
                _oldHeaderName = headerName;
                question = name.Split('_')[2];
                transform.GetChild(0).GetComponent<TextMeshProUGUI>().text = question;
                name = name.Split('_')[0] + "_" + name.Split('_')[1];
                
                if (!Application.isPlaying)
                {
                    //Editor mode
                    _questionnaireManager.BuildHeaderItems();
            
                    _questionPageManager = transform.parent.parent.parent.parent.GetComponent<QTQuestionPageManager>();
                    if (_questionPageManager.automaticFill)
                    {
                        // if automatic fill is enabled in the page manager, then 5 options will be added by default.
                        for (var i = 0; i < 5; i++)
                        {
                            AddOption();
                        }
                    }
                }
                else
                {
                    //Play mode
                    if (options.Count == 0)
                    {
                        answerRequired = false;
                        transform.GetChild(0).GetChild(0).gameObject.SetActive(answerRequired);
                    }
                }
            }
            catch (Exception)
            {
                // ignored
            }

            HidePriorRatingHint();
        }

        private void OnValidate()
        {
            try
            {
                if (!Application.isPlaying)
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
                }
            }
            catch (Exception)
            {
                // ignored
            }
        }

        /// <summary>
        /// Adds a new radio button option to the linear scale item.
        /// </summary>
        public void AddOption(bool import = false, QTQuestionnaireManager manager = null, bool mandatory = false, string a_value = null, string a_option = null)
        {
            if (import)
            {
                answerRequired = mandatory;
                answerValue = a_value;
                answerOption = a_option;
                _questionnaireManager = manager;
            }
            
            // Instantiate a new radio button option inside the question item
            var contentParentTransform = transform.GetChild(1);
            var o = Resources.Load("QuestionnaireToolkitPrefabs/LinearScaleOption");
            var inWorldSpace = _questionnaireManager.spawnObjectsInWorldSpace;
            var g = Instantiate(o, contentParentTransform, inWorldSpace) as GameObject;
            options.Add(g);
            
            if (answerValue.Equals(""))
            {
                answerValue = "" + options.Count;
            }
            /*
            if (answerOption.Equals(""))
            {
                g.name = answerValue + "_Option " + options.Count;
            }
            */
            else
            {
                g.name = answerValue + "_" + answerOption;
            }
            g.transform.GetChild(1).gameObject.GetComponent<TextMeshProUGUI>().text = g.name.Split('_')[1];
            g.GetComponent<Toggle>().group = contentParentTransform.GetComponent<ToggleGroup>();
            
            // If in VR mode set position and scaling as needed
            if (inWorldSpace)
            {
                var rectTransform = g.GetComponent<RectTransform>();
                rectTransform.anchoredPosition3D = Vector3.zero;
                rectTransform.localScale = Vector3.one;
            }
            answerOption = "";
            answerValue = "" + (options.Count + 1);
            
            // Resize all options to fit the parent width
            #if UNITY_EDITOR
            ResizeOptions();
            #endif
            
            if(import) OnValidate();
        }

        /// <summary>
        /// Set the answerOption and answerValue field with the values of the current selected option determined by the given index.
        /// </summary>
        public void OptionSelected(int sel)
        {
            answerOption = options[sel].name.Split('_')[1];
            answerValue = options[sel].name.Split('_')[0];
        }
        
        /// <summary>
        /// Edits the selected option with the given values in the answerValue and answerOption fields.
        /// </summary>
        public void EditOption()
        {
            if (selectedIndex > options.Count - 1) return;
            
            var o = options[selectedIndex];
            //if (answerOption.Equals("") || answerOption.Equals(o.name) || answerValue.Equals("")) return;
            if (answerOption.Equals(o.name) || answerValue.Equals("")) return;
            o.name = answerValue + "_" + answerOption;
            o.transform.GetChild(1).gameObject.GetComponent<TextMeshProUGUI>().text = o.name.Split('_')[1];
            answerOption = "";
            answerValue = "" + (options.Count + 1);
        }

        /// <summary>
        /// Deletes an item from the options list of this question item.
        /// </summary>
        public void DeleteItem(int listCount, int sel)
        {
            if (listCount < options.Count) // item was removed
            {
                var optionToDelete = options[sel];
                _priorHintImagesByOption.Remove(optionToDelete);
                DestroyImmediate(options[sel]);
                options.RemoveAt(sel);
                #if UNITY_EDITOR
                ResizeOptions();
                #endif
            }
        }

        /// <summary>
        /// Reorders the option list of this question item based on the reorderable list in the editor.
        /// </summary>
        public void ReorderItems(int listCount, int sel)
        {
            if (listCount == options.Count)
            {
                options[sel].transform.SetSiblingIndex(sel);
                for(var i  = 0; i < options.Count; i++)
                {
                    options[i].name = (i+1) + "_" + options[i].name.Split('_')[1];
                }
            }
        }

        public void ApplyPriorRatingHint(bool enabled, string priorAnswerValue, float alpha = 0.16f)
        {
            HidePriorRatingHint();
            if (!enabled || string.IsNullOrWhiteSpace(priorAnswerValue))
                return;

            string targetValue = priorAnswerValue.Trim();
            foreach (var option in options)
            {
                if (option == null)
                    continue;

                string optionValue = GetOptionValue(option);
                if (!string.Equals(optionValue, targetValue, StringComparison.OrdinalIgnoreCase))
                    continue;

                var hintImage = EnsurePriorRatingHint(option);
                if (hintImage == null)
                    return;

                float subtleAlpha = Mathf.Clamp(alpha, 0.05f, 0.45f);
                Color markerColor = new Color(0f, 0f, 0f, subtleAlpha);
                var backgroundImage = option.transform.childCount > 0
                    ? option.transform.GetChild(0).GetComponent<Image>()
                    : null;
                if (backgroundImage != null)
                {
                    Color baseColor = backgroundImage.color;
                    float luminance = (0.2126f * baseColor.r) + (0.7152f * baseColor.g) + (0.0722f * baseColor.b);
                    markerColor = luminance < 0.5f
                        ? new Color(1f, 1f, 1f, subtleAlpha)
                        : new Color(0f, 0f, 0f, subtleAlpha);
                }

                hintImage.color = markerColor;
                hintImage.gameObject.SetActive(true);
                break;
            }
        }

        public void HidePriorRatingHint()
        {
            foreach (var hintImage in _priorHintImagesByOption.Values)
            {
                if (hintImage != null)
                {
                    hintImage.gameObject.SetActive(false);
                }
            }
        }

        private static string GetOptionValue(GameObject option)
        {
            if (option == null || string.IsNullOrWhiteSpace(option.name))
                return string.Empty;

            return option.name.Split('_')[0].Trim();
        }

        private Image EnsurePriorRatingHint(GameObject option)
        {
            if (option == null)
                return null;

            if (_priorHintImagesByOption.TryGetValue(option, out var existingHint) && existingHint != null)
                return existingHint;

            RectTransform parent = null;
            Image backgroundImage = null;
            if (option.transform.childCount > 0)
            {
                parent = option.transform.GetChild(0) as RectTransform;
                backgroundImage = option.transform.GetChild(0).GetComponent<Image>();
            }
            if (parent == null)
            {
                parent = option.GetComponent<RectTransform>();
            }
            if (parent == null)
                return null;

            var existingTransform = parent.Find(PriorRatingHintObjectName) as RectTransform;
            Image hintImage;
            if (existingTransform != null)
            {
                hintImage = existingTransform.GetComponent<Image>();
                if (hintImage == null)
                {
                    hintImage = existingTransform.gameObject.AddComponent<Image>();
                }
            }
            else
            {
                var hintGo = new GameObject(PriorRatingHintObjectName, typeof(RectTransform), typeof(Image));
                var hintRect = hintGo.GetComponent<RectTransform>();
                hintRect.SetParent(parent, false);
                hintRect.SetAsFirstSibling();
                hintImage = hintGo.GetComponent<Image>();
            }

            var rectTransform = hintImage.rectTransform;
            rectTransform.anchorMin = Vector2.zero;
            rectTransform.anchorMax = Vector2.one;
            rectTransform.sizeDelta = Vector2.zero;
            rectTransform.anchoredPosition = Vector2.zero;

            if (backgroundImage != null)
            {
                hintImage.sprite = backgroundImage.sprite;
                hintImage.type = backgroundImage.type;
                hintImage.preserveAspect = backgroundImage.preserveAspect;
            }

            hintImage.raycastTarget = false;
            hintImage.maskable = true;
            _priorHintImagesByOption[option] = hintImage;
            return hintImage;
        }

#if UNITY_EDITOR
        /// <summary>
        /// Resizes all options of a linear scale to always fill the available horizontal space.
        /// </summary>
        private void ResizeOptions()
        {
            EditorApplication.delayCall += () =>
            {
                var parentWidth = GetComponent<RectTransform>().sizeDelta.x;
                foreach (var option in options)
                {
                    option.GetComponent<LayoutElement>().minWidth = parentWidth / options.Count;
                }
            };
        }
#endif
//#endif
    }
}
