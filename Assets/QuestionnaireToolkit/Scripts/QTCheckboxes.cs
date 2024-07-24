using System;
using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace QuestionnaireToolkit.Scripts
{
    /// <summary>
    /// This class is used to manage (add, remove, edit) options of a Checkboxes question item and to edit its general properties.
    /// </summary>
    [ExecuteInEditMode]
    public class QTCheckboxes : MonoBehaviour
    {
        // bool to determine if this question items must be answered
        public bool answerRequired = true;
        // the name of the item which appears in the results file as header entry
        public string headerName = "";
        private string _oldHeaderName = "";
        
        // the displayed question of this question item
        public string question = "";
        // bool to enable the automatic addition of a 'Other' option at the end of the item
        public bool includeOtherOption = true;
        // list which contains the checkboxes options of this item
        public List<GameObject> options = new List<GameObject>();

        // the visible field to edit the text of a selected option
        public string answerOption = "";
        // the visible field to edit the csv value of a answer option
        public string answerValue = "1";

        [HideInInspector]
        public int selectedIndex;

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
                        // if automatic fill is enabled in the page manager, then three options will be added by default.
                        for (var i = 0; i < 3; i++)
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
        }

        private void OnValidate()
        {
            try
            {
                if (Application.isPlaying)
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
                    
                    // update others field visibility
                    transform.GetChild(1).GetChild(options.Count).gameObject.SetActive(includeOtherOption);
                }
            }
            catch (Exception)
            {
                // ignored
            }
        }

        /// <summary>
        /// Adds a new checkbox option to the question item.
        /// </summary>
        public void AddOption(bool import = false, QTQuestionnaireManager manager = null, bool mandatory = false, string a_value = null, string a_option = null, bool other_option = true)
        {
            if (import)
            {
                _questionnaireManager = manager;
                answerRequired = mandatory;
                includeOtherOption = other_option;
                answerValue = a_value;
                answerOption = a_option;
            }
            
            // Instantiate a new checkbox option inside the question item
            var contentParentTransform = transform.GetChild(1);
            var o = Resources.Load("QuestionnaireToolkitPrefabs/CheckboxesOption");
            var inWorldSpace = _questionnaireManager.spawnObjectsInWorldSpace;
            var g = Instantiate(o, contentParentTransform, inWorldSpace) as GameObject;
            options.Add(g);
            g.transform.SetSiblingIndex(options.Count-1);

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
            
            // If in VR mode set position and scaling as needed
            if (inWorldSpace)
            {
                var rectTransform = g.GetComponent<RectTransform>();
                rectTransform.anchoredPosition3D = Vector3.zero;
                rectTransform.localScale = Vector3.one;
            }
            answerOption = "";
            answerValue = "" + (options.Count + 1);
            
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
                DestroyImmediate(options[sel]);
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
//#endif
    }
}
