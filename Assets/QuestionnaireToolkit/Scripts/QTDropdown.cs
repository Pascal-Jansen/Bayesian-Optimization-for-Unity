using System;
using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace QuestionnaireToolkit.Scripts
{
    /// <summary>
    /// This class is used to manage (add, remove, edit) options of a Dropdown question item and to edit its general properties.
    /// </summary>
    [ExecuteInEditMode]
    public class QTDropdown : MonoBehaviour
    {
        // bool to determine if this question items must be answered
        public bool answerRequired = true;
        // the name of the item which appears in the results file as header entry
        public string headerName = "";
        private string _oldHeaderName = "";
        
        // the displayed question of this question item
        public string question = "";
        // list of Dropdown OptionData to manage the dropdown options
        public List<TMP_Dropdown.OptionData> options = new List<TMP_Dropdown.OptionData>();

        // visible field to edit the text of a dropdown option
        public string optionText = "";
        // visible field to add an image to a dropdown option
        public Sprite optionImage = null;

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
                    
                    // update options list
                    transform.GetChild(1).GetComponent<TMP_Dropdown>().options = options;
                }
            }
            catch (Exception)
            {
                // ignored
            }
        }
        
        /// <summary>
        /// Adds an option to the dropdown question item.
        /// </summary>
        public void AddOption(bool import = false, bool mandatory = false, string optionsT = null, Sprite img = null)
        {
            if (import)
            {
                answerRequired = mandatory;
                optionText = optionsT;
                optionImage = img;
            }
            
            TMP_Dropdown.OptionData optionData = new TMP_Dropdown.OptionData();
            
            if (optionText.Equals(""))
            {
                optionData.text = "Option " + (options.Count+1);
            }
            else
            {
                optionData.text = optionText;
            }
            optionData.image = optionImage;
            options.Add(optionData);
            transform.GetChild(1).GetComponent<TMP_Dropdown>().options = options;
            
            optionText = "";
            optionImage = null;
            
            if(import) OnValidate();
        }

        /// <summary>
        /// Shows the text and the image of a selected option.
        /// </summary>
        public void OptionSelected(int sel)
        {
            optionText = options[sel].text;
            optionImage = options[sel].image;
        }

        /// <summary>
        /// Deletes an option from the dropdown question item.
        /// </summary>
        public void DeleteItem(int listCount, int sel)
        {
            if (listCount < options.Count) // item was removed
            {
                options.RemoveAt(sel);
                transform.GetChild(1).GetComponent<TMP_Dropdown>().options = options;
            }
        }

        /// <summary>
        /// Reorders the option list according to the reorderable list in the editor.
        /// </summary>
        public void ReorderItems(int listCount, int sel)
        {
            if (listCount == options.Count)
            {
                transform.GetChild(1).GetComponent<TMP_Dropdown>().options = options;
                //options[sel].transform.SetSiblingIndex(sel);
            }
        }
//#endif
    }
}
