using System;
using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace QuestionnaireToolkit.Scripts
{
    /// <summary>
    /// This class is used to set and validate the properties of a text input item.
    /// </summary>
    [ExecuteInEditMode]
    public class QTTextInput : MonoBehaviour
    {
        // bool to determine if this question items must be answered
        public bool answerRequired = true;
        // the name of the item which appears in the results file as header entry
        public string headerName = "";
        private string _oldHeaderName = "";
        
        // the displayed question of this question item
        public string question = "";
        // placeholder text to display inside the text field if it is empty
        public string placeholderText = "Enter text...";
        
        private QTQuestionnaireManager _questionnaireManager;
        
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
                    
                    // update placeholder text
                    transform.GetChild(1).GetChild(0).GetChild(0).GetComponent<TextMeshProUGUI>().text = placeholderText;
                }
            }
            catch (Exception)
            {
                // ignored
            }
        }

        public void ImportTextInputValues(bool mandatory, string pHolderText)
        {
            answerRequired = mandatory;
            placeholderText = pHolderText;
            OnValidate();
        }
//#endif
    }
}
