using System;
using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace QuestionnaireToolkit.Scripts
{
    /// <summary>
    /// This class is used to manage (add) buttons of a Button question item.
    /// </summary>
    [ExecuteInEditMode]
    public class QTButton : MonoBehaviour
    {
        // list that contains all UI buttons that are added to the question item
        private List<GameObject> buttons = new List<GameObject>();

        private QTQuestionnaireManager _questionnaireManager;
        private QTQuestionPageManager _questionPageManager;
        
//#if UNITY_EDITOR
        private void Start()
        {
            _questionnaireManager = transform.parent.parent.parent.parent.parent.GetComponent<QTQuestionnaireManager>();
            try
            {
                if (!Application.isPlaying)
                {
                    
                }
                _questionPageManager = transform.parent.parent.parent.parent.GetComponent<QTQuestionPageManager>();
            }
            catch (Exception)
            {
                // ignored
            }
        }

        /// <summary>
        /// Adds a new TextMeshPro button to the question item.
        /// </summary>
        public void AddOption(bool import = false, QTQuestionnaireManager manager = null)
        {
            if (import)
            {
                _questionnaireManager = manager;
            }
            
            // Instantiate a UI button inside the question item
            var o = Resources.Load("QuestionnaireToolkitPrefabs/ButtonItem");
            var inWorldSpace = _questionnaireManager.spawnObjectsInWorldSpace;
            var g = Instantiate(o, transform, inWorldSpace) as GameObject;
            buttons.Add(g);
            
            // If in VR mode set position and scaling as needed
            if (inWorldSpace)
            {
                var rectTransform = g.GetComponent<RectTransform>();
                rectTransform.anchoredPosition3D = Vector3.zero;
                rectTransform.localScale = Vector3.one;
            }
            g.name = "Button-" + buttons.Count;
        }
//#endif
    }
}
