using System;
using System.Collections;
using System.Collections.Generic;
using QuestionnaireToolkit.Scripts.SimpleJSON;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace QuestionnaireToolkit.Scripts
{
    /// <summary>
    /// This class is used to manage (add, remove, edit) rows/columns of a CheckboxesGrid question item and to edit its general properties.
    /// </summary>
    [ExecuteInEditMode]
    public class QTCheckboxesGrid : MonoBehaviour
    {
        // bool to determine if this question items must be answered
        public bool answerRequired = true;
        // the name of the item which appears in the results file as header entry
        public string headerName = "";
        private string _oldHeaderName = "";
        
        // the displayed question of this question item
        public string question = "";
        
        // list to manage the rows and columns of the checkboxes grid
        [Header("Grid Content")]
        public List<string> rowTexts = new List<string>();
        public List<string> columnTexts = new List<string>();

        private int _oldRows = 0;
        private int _oldColumns = 0;
        
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
                    _questionPageManager = transform.parent.parent.parent.parent.GetComponent<QTQuestionPageManager>();
                    if (_questionPageManager.automaticFill && (rowTexts.Count == 0 || columnTexts.Count == 0))
                    {
                        // if automatic fill is enabled in the page manager, then a predefined 3x5 grid will be created.
                        _oldRows = 3;
                        _oldColumns = 5;
                        rowTexts = new List<string>() {"Row 1", "Row 2", "Row 3"};
                        columnTexts = new List<string>() {"Column 1", "Column 2", "Column 3", "Column 4", "Column 5"};
                        #if UNITY_EDITOR
                        BuildGrid();
                        #endif
                    }
                }
                else
                {
                    //Play mode
                    if (rowTexts.Count == 0 || columnTexts.Count == 0)
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
                    }
            
                    // update question field
                    transform.GetChild(0).GetComponent<TextMeshProUGUI>().text = question;
            
                    // update required bool
                    transform.GetChild(0).GetChild(0).gameObject.SetActive(answerRequired);

                    if (_oldRows != rowTexts.Count || _oldColumns != columnTexts.Count)
                    {
                        #if UNITY_EDITOR
                        BuildGrid();
                        #endif
                        _oldRows = rowTexts.Count;
                        _oldColumns = columnTexts.Count;
                    }
                    
                    // update row text
                    // update col text
                    UpdateGridTexts();
                }
            }
            catch (Exception)
            {
                // ignored
            }
        }

        public void ImportGrid(QTQuestionnaireManager manager, bool mandatory, JSONArray rTexts, JSONArray cTexts)
        {
            _questionnaireManager = manager;
            answerRequired = mandatory;
            rowTexts = ConvertToArrayList(rTexts);
            columnTexts = ConvertToArrayList(cTexts);
            
            var contentParentTransform = transform.GetChild(1);
            if (!_questionnaireManager) return;
            var inWorldSpace = _questionnaireManager.spawnObjectsInWorldSpace;
            // load grid content prefabs
            var gridText = Resources.Load("QuestionnaireToolkitPrefabs/GridText");
            var option = Resources.Load("QuestionnaireToolkitPrefabs/CheckboxesGridOption");

            // spawn the top left-most grid cell, which needs to be empty
            var empty = new GameObject("Empty", typeof(RectTransform));
            if (inWorldSpace)
            {
                var rectTransform = empty.GetComponent<RectTransform>();
                rectTransform.anchoredPosition3D = Vector3.zero;
                rectTransform.localScale = Vector3.one;
            }
            empty.transform.SetParent(contentParentTransform);

            // add columns
            foreach (var column in columnTexts)
            {
                SpawnGridOption(gridText, inWorldSpace, contentParentTransform, column, "Column");
            }

            // add rows
            foreach (var row in rowTexts)
            {
                SpawnGridOption(gridText, inWorldSpace, contentParentTransform, row, "Row");

                // add options within a row
                for (var i = 0; i < columnTexts.Count; i++)
                {
                    SpawnGridOption(option, inWorldSpace, contentParentTransform, "Option_" + (i+1), "Option");
                }
            }

            LayoutRebuilder.ForceRebuildLayoutImmediate(contentParentTransform.GetComponent<RectTransform>());
            
            ResizeGrid(_questionnaireManager.orientation == QTQuestionnaireManager.Orientation.Vertical ? 928 : 1378);
            
            // update headerName field
            if (!_oldHeaderName.Equals(headerName))
            {
                _oldHeaderName = headerName;
                name = name.Split('_')[0] + "_" + headerName;
            }
            
            // update question field
            transform.GetChild(0).GetComponent<TextMeshProUGUI>().text = question;
            
            // update required bool
            transform.GetChild(0).GetChild(0).gameObject.SetActive(answerRequired);
        }
        private List<string> ConvertToArrayList(JSONArray array)
        {
            var list = new List<string>();
            for (var i = 0; i < array.Count; i++)
            {
                list.Add(array[i].Value);
            }
            return list;
        }
        
#if UNITY_EDITOR
        /// <summary>
        /// (Re-)Builds the entire grid and resizes it.
        /// </summary>
        public void BuildGrid()
        {
            UnityEditor.EditorApplication.delayCall+=()=>
            {
                try
                {
                    var contentParentTransform = transform.GetChild(1);
                    if (!_questionnaireManager) return;
                    var inWorldSpace = _questionnaireManager.spawnObjectsInWorldSpace;
                    // load grid content prefabs
                    var gridText = Resources.Load("QuestionnaireToolkitPrefabs/GridText");
                    var option = Resources.Load("QuestionnaireToolkitPrefabs/CheckboxesGridOption");

                    ClearParent(contentParentTransform);
                
                    // spawn the top left-most grid cell, which needs to be empty
                    var empty = new GameObject("Empty", typeof(RectTransform));
                    if (inWorldSpace)
                    {
                        var rectTransform = empty.GetComponent<RectTransform>();
                        rectTransform.anchoredPosition3D = Vector3.zero;
                        rectTransform.localScale = Vector3.one;
                    }
                    empty.transform.SetParent(contentParentTransform);

                    // add columns
                    foreach (var column in columnTexts)
                    {
                        SpawnGridOption(gridText, inWorldSpace, contentParentTransform, column, "Column");
                    }

                    // add rows
                    foreach (var row in rowTexts)
                    {
                        SpawnGridOption(gridText, inWorldSpace, contentParentTransform, row, "Row");

                        // add options within a row
                        for (var i = 0; i < columnTexts.Count; i++)
                        {
                            SpawnGridOption(option, inWorldSpace, contentParentTransform, "Option_" + (i+1), "Option");
                        }
                    }

                    ResizeGrid();
                }
                catch (Exception)
                {
                    // ignored
                }
            };
        }
#endif

        /// <summary>
        /// Instantiate the given object into the grid.
        /// </summary>
        private void SpawnGridOption(UnityEngine.Object o, bool inWorldSpace, Transform contentParentTransform, string text, string type)
        {
            var g = Instantiate(o, contentParentTransform, inWorldSpace) as GameObject;
            g.name = text;
            
            try
            {
                g.transform.GetChild(0).GetComponent<TextMeshProUGUI>().text = text;
            }
            catch (Exception)
            {
                // ignored
            }

            // add a tag to find the row in the options child list later.
            if (type.Equals("Row"))
            {
                g.tag = "QTGridRowHeader";
            }
            else if(type.Equals("Column"))
            {
                g.tag = "Untagged";
            }
            
            if (inWorldSpace)
            {
                var rectTransform = g.GetComponent<RectTransform>();
                rectTransform.anchoredPosition3D = Vector3.zero;
                rectTransform.localScale = Vector3.one;
            }
        }

        /// <summary>
        /// Sets the width of each grid cell to always expand them to the available horizontal space.
        /// </summary>
        public void ResizeGrid(float availableWidth = 0)
        {
            var optionParent = transform.GetChild(1);
            var currWidth = 0f;
            if (availableWidth == 0)
            {
                currWidth = optionParent.GetComponent<RectTransform>().sizeDelta.x / (columnTexts.Count + 1);
            }
            else
            {
                currWidth = availableWidth / (columnTexts.Count + 1);
            }
            
            optionParent.GetComponent<GridLayoutGroup>().cellSize = new Vector2(currWidth, 50);
        }

        /// <summary>
        /// Clears the old grid before rebuilding it.
        /// </summary>
        private void ClearParent(Transform parent)
        {
            for (var i = parent.childCount - 1; i >= 0; i--)
            {
                DestroyImmediate(parent.GetChild(i).gameObject);
            }
        }
        
        /// <summary>
        /// Updates the row/column texts in case of changes.
        /// </summary>
        private void UpdateGridTexts()
        {
            var parent = transform.GetChild(1);

            for (var c = 1; c < columnTexts.Count+1; c++)
            {
                parent.GetChild(c).GetChild(0).GetComponent<TextMeshProUGUI>().text = columnTexts[c - 1];
            }
            
            for (var r = columnTexts.Count+1; r < (rowTexts.Count+1) * (columnTexts.Count+1); r += columnTexts.Count+1)
            {
                parent.GetChild(r).GetChild(0).GetComponent<TextMeshProUGUI>().text = rowTexts[(r / (columnTexts.Count + 1)) - 1];
            }
        }
//#endif
    }
}
