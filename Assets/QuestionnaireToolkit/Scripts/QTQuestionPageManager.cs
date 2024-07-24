using System;
using System.Collections.Generic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.Serialization;

namespace QuestionnaireToolkit.Scripts
{
    /// <summary>
    /// This class is used to manage a question page and to edit/create question items.
    /// </summary>
    [ExecuteInEditMode]
    public class QTQuestionPageManager : MonoBehaviour
    {
        private bool _showFullscreenTextState = false;
        public bool showFullscreenText = false;
        private string _oldFullscreenText;
        public string fullscreenText = "This is a fullscreen text item that can be used display long texts, e.g., to create a start screen to welcome users and an end screen to thank for participation.";
        
        private bool _topPanelState = true;
        public bool showTopPanel = true;
        private string _oldInstructionText;
        public string instructionText = "This is an instruction text for this question page.";
        
        // list of all question items that are displayed within this page
        [SerializeField]
        public List<GameObject> questionItems = new List<GameObject>();

        // available items
        public enum QuestionItemsEnum
        {
            Slider,
            Dropdown,
            LinearScale,
            MultipleChoice,
            MultipleChoiceGrid,
            Checkboxes,
            CheckboxesGrid,
            TextInputShort,
            TextInputLong,
            Text,
            Image,
            Button,
            Video
        }

        [Header("Item Creation")]
        public QuestionItemsEnum type = QuestionItemsEnum.LinearScale;
        // the name of the item which appears in the results file as header entry
        public string headerName = "Item";
        // the displayed question of this question item
        public string question = "This is a question.";
        // bool to determine if the new item should be filled when created
        [Tooltip("Fills the question item with default options at creation.")]
        public bool automaticFill = true;
        
        [HideInInspector]
        public int selectedIndex = -1;
        [HideInInspector]
        public bool moveItemPressed = false;
        public GameObject selectedItem = null;
        public int desiredPage = 0;
        [HideInInspector]
        public bool wrongPage = false;
        [HideInInspector]
        public string moveItemHint = "";

        public QTManager _qtManager = null;
        public QTQuestionnaireManager _questionnaireManager;

        private void Start()
        {
            _questionnaireManager = transform.parent.GetComponent<QTQuestionnaireManager>();
            try
            {
                _qtManager = _questionnaireManager.transform.parent.GetComponent<QTManager>();
            }
            catch (Exception) { }

            _oldInstructionText = instructionText;
        }

        // Runtime method
        public void PrevPage()
        {
            _questionnaireManager.PrevPage();
        }
        
        // Runtime method
        public void NextPage()
        {
            _questionnaireManager.NextPage();
        }
        
        public void OnValidate()
        {
            //if (EditorApplication.isPlayingOrWillChangePlaymode) return;

            // update fullscreen text visibility
            if (_showFullscreenTextState != showFullscreenText)
            {
                transform.GetChild(0).gameObject.SetActive(!showFullscreenText); // disable scrollview
                transform.GetChild(8).gameObject.SetActive(showFullscreenText);
                var rectTransform = transform.GetChild(8).GetComponent<RectTransform>();
                rectTransform.offsetMax = showTopPanel ? new Vector2(-32, -70) : new Vector2(-32, -20);
                rectTransform.offsetMin = transform.parent.GetComponent<QTQuestionnaireManager>().showBottomPanel ? new Vector2(32, 70) : new Vector2(32, 20);
                if (showFullscreenText)
                {
                    questionItems.Clear(); // clear question items list to prevent them from being recorded
                }
                else
                {
                    var contentTransform = transform.GetChild(0).GetChild(0).GetChild(0);
                    for (var i = 0; i < contentTransform.childCount; i++)
                    {
                        questionItems.Add(contentTransform.GetChild(i).gameObject); // restore question items list
                    }
                }
                _questionnaireManager.BuildHeaderItems();
                _showFullscreenTextState = showFullscreenText;
            }
            
            // update fullscreen text
            if (_oldFullscreenText != fullscreenText)
            {
                transform.GetChild(8).GetChild(0).GetComponent<TextMeshProUGUI>().text = fullscreenText;
                _oldFullscreenText = fullscreenText;
            }
            
            // update top panel visibility
            if (_topPanelState != showTopPanel)
            {
                try
                {
                    var rectTransform = transform.GetChild(0).GetComponent<RectTransform>();
                    var offsetMax = rectTransform.offsetMax;
                    rectTransform.offsetMax = showTopPanel ? new Vector2(offsetMax.x, -56) : new Vector2(offsetMax.x, 0);
                    
                    rectTransform = transform.GetChild(8).GetComponent<RectTransform>();
                    rectTransform.offsetMax = showTopPanel ? new Vector2(-32, -70) : new Vector2(-32, -20);
                   
                    transform.GetChild(6).gameObject.SetActive(showTopPanel); // background
                    transform.GetChild(7).gameObject.SetActive(showTopPanel); // instruction text
                    _topPanelState = showTopPanel;
                }
                catch (Exception) { }
            }
            
            // update instruction text
            if (_oldInstructionText != instructionText)
            {
                transform.GetChild(7).GetComponent<TextMeshProUGUI>().text = instructionText;
                _oldInstructionText = instructionText;
            }
        }

        /// <summary>
        /// Adds the selected item to this page.
        /// </summary>
        public void AddItem(bool import = false, QTQuestionnaireManager manager = null, QuestionItemsEnum i_type = QuestionItemsEnum.LinearScale,  string i_question = null, string i_headerName = null)
        {
            if (import)
            {
                _questionnaireManager = manager;
                type = i_type;
                question = i_question;
                headerName = i_headerName;
                automaticFill = false;
            }
            
            // load and instantiate the selected item prefab
            var contentParentTransform = transform.GetChild(0).GetChild(0).GetChild(0).transform;
            //var o = AssetDatabase.LoadAssetAtPath("Assets/QuestionnaireToolkit/Prefabs/" + type + ".prefab", typeof(GameObject));
            var o = Resources.Load("QuestionnaireToolkitPrefabs/" + type);
            var inWorldSpace = _questionnaireManager.spawnObjectsInWorldSpace;
            var g = Instantiate(o, contentParentTransform, inWorldSpace) as GameObject;
            questionItems.Add(g);
            
            // if in VR mode set position and scale as needed
            if (inWorldSpace)
            {
                var rectTransform = g.GetComponent<RectTransform>();
                rectTransform.anchoredPosition3D = Vector3.zero;
                rectTransform.localScale = Vector3.one;
            }
            
            if (type == QuestionItemsEnum.Text || type == QuestionItemsEnum.Image || type == QuestionItemsEnum.Button || type == QuestionItemsEnum.Video)
            {
                g.name = type + "";
                if (import) _questionnaireManager.currentImportedItem = g;
                return;
            }
            if (headerName.Equals("Item"))
            {
                headerName += " " + questionItems.Count;
            }
            g.name = type + "_" + headerName + "_" + question;
            g.transform.GetChild(0).gameObject.GetComponent<TextMeshProUGUI>().text = question;

            headerName = "Item";
            // header items list needs to be rebuild in the respective question item script

            if (import)
            {
                _questionnaireManager.currentImportedItem = g;
            }
        }

#if UNITY_EDITOR
        /// <summary>
        /// Copy the selected page.
        /// </summary>
        public void CopySelection()
        {
            if (selectedIndex >= 0 && selectedIndex < questionItems.Count)
            {
                var contentParentTransform = transform.GetChild(0).GetChild(0).GetChild(0).transform;
                var inWorldSpace = _questionnaireManager.spawnObjectsInWorldSpace;
                var g = Instantiate(questionItems[selectedIndex], contentParentTransform, inWorldSpace);
                questionItems.Add(g);
            }
        }

        /// <summary>
        /// Deletes the selected page.
        /// </summary>
        public void DeleteItem(int listCount, int sel)
        {
            if (listCount < questionItems.Count) 
            {
                // item was removed
                DestroyImmediate(questionItems[sel]);
                questionItems.RemoveAt(sel);
                _questionnaireManager.BuildHeaderItems();
            }
        }

        /// <summary>
        /// Reorder the page list according to the changes in the reorderable editor list.
        /// </summary>
        public void ReorderItems(int listCount, int sel)
        {
            if (listCount == questionItems.Count)
            {
                questionItems[sel].transform.SetSiblingIndex(sel);
                _questionnaireManager.BuildHeaderItems();
            }
        }

        /// <summary>
        /// Forward the call to the QuestionnaireMananger.
        /// </summary>
        public void ShowThisPage()
        {
            _questionnaireManager.ShowPage(transform.GetSiblingIndex());
        }

        /// <summary>
        /// Moves an item from this page to a desired page if the desired page exists.
        /// </summary>
        public void MoveItem()
        {
            if (desiredPage < 0 || desiredPage >= _questionnaireManager.transform.childCount)
            {
                wrongPage = true;
                moveItemHint = "Target page does not exist!";
                return;
            }
            if (desiredPage == transform.GetSiblingIndex())
            {
                wrongPage = true;
                moveItemHint = "Cannot move item to the same page!";
                return;
            }
            if (selectedItem == null)
            {
                wrongPage = true;
                moveItemHint = "Nothing selected!";
                return;
            }
            
            if (selectedIndex >= 0 && selectedIndex < questionItems.Count)
            {
                var targetPage = _questionnaireManager.transform.GetChild(desiredPage);
                var contentParentTransform = targetPage.GetChild(0).GetChild(0).GetChild(0).transform;
                var g = Instantiate(questionItems[selectedIndex], contentParentTransform, _questionnaireManager.spawnObjectsInWorldSpace);
                g.name = g.name.Remove(g.name.Length - 7); // remove (Clone) from name
                targetPage.GetComponent<QTQuestionPageManager>().questionItems.Add(g);
                moveItemPressed = false;
                wrongPage = false;
                selectedItem = null;
                DeleteItem(questionItems.Count-1, selectedIndex);
            }
        }
#endif
    }
}
