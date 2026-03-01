using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
 using System.Text;
 using BOforUnity;
 using QuestionnaireToolkit.Scripts.MemberReflection.Reflection;
using QuestionnaireToolkit.Scripts.SimpleJSON;
using QuestionnaireToolkit.Scripts.StandaloneFileBrowser;
using TMPro;
#if UNITY_EDITOR
using UnityEditor;
#endif
 using UnityEngine;
using UnityEngine.Events;
 using UnityEngine.EventSystems;
 using UnityEngine.Serialization;
using UnityEngine.UI;
using UnityEngine.Video;

#if HTC_Vive
 using HTC.UnityPlugin.Pointer3D;
 using HTC.UnityPlugin.Vive;
#endif
#if Oculus

#endif

namespace QuestionnaireToolkit.Scripts
{
    /// <summary>
    /// This class is used to manage all available questionnaire settings and to write the results in a file.
    /// </summary>
    [ExecuteInEditMode, DisallowMultipleComponent]
    public class QTQuestionnaireManager : MonoBehaviour
    {
        public enum DisplayMode { Desktop, VR, Mobile }
        public enum FileFormat { csv }
        public enum Orientation { Horizontal, Vertical }
        public enum DeviceType { Other, Vive, Oculus }
        public enum ColorScheme { Default, Red, Green, Blue, Custom }
        [HideInInspector]
        public bool spawnObjectsInWorldSpace = false;

        // reference to the scriptableObject which stores the current response_id and run.
        public QuestionnaireMeta qtMetaData;

        public bool overrideManagerSettings = false;
        
        [Header("Display Settings")]
        [Tooltip("If true, then the questionnaire starts when scene is loaded. Otherwise the questionnaire needs to be started manually.")]
        public bool startWithScene = false;
        [Tooltip("Set the display mode of the questionnaire according to the target platform.")]
        public DisplayMode displayMode = DisplayMode.Desktop;
        [Tooltip("Orientation of a question page. Horizontal aspect ratio: 16:9, Vertical aspect ratio: 10:16.")]
        public Orientation orientation = Orientation.Horizontal;
        public DeviceType deviceType = DeviceType.Other;
        public float pageHeight = 900;
        public bool dynamicHeight = false;
        [Tooltip("(VR only) If true, then you can set the position/rotation of the questionnaire in world space. Otherwise the distance below will be used and the questionnaire will be positioned in the center of the view.")]
        public bool useCustomTransform = false;
        [Tooltip("(VR only) Distance to main camera in meters.")]
        public float distanceToCamera = 1;
        [Tooltip("(VR only) Determines the size of a question page in meters.")]
        public float pageScaleFactor = 1;
        public ColorScheme colorScheme = ColorScheme.Default;
        public float sliderValue = 20;
        public bool showTopPanel = true;
        [Tooltip("If the bottom panel is deactivated the questionnaire cannot be finished! Use only for basic UI elements!")]
        public bool showBottomPanel = true;
        public bool showPageNumber = true;
        [Tooltip("Enables the ability to go back to the previous page.")]
        public bool showPrevButton = false;

        [Header("Results File Settings")] 
        public bool generateResultsFile = true;
        public string resultsSavePath = "Assets/QuestionnaireToolkit/Results/";
        public string resultsFileName = "MyQuestionnaire";
        public FileFormat resultsFileFormat = FileFormat.csv;
        [Tooltip("If true, then a new results file will be created whenever a questionnaire is started. The name above will be used with an additional counter for the file name.")]
        public bool newFileEachStart = true;
        [Tooltip("Custom method to call when finished.")]
        public UnityEvent onQuestionnaireFinished;

        public bool questionnaireInitialized = false;

        public string importPath = "..select..";
        private GameObject currentImportedPage;
        public GameObject currentImportedItem;
        public string exportPath = "..select..";
        
        [Header("Results Visualization")]
        public bool customUserId = false;
        public string userId = "";
        [Tooltip("Indicates how many times a user should answer this questionnaire.")]
        public int runsPerUser = 1;
        [Tooltip("Include the timestamp of when the questionnaire was started.")]
        public bool generateStartTimestamp = false;
        [Tooltip("Include the timestamp of when the questionnaire was finished.")]
        public bool generateFinishTimestamp = false;
        public bool overwriteResultsHeaderItems = false;
        public List<string> resultsHeaderItems = new List<string>();

        [Header("Page Management")]
        [SerializeField]
        public List<GameObject> questionPages = new List<GameObject>();
        
        [Reorderable]
        public ReorderableChildList additionalCsvItems;
        [System.Serializable]
        public class AdditionalCsvItem
        {
            public string headerName = "Additional Item";
            [Filter(Fields = true, Properties = true)]
            public UnityMember itemValue;
        }
        [System.Serializable]
        public class ReorderableChildList : ReorderableArray<AdditionalCsvItem> { }
        
        private const float pageScale = 0.001f;
        private const float AspectRatio = 0.5625f;
        private float _pageWidth = 1600;
        private bool _bottomPanelState = true;
        private bool _pageNumberState = true;
        private bool _prevButtonState;
        private bool _topPanelState = true;
        
        private int _oldDisplayMode;
        private int _oldOrientation;
        private int _oldColorScheme;
        private int _oldDeviceType;
        private bool _oldDynamicHeight ;
        private float _oldPageHeight = 900;
        private string _oldResultFileName = "";
        private float _oldSliderValue = 20;

        private Color _oldBackgroundColor;
        private Color _oldBottomColor;
        private Color _oldHighlightColor;
        public Color pageBackgroundColor = new Color(0.9245f, 0.9245f, 0.9245f, 1);
        public Color pageBottomColor = new Color(0.76f, 0.88f, 1, 1);
        public Color highlightColor = new Color(0, 0.67f, 1, 1);

        [HideInInspector]
        public bool running;
        
        private string startedTimestamp;
        private string finishedTimestamp;
        private string userPath;
        private GameObject _visiblePage;
        private int _currentPage;
        private GameObject background;
        [HideInInspector]
        public int selectedPage = -1;

        private string metaDataStr;
        private int currentResponseId;
        private int currentRun;
        [HideInInspector]
        public QTManager _qtManager;
        private BoForUnityManager _cachedBoForUnityManager;

        /// <summary>
        /// Start this questionnaire.
        /// </summary>
        /// <param name="position">(optional) specifies the questionnaire position.</param>
        /// <param name="rotation">(optional) specifies the questionnaire rotation.</param>
        /// <param name="restart">(optional) default is true. If true, the questionnaire will be reset and restarted.</param>
        /// <returns>True if the questionnaire was initialized. False if the questionnaire is already running.</returns>
        public bool StartQuestionnaire(bool restart = true, Vector3 position = default, Quaternion rotation = default)
        {
            if (running && !restart)
            {
                ShowQuestionnaire();
                return false;
            }
            
            ResetQuestionnaire();
            return InitQuestionnaire(position, rotation);
        }

        /// <summary>
        /// Initializes the questionnaire with the settings specified in the inspector.
        /// </summary>
        public bool InitQuestionnaire(Vector3 position = default, Quaternion rotation = default)
        {
            try
            {
#if UNITY_EDITOR
                if (!EditorApplication.isPlayingOrWillChangePlaymode) return false;
#endif
                if (questionPages.Count == 0)
                {
                    Debug.Log("No question pages!");
                    return false;
                }

#if !UNITY_EDITOR
                metaDataStr = LoadMetaData();
                currentResponseId = int.Parse(metaDataStr.Split(';')[0]);
                currentRun = int.Parse(metaDataStr.Split(';')[1]);
#endif
#if UNITY_EDITOR
                currentResponseId = qtMetaData.currentResponseId;
                currentRun = qtMetaData.currentUserRun;
#endif
                
                if(!overwriteResultsHeaderItems) 
                    BuildHeaderItems(); // build the final header item list
                
                if (displayMode == DisplayMode.VR)
                {
                    SetupVRSettings();
                    
                    if (!position.Equals(Vector3.zero))
                    {
                        transform.position = position;
                    }
                    else if (!useCustomTransform)
                    {
                        transform.position = Camera.main.transform.position + Camera.main.transform.forward * distanceToCamera;
                    }
                    
                    if (!IsApproximate(rotation, Quaternion.identity, 0.0000004f))
                    {
                        transform.rotation = rotation;
                    }
                    else if (!useCustomTransform)
                    {
                        transform.LookAt(Camera.main.transform);
                        transform.rotation *= Quaternion.Euler(0, 180, 0);
                        transform.rotation = Quaternion.Euler(transform.eulerAngles.x, transform.eulerAngles.y, 0);
                    }
                    
                    background = Instantiate(Resources.Load("QTBackgroundVR", typeof(GameObject)), transform) as GameObject;
                    background.transform.localPosition = new Vector3(0, 0, 11);
                    var pageSizeDelta = questionPages[0].GetComponent<RectTransform>().sizeDelta;
                    background.transform.localScale = new Vector3(pageSizeDelta.x, pageSizeDelta.y, 20);
                }

                if (generateResultsFile)
                {
                    var systemPath = Application.persistentDataPath + resultsSavePath;
                    userPath = systemPath.TrimEnd('/') + "/" + resultsFileName;

                    if (newFileEachStart)
                    {
                        userPath += "_" + (currentResponseId + 1);
                    }
                    userPath += "." + resultsFileFormat;

                    if (!Directory.Exists(systemPath))
                    {
                        Directory.CreateDirectory(systemPath);
                    }
                    
                    if (!(File.Exists(userPath)))
                    {
                        using (var writer = new StreamWriter(userPath, true))
                        {
                            var headerCells = new List<string> { "response_id" };
                            if (customUserId)
                            {
                                headerCells.Add("user_id");
                            }
                            if (runsPerUser > 1)
                            {
                                headerCells.Add("run");
                            }
                            if (generateStartTimestamp)
                            {
                                headerCells.Add("started");
                            }
                            if (generateFinishTimestamp)
                            {
                                headerCells.Add("finished");
                            }

                            foreach (var t in resultsHeaderItems)
                            {
                                headerCells.Add(string.IsNullOrWhiteSpace(t) ? "NULL" : t);
                            }

                            if (additionalCsvItems != null)
                            {
                                foreach (var aci in additionalCsvItems)
                                {
                                    if (aci == null)
                                    {
                                        Debug.LogWarning("Skipping null entry in additionalCsvItems while creating questionnaire header.");
                                        continue;
                                    }

                                    var header = string.IsNullOrWhiteSpace(aci.headerName) ? "NULL" : aci.headerName;
                                    headerCells.Add(header);
                                }
                            }

                            writer.WriteLine(string.Join(",", headerCells.Select(EscapeCsvCell)));
                        }
                    }
                    Debug.Log("Questionnaire: " + resultsFileName + (newFileEachStart ? "_" + (currentResponseId + 1) : "") + 
                              (runsPerUser > 1 ? " run: " + (currentRun + 1) : "") + " started!");
                    startedTimestamp = DateTime.UtcNow + "";
                }
                else
                {
                    Debug.Log("Questionnaire started. (Results will not be recorded!)");
                }
                
                SetEventCamera();                
                _currentPage = 0;
                _visiblePage = questionPages[_currentPage];
                ShowQuestionnaire();
                running = true;
                return true;
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                return false;
            }
        }

        private void Start()
        {
            try
            {
                _qtManager = transform.parent.GetComponent<QTManager>();
            }
            catch (Exception) { _qtManager = null; }
#if UNITY_EDITOR
            if (!_qtManager) // check for tags if there is no QTManager in this scene
            {
                TagsAndLayers.RefreshQtTags();
            }
            
            if (!EditorApplication.isPlayingOrWillChangePlaymode)
            {
                if (_qtManager == null && GameObject.FindGameObjectsWithTag("QTQuestionnaireManager").Length > 1)
                {
                    Debug.Log("Only one individual Questionnaire allowed in a scene!\n" +
                              "In order to have more, please use the QTManager.");
                    DestroyImmediate(gameObject);
                    return;
                }
                if (_qtManager == null && GameObject.FindGameObjectsWithTag("QTManager").Length == 1)
                {
                    Debug.Log("QTManager cannot exist besides individual Questionnaires!");
                    DestroyImmediate(gameObject);
                    return;
                }
                
                // create a new empty page
                if (questionPages.Count == 0)
                    CreatePage();
                
                try
                {
                    // unpack the prefab completely
                    PrefabUtility.UnpackPrefabInstance(gameObject, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
                }
                catch (Exception)
                {
                    // ignored
                }
                // add a new EventSystem if needed
                if (FindObjectOfType<EventSystem>() == null)
                {
                    var o = new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
                }
                return;
            }
            
#pragma warning disable 618
            EditorApplication.playmodeStateChanged += () =>
#pragma warning restore 618
            {
                // show the last selected page before playmode was entered.
                if (!EditorApplication.isPlayingOrWillChangePlaymode && EditorApplication.isPlaying) 
                    ShowPage(_currentPage);
            };
#endif
            if (!running)
            {
                HideQuestionnaire();
            }
            
            if (startWithScene)
            {
                StartQuestionnaire();
            }
        }
#if UNITY_EDITOR
        
        public void OnValidate()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;

            EditorApplication.delayCall += () =>
            {
                try
                {
                    if (!overrideManagerSettings)
                    {
                        displayMode = (DisplayMode)(int)_qtManager.displayMode;
                        deviceType = (DeviceType)(int)_qtManager.deviceType;
                        orientation = (Orientation)(int)_qtManager.orientation;
                        colorScheme = (ColorScheme)((int)_qtManager.colorScheme);
                    
                        pageHeight = _qtManager.pageHeight;
                        pageScaleFactor = _qtManager.pageScaleFactor;
                        dynamicHeight = _qtManager.dynamicHeight;
                        useCustomTransform = _qtManager.useCustomTransform;
                        distanceToCamera = _qtManager.distanceToCamera;

                        pageBackgroundColor = _qtManager.pageBackgroundColor;
                        pageBottomColor = _qtManager.pageBottomColor;
                        highlightColor = _qtManager.highlightColor;
                        sliderValue = _qtManager.sliderValue;
                        
                        showTopPanel = _qtManager.showTopPanel;
                        showBottomPanel = _qtManager.showBottomPanel;
                        showPageNumber = _qtManager.showPageNumber;
                        showPrevButton = _qtManager.showPrevButton;
                    }
                }
                catch (Exception ) { }
                
                // update questionnaire name (only if there is a QTManager)
                if (_oldResultFileName != resultsFileName && _qtManager != null)
                {
                    resultsFileName = TryResultFileName(resultsFileName);
                    _oldResultFileName = resultsFileName;
                }
                
                // update device type config
                if (_oldDeviceType != (int) deviceType)
                {
                    switch (deviceType)
                    {
                        case DeviceType.Other:
                            AddDefineIfNeeded("Other", new []{"HTC_Vive", "Oculus"});
                            break;
                        case DeviceType.Vive:
                            AddDefineIfNeeded("HTC_Vive", new []{"Oculus"});
                            break;
                        case DeviceType.Oculus:
                            AddDefineIfNeeded("Oculus", new []{"HTC_Vive"});
                            break;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                    _oldDeviceType = (int) deviceType;
                }
                
                // update custom page height
                if (_oldPageHeight != pageHeight && _oldOrientation == (int) orientation)
                {
                    foreach (var page in questionPages)
                    {
                        page.GetComponent<RectTransform>().sizeDelta = new Vector2(page.GetComponent<RectTransform>().sizeDelta.x, pageHeight);
                        if (displayMode == DisplayMode.Desktop || displayMode == DisplayMode.Mobile)
                            page.GetComponent<CanvasScaler>().referenceResolution = new Vector2(1600, pageHeight + 200);
                    }
                    _oldPageHeight = pageHeight;
                }
                
                // update dynamic page height
                if (_oldDynamicHeight != dynamicHeight)
                {
                    SetDynamicHeight(displayMode == DisplayMode.Desktop || displayMode == DisplayMode.Mobile);
                    _oldDynamicHeight = dynamicHeight;
                }
                
                // update display mode settings
                if (_oldDisplayMode != (int)displayMode)
                {
                    switch (displayMode)
                    {
                        case DisplayMode.Desktop:
                            spawnObjectsInWorldSpace = false;
                            break;
                        case DisplayMode.VR:
                            spawnObjectsInWorldSpace = true;
                            break;
                        case DisplayMode.Mobile:
                            spawnObjectsInWorldSpace = false;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                    ChangeRendering();
                    _oldDisplayMode = (int) displayMode;
                }

                // update page scale in VR mode
                if (displayMode == DisplayMode.VR && !useCustomTransform)
                {
                    transform.position = new Vector3(0, 0, distanceToCamera);
                    if (pageScaleFactor > 0)
                        transform.localScale = new Vector3(pageScale * pageScaleFactor, pageScale * pageScaleFactor, pageScale * pageScaleFactor);
                }

                // update page orientation
                if (_oldOrientation != (int)orientation)
                {
                    switch (orientation)
                    {
                        case Orientation.Horizontal:
                            _pageWidth = 1600;
                            pageHeight = 900;
                            break;
                        case Orientation.Vertical:
                            _pageWidth = 1000;
                            pageHeight = 1600;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                    SetPageOrientation();
                    ResizeItemPanels();
                    _oldOrientation = (int) orientation;
                }

                // update overall color scheme
                if (_oldColorScheme != (int)colorScheme)
                {
                    ChangeColorScheme((int) colorScheme);
                    _oldColorScheme = (int) colorScheme;
                }

                // update custom colors
                if (_oldBackgroundColor != pageBackgroundColor || _oldBottomColor != pageBottomColor || _oldHighlightColor != highlightColor)
                {
                    ChangeColorScheme((int) colorScheme);
                    _oldBackgroundColor = pageBackgroundColor;
                    _oldBottomColor = pageBottomColor;
                    _oldHighlightColor = highlightColor;
                }
                
                // update page transparency
                if ((int)_oldSliderValue != (int)sliderValue)
                {
                    pageBackgroundColor.a = 1 - sliderValue * 0.01f;
                    for (var i = 0; i < questionPages.Count; i++)
                    {
                        questionPages[i].GetComponent<Image>().color = pageBackgroundColor;
                    }
                    _oldSliderValue = sliderValue;
                }

                // update top panel visibility
                if (_topPanelState != showTopPanel)
                {
                    try
                    {
                        for (var i = 0; i < questionPages.Count; i++)
                        {
                            var rectTransform = questionPages[i].transform.GetChild(0).GetComponent<RectTransform>();
                            var offsetMax = rectTransform.offsetMax;
                            offsetMax = showTopPanel ? new Vector2(offsetMax.x, -56) : new Vector2(offsetMax.x, 0);
                            rectTransform.offsetMax = offsetMax;
                            questionPages[i].transform.GetChild(6).gameObject.SetActive(showTopPanel); // background
                            questionPages[i].transform.GetChild(7).gameObject.SetActive(showTopPanel); // instruction text
                        }
                        _topPanelState = showTopPanel;
                    }
                    catch (Exception) { }
                }
                
                // update bottom panel visuals
                if (_bottomPanelState != showBottomPanel)
                {
                    try
                    {
                        for (var i = 0; i < questionPages.Count; i++)
                        {
                            var rectTransform = questionPages[i].transform.GetChild(0).GetComponent<RectTransform>();
                            var offsetMin = rectTransform.offsetMin;
                            offsetMin = showBottomPanel ? new Vector2(offsetMin.x, 53) : new Vector2(offsetMin.x, 0);
                            rectTransform.offsetMin = offsetMin;
                            questionPages[i].transform.GetChild(1).gameObject.SetActive(showBottomPanel); // background
                            questionPages[i].transform.GetChild(2).gameObject.SetActive(showBottomPanel); // next button
                            questionPages[i].transform.GetChild(3).gameObject.SetActive(showBottomPanel ? showPageNumber : showBottomPanel); // page number
                            questionPages[i].transform.GetChild(5).gameObject.SetActive(showBottomPanel ? showPrevButton : showBottomPanel); // prev button
                        }
                        _bottomPanelState = showBottomPanel;
                    }
                    catch (Exception) { }
                }

                // update visibility of page number
                if (_pageNumberState != showPageNumber)
                {
                    for (var i = 0; i < questionPages.Count; i++)
                    {
                        questionPages[i].transform.GetChild(3).gameObject.SetActive(showPageNumber);
                    }
                    _pageNumberState = showPageNumber;
                }
                
                // update visibility of the previous button
                if (_prevButtonState != showPrevButton)
                {
                    try
                    {
                        for (var i = 0; i < questionPages.Count; i++)
                        {
                            questionPages[i].transform.GetChild(5).gameObject.SetActive(showPrevButton && i > 0);
                        }
                        _prevButtonState = showPrevButton;
                    }
                    catch (Exception)
                    {
                        // ignored
                    }
                }
            };
        }
#endif
        
        public void ImportPages()
        {
            var extensions = new [] {
                new ExtensionFilter("JSON Files ", "json" ),
                new ExtensionFilter("All Files ", "*" ),
            };
            var paths = StandaloneFileBrowser.StandaloneFileBrowser.OpenFilePanel("Open Questionnaire File", "", extensions, false);
            if (paths.Length > 0)
            {
                importPath = paths[0];
                ReadJson(importPath);
                BuildHeaderItems();
            }
            else
            {
                importPath = "..select..";
            }
        }

        public void ImportPagesFile(string filepath)
        {
            ReadJson(filepath);
        }

        public void ExportPages()
        {
            var extensionList = new [] {
                new ExtensionFilter("JSON ", "json"),
                new ExtensionFilter("Text ", "txt"),
            };
            var path = StandaloneFileBrowser.StandaloneFileBrowser.SaveFilePanel("Save File", "", "MyQuestionnaire", extensionList);
            if (path != "")
            {
                exportPath = path;
                WriteJson(exportPath);
            }
            else
            {
                exportPath = "..select..";
            }
        }

        private void ReadJson(string jsonPath)
        {
            // reads and parses .json input file
            var JSONString = File.ReadAllText(jsonPath);
            var N = JSON.Parse(JSONString);
            pageBackgroundColor = new Color(
                float.Parse(N["page_background_color"][0], CultureInfo.InvariantCulture),
                float.Parse(N["page_background_color"][1], CultureInfo.InvariantCulture),
                float.Parse(N["page_background_color"][2], CultureInfo.InvariantCulture),
                float.Parse(N["page_background_color"][3], CultureInfo.InvariantCulture));
            pageBottomColor = new Color(
                float.Parse(N["page_panel_color"][0], CultureInfo.InvariantCulture),
                float.Parse(N["page_panel_color"][1], CultureInfo.InvariantCulture),
                float.Parse(N["page_panel_color"][2], CultureInfo.InvariantCulture),
                float.Parse(N["page_panel_color"][3], CultureInfo.InvariantCulture));
            highlightColor = new Color(
                float.Parse(N["highlight_color"][0], CultureInfo.InvariantCulture),
                float.Parse(N["highlight_color"][1], CultureInfo.InvariantCulture),
                float.Parse(N["highlight_color"][2], CultureInfo.InvariantCulture),
                float.Parse(N["highlight_color"][3], CultureInfo.InvariantCulture));
            sliderValue = N["background_transparency"].AsFloat;
            showBottomPanel = N["show_bottom_panel"].AsBool;
            showPageNumber = N["show_page_number"].AsBool;
            showPrevButton = N["show_prev_button"].AsBool;
            overrideManagerSettings = true;
            colorScheme = ColorScheme.Custom;
            
            // clear all existing pages before importing the new ones
            foreach (var page in questionPages)
            {
                DestroyImmediate(page.gameObject);
            }
            questionPages.Clear();

            var i = 0;
            // continuously reads data from the .json file 
            while (true)
            {
                var pId = N["qPages"][i]["pId"].Value; //read new page
                if (pId != "")
                {
                    var showFullscreenText = N["qPages"][i]["show_fullscreen_text"].AsBool;
                    var fullscreenText = N["qPages"][i]["fullscreen_text"].Value.Replace("\\\"", "\"");
                    var showInstructionPanel = N["qPages"][i]["show_instruction_panel"].AsBool;
                    var instructionText = N["qPages"][i]["instruction_text"].Value.Replace("\\\"", "\"");
                    
                    CreatePage(true, showFullscreenText, fullscreenText, showInstructionPanel, instructionText);

                    var qItems = N["qPages"][i]["qItems"].AsArray;
                    if (qItems == "")
                        qItems[0] = N["qPages"][i]["qItems"].Value;

                    for (var j = 0; j < qItems.Count; j++)
                    {
                        var qType = qItems[j]["qType"].Value;
                        var question = qItems[j]["question"].Value;
                        var headerName = qItems[j]["header_name"].Value;
                        if (headerName.Contains('_'))
                        {
                            Debug.LogError("'_' is not supported in header name!");
                            break;
                        }
                        var mandatory = qItems[j]["mandatory"].AsBool;
                        var options = qItems[j]["options"].AsArray;
                        switch (qType)
                        {
                            case "slider":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this, QTQuestionPageManager.QuestionItemsEnum.Slider, question, headerName);
                                var minValue = qItems[j]["min_value"].AsInt;
                                var maxValue = qItems[j]["max_value"].AsInt;
                                var wholeNumbers = qItems[j]["whole_numbers"].AsBool;
                                var showPanels = qItems[j]["show_panels"].AsBool;
                                var showIntermediatePanels = qItems[j]["show_intermediate_panels"].AsBool;
                                var autoLabels = qItems[j]["auto_labels"].AsBool;
                                var labelZero = qItems[j]["label_zero"].Value;
                                var labelQuarter = qItems[j]["label_quarter"].Value;
                                var labelHalf = qItems[j]["label_half"].Value;
                                var labelThreeQuarters = qItems[j]["label_three_quarters"].Value;
                                var labelFull = qItems[j]["label_full"].Value;
                                currentImportedItem.GetComponent<QTSlider>().ImportSliderValues(mandatory, minValue, maxValue, wholeNumbers, showPanels, showIntermediatePanels, autoLabels, labelZero, labelQuarter, labelHalf, labelThreeQuarters, labelFull);
                                break;
                            case "dropdown":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.Dropdown, question, headerName);
                                for (var op = 0; op < options.Count; op++)
                                {
                                    currentImportedItem.GetComponent<QTDropdown>().AddOption(true, mandatory, options[op]["option_text"].Value, null); // a sprite cannot be saved in json txt format!
                                }
                                break;
                            case "linear_scale":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.LinearScale, question, headerName);
                                for (var op = 0; op < options.Count; op++)
                                {
                                    currentImportedItem.GetComponent<QTLinearScale>().AddOption(true, this, mandatory, options[op]["answer_value"].Value, options[op]["answer_option"].Value);
                                }
                                break;
                            case "multiple_choice":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.MultipleChoice, question, headerName);
                                var otherOptionMc = qItems[j]["include_other_option"].AsBool;
                                for (var op = 0; op < options.Count; op++)
                                {
                                    currentImportedItem.GetComponent<QTMultipleChoice>().AddOption(true, this, mandatory, options[op]["answer_value"].Value, options[op]["answer_option"].Value, otherOptionMc);
                                }
                                break;
                            case "multiple_choice_grid":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.MultipleChoiceGrid, question, headerName);
                                currentImportedItem.GetComponent<QTMultipleChoiceGrid>().ImportGrid(this, mandatory, qItems[j]["row_texts"].AsArray, qItems[j]["column_texts"].AsArray);
                                break;
                            case "checkboxes":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.Checkboxes, question, headerName);
                                var otherOption = qItems[j]["include_other_option"].AsBool;
                                for (var op = 0; op < options.Count; op++)
                                {
                                    currentImportedItem.GetComponent<QTCheckboxes>().AddOption(true, this, mandatory, options[op]["answer_value"].Value, options[op]["answer_option"].Value, otherOption);
                                }
                                break;
                            case "checkboxes_grid":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.CheckboxesGrid, question, headerName);
                                currentImportedItem.GetComponent<QTCheckboxesGrid>().ImportGrid(this, mandatory, qItems[j]["row_texts"].AsArray, qItems[j]["column_texts"].AsArray);
                                break;
                            case "text_input_short":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.TextInputShort, question, headerName);
                                currentImportedItem.GetComponent<QTTextInput>().ImportTextInputValues(mandatory, qItems[j]["placeholder_text"]);
                                break;
                            case "text_input_long":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.TextInputLong, question, headerName);
                                currentImportedItem.GetComponent<QTTextInput>().ImportTextInputValues(mandatory, qItems[j]["placeholder_text"]);
                                break;
                            case "text":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.Text, question, headerName);
                                var elem = currentImportedItem.transform.GetChild(0).GetComponent<TextMeshProUGUI>();
                                elem.text = qItems[j]["text_content"].Value.Replace('\"','"');
                                elem.fontStyle = (FontStyles) Enum.Parse( typeof(FontStyles), qItems[j]["font_style"].Value );
                                elem.fontSize = qItems[j]["font_size"].AsFloat;
                                elem.enableAutoSizing = qItems[j]["auto_sizing"].AsBool;
                                elem.fontSizeMin = qItems[j]["min_size"].AsFloat;
                                elem.fontSizeMax = qItems[j]["max_size"].AsFloat;
                                elem.wordSpacing = qItems[j]["word_spacing"].AsFloat;
                                elem.lineSpacing = qItems[j]["line_spacing"].AsFloat;
                                var cc = qItems[j]["color"].Value.Split(',');
                                elem.color = new Color(
                                    float.Parse(cc[0].Substring(5), CultureInfo.InvariantCulture),
                                    float.Parse(cc[1], CultureInfo.InvariantCulture),
                                    float.Parse(cc[2], CultureInfo.InvariantCulture),
                                    float.Parse(cc[3].Trim(')'), CultureInfo.InvariantCulture));
                                elem.alignment = (TextAlignmentOptions) Enum.Parse( typeof(TextAlignmentOptions), qItems[j]["alignment"].Value );
                                elem.enableWordWrapping = qItems[j]["word_wrapping"].AsBool;
                                elem.overflowMode = (TextOverflowModes) Enum.Parse( typeof(TextOverflowModes), qItems[j]["overflow_mode"].Value );
                                elem.horizontalMapping = (TextureMappingOptions) Enum.Parse( typeof(TextureMappingOptions), qItems[j]["horizontal_mapping"].Value );
                                elem.verticalMapping = (TextureMappingOptions) Enum.Parse( typeof(TextureMappingOptions), qItems[j]["vertical_mapping"].Value );
                                break;
                            case "image":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.Image, question, headerName);
                                try
                                {
                                    currentImportedItem.transform.GetChild(0).GetComponent<Image>().sprite = Resources.Load<Sprite>("QuestionnaireToolkitCustomResources/" + qItems[j]["image_name"].Value);
                                }
                                catch (Exception) { Console.WriteLine("Could not import image content from json! for image name: " + qItems[j]["image_name"].Value); }
                                break;
                            case "button":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.Button, question, headerName);
                                for (var op = 0; op < qItems[j]["button_count"].AsInt - 1; op++)
                                {
                                    currentImportedItem.GetComponent<QTButton>().AddOption(true, this);
                                }
                                break;
                            case "video":
                                currentImportedPage.GetComponent<QTQuestionPageManager>().AddItem(true, this,QTQuestionPageManager.QuestionItemsEnum.Video, question, headerName);
                                try
                                {
                                    currentImportedItem.transform.GetChild(0).GetComponent<VideoPlayer>().clip = Resources.Load<VideoClip>("QuestionnaireToolkitCustomResources/" + qItems[j]["video_name"].Value);
                                }
                                catch (Exception) { Console.WriteLine("Could not import video content from json! for video name: " + qItems[j]["video_name"].Value); }
                                break;
                        }
                    }
                    i++;
                }
                else
                {
                    try { ShowPage(0); } catch (Exception) { } // always show the first page of a newly imported questionnaire
                    break;
                }
            }
        }

        private void WriteJson(string savePath)
        {
            var jsonString = "{\"questionnaire_id\":\"" + transform.GetSiblingIndex() + "\","  
                             + "\"page_background_color\":[\"" + pageBackgroundColor.r +"\",\""+ pageBackgroundColor.g +"\",\""+ pageBackgroundColor.b +"\",\""+ pageBackgroundColor.a + "\"]," 
                             + "\"page_panel_color\":[\"" + pageBottomColor.r +"\",\""+ pageBottomColor.g +"\",\""+ pageBottomColor.b +"\",\""+ pageBottomColor.a + "\"],"
                             + "\"highlight_color\":[\"" + highlightColor.r +"\",\""+ highlightColor.g +"\",\""+ highlightColor.b +"\",\""+ highlightColor.a + "\"],"
                             + "\"background_transparency\":\"" + sliderValue + "\","
                             + "\"show_bottom_panel\":\"" + showBottomPanel + "\","
                             + "\"show_page_number\":\"" + showPageNumber + "\","
                             + "\"show_prev_button\":\"" + showPrevButton + "\","
                             + "\"qPages\":[";
            var pageCount = 1;
            foreach (var page in questionPages) 
            {
                page.SetActive(true); // re-enable all pages otherwise the items cannot be found
                var pageScript = page.GetComponent<QTQuestionPageManager>();
                
                jsonString += "{\"pId\":\"page" + pageCount + "\"," 
                              + "\"show_fullscreen_text\":\"" + pageScript.showFullscreenText + "\","
                              + "\"fullscreen_text\":\"" + pageScript.fullscreenText.Replace("\"", "\\\"").Replace("\n", "\\n") + "\","
                              + "\"show_instruction_panel\":\"" + pageScript.showTopPanel + "\","
                              + "\"instruction_text\":\"" + pageScript.instructionText.Replace("\"", "\\\"").Replace("\n", "\\n") + "\","
                              + "\"qItems\":[";
                pageCount++;

                var questionCount = 1;
                foreach (var question in page.GetComponent<QTQuestionPageManager>().questionItems)
                {
                    jsonString += "{\"qId\":\"q" + questionCount + "\",";
                    questionCount++;
                    
                    switch (question.tag)
                    {
                        case "QTLinearScale":
                            jsonString += "\"qType\":\"linear_scale\","
                                          + "\"question\":\"" + question.transform.GetChild(0).GetComponent<TextMeshProUGUI>().text + "\","
                                          + "\"header_name\":\"" + question.name.Split('_')[1] + "\","
                                          + "\"mandatory\":\"" + question.GetComponent<QTLinearScale>().answerRequired + "\","
                                          + "\"options\":[";
                            foreach (var option in question.GetComponent<QTLinearScale>().options)
                            {
                                jsonString += "{\"answer_value\":\"" + option.name.Split('_')[0] + "\",\"answer_option\":\"" + option.name.Split('_')[1] + "\"},";
                            }
                            jsonString = jsonString.Trim(',') + "]},";
                            break;
                        case "QTCheckboxes":
                            jsonString += "\"qType\":\"checkboxes\"," 
                                          + "\"question\":\"" + question.transform.GetChild(0).GetComponent<TextMeshProUGUI>().text + "\","
                                          + "\"header_name\":\"" + question.name.Split('_')[1] + "\","
                                          + "\"mandatory\":\"" + question.GetComponent<QTCheckboxes>().answerRequired + "\","
                                          + "\"include_other_option\":\"" + question.GetComponent<QTCheckboxes>().includeOtherOption + "\","
                                          + "\"options\":[";
                            foreach (var option in question.GetComponent<QTCheckboxes>().options)
                            {
                                jsonString += "{\"answer_value\":\"" + option.name.Split('_')[0] + "\",\"answer_option\":\"" + option.name.Split('_')[1] + "\"},";
                            }
                            jsonString = jsonString.Trim(',') + "]},";
                            break;
                        case "QTSlider":
                            var sliderScript = question.GetComponent<QTSlider>();
                            jsonString += "\"qType\":\"slider\"," 
                                          + "\"question\":\"" + question.transform.GetChild(0).GetComponent<TextMeshProUGUI>().text + "\","
                                          + "\"header_name\":\"" + question.name.Split('_')[1] + "\","
                                          + "\"mandatory\":\"" + sliderScript.answerRequired + "\","
                                          + "\"min_value\":\"" + sliderScript.minValue + "\","
                                          + "\"max_value\":\"" + sliderScript.maxValue + "\","
                                          + "\"whole_numbers\":\"" + sliderScript.wholeNumbers + "\","
                                          + "\"show_panels\":\"" + sliderScript.showPanels + "\","
                                          + "\"show_intermediate_panels\":\"" + sliderScript.showIntermediatePanels + "\","
                                          + "\"auto_labels\":\"" + sliderScript.automaticLabelNames + "\","
                                          + "\"label_zero\":\"" + sliderScript.labelZero + "\","
                                          + "\"label_quarter\":\"" + sliderScript.labelQuarter + "\","
                                          + "\"label_half\":\"" + sliderScript.labelHalf + "\","
                                          + "\"label_three_quarters\":\"" + sliderScript.labelThreeQuarters + "\","
                                          + "\"label_full\":\"" + sliderScript.labelFull + "\"},";
                            break;
                        case "QTMultipleChoice":
                            jsonString += "\"qType\":\"multiple_choice\","
                                          + "\"question\":\"" + question.transform.GetChild(0).GetComponent<TextMeshProUGUI>().text + "\","
                                          + "\"header_name\":\"" + question.name.Split('_')[1] + "\","
                                          + "\"mandatory\":\"" + question.GetComponent<QTMultipleChoice>().answerRequired + "\","
                                          + "\"include_other_option\":\"" + question.GetComponent<QTMultipleChoice>().includeOtherOption + "\","
                                          + "\"options\":[";
                            foreach (var option in question.GetComponent<QTMultipleChoice>().options)
                            {
                                jsonString += "{\"answer_value\":\"" + option.name.Split('_')[0] + "\",\"answer_option\":\"" + option.name.Split('_')[1] + "\"},";
                            }
                            jsonString = jsonString.Trim(',') + "]},";
                            break;
                        case "QTTextInput":
                            jsonString += question.name.Contains("Short") ? "\"qType\":\"text_input_short\"," : "\"qType\":\"text_input_long\",";
                            jsonString += "\"question\":\"" + question.transform.GetChild(0).GetComponent<TextMeshProUGUI>().text + "\","
                                          + "\"header_name\":\"" + question.name.Split('_')[1] + "\","
                                          + "\"mandatory\":\"" + question.GetComponent<QTTextInput>().answerRequired + "\","
                                          + "\"placeholder_text\":\"" + question.GetComponent<QTTextInput>().placeholderText + "\"},";
                            break;
                        case "QTDropdown":
                            jsonString += "\"qType\":\"dropdown\","
                                          + "\"question\":\"" + question.transform.GetChild(0).GetComponent<TextMeshProUGUI>().text + "\","
                                          + "\"header_name\":\"" + question.name.Split('_')[1] + "\","
                                          + "\"mandatory\":\"" + question.GetComponent<QTDropdown>().answerRequired + "\","
                                          + "\"options\":[";
                            foreach (var option in question.GetComponent<QTDropdown>().options)
                            {
                                jsonString += "{\"option_text\":\"" + option.text + "\"},";
                            }
                            jsonString = jsonString.Trim(',') + "]},";
                            break;
                        case "QTCheckboxesGrid":
                            jsonString += "\"qType\":\"checkboxes_grid\","
                                          + "\"question\":\"" + question.transform.GetChild(0).GetComponent<TextMeshProUGUI>().text + "\","
                                          + "\"header_name\":\"" + question.name.Split('_')[1] + "\","
                                          + "\"mandatory\":\"" + question.GetComponent<QTCheckboxesGrid>().answerRequired + "\","
                                          + "\"row_texts\":[";
                            foreach (var rowText in question.GetComponent<QTCheckboxesGrid>().rowTexts)
                            {
                                jsonString += "\"" + rowText + "\",";
                            }
                            jsonString = jsonString.Trim(',') + "]," + "\"column_texts\":[";
                            foreach (var columnText in question.GetComponent<QTCheckboxesGrid>().columnTexts)
                            {
                                jsonString += "\"" + columnText + "\",";
                            }
                            jsonString = jsonString.Trim(',') + "]},";
                            break;
                        case "QTMultipleChoiceGrid":
                            jsonString += "\"qType\":\"multiple_choice_grid\","
                                          + "\"question\":\"" + question.transform.GetChild(0).GetComponent<TextMeshProUGUI>().text + "\","
                                          + "\"header_name\":\"" + question.name.Split('_')[1] + "\","
                                          + "\"mandatory\":\"" + question.GetComponent<QTMultipleChoiceGrid>().answerRequired + "\","
                                          + "\"row_texts\":[";
                            foreach (var rowText in question.GetComponent<QTMultipleChoiceGrid>().rowTexts)
                            {
                                jsonString += "\"" + rowText + "\",";
                            }
                            jsonString = jsonString.Trim(',') + "]," + "\"column_texts\":[";
                            foreach (var columnText in question.GetComponent<QTMultipleChoiceGrid>().columnTexts)
                            {
                                jsonString += "\"" + columnText + "\",";
                            }
                            jsonString = jsonString.Trim(',') + "]},";
                            break;
                        case "QTText":
                            var child = question.transform.GetChild(0).GetComponent<TextMeshProUGUI>();
                            jsonString += "\"qType\":\"text\","
                                          + "\"text_content\":\"" + child.text.Replace('"','\"') + "\","
                                          + "\"font_style\":\"" + child.fontStyle + "\","
                                          + "\"font_size\":\"" + child.fontSize + "\","
                                          + "\"auto_sizing\":\"" + child.enableAutoSizing + "\","
                                          + "\"min_size\":\"" + child.fontSizeMin + "\","
                                          + "\"max_size\":\"" + child.fontSizeMax + "\","
                                          + "\"word_spacing\":\"" + child.wordSpacing + "\","
                                          + "\"line_spacing\":\"" + child.lineSpacing + "\","
                                          + "\"color\":\"" + child.color + "\","
                                          + "\"alignment\":\"" + child.alignment + "\","
                                          + "\"word_wrapping\":\"" + child.enableWordWrapping + "\","
                                          + "\"overflow_mode\":\"" + child.overflowMode + "\","
                                          + "\"horizontal_mapping\":\"" + child.horizontalMapping + "\","
                                          + "\"vertical_mapping\":\"" + child.verticalMapping + "\"},";
                            break;
                        case "QTImage":
                            var imageName = "missing";
                            try { imageName = question.transform.GetChild(0).GetComponent<Image>().sprite.name; } catch (Exception) { }
                            jsonString += "\"qType\":\"image\","
                                          + "\"image_name\":\"" + imageName + "\"},";
                            break;
                        case "QTButton":
                            jsonString += "\"qType\":\"button\","
                                          + "\"button_count\":\"" + question.transform.childCount + "\"},";
                            break;
                        case "QTVideo":
                            var videoName = "missing";
                            try { videoName = question.transform.GetChild(0).GetComponent<VideoPlayer>().clip.name; } catch (Exception) { }
                            jsonString += "\"qType\":\"video\","
                                          + "\"video_name\":\"" + videoName + "\"},";
                            break;
                    }
                }
                jsonString = jsonString.Trim(',') + "]},";
            }
            File.WriteAllText(savePath, jsonString.TrimEnd(',') + "]}");
            ShowPage(selectedPage);
        }

        /// <summary>
        /// Creates a new empty page.
        /// </summary>
        public void CreatePage(bool import = false, bool showFullscreenT = false, string fullscreenT = null, bool showTopP = false, string instr = null)
        {
            // load and instantiate the page prefab
            var o = Resources.Load("QuestionnaireToolkitPrefabs/QuestionPage");
            var g = Instantiate(o, transform, spawnObjectsInWorldSpace) as GameObject;
            questionPages.Add(g);
            g.name = "QuestionPage-" + questionPages.Count;
            g.GetComponent<Image>().color = pageBackgroundColor;
            g.transform.GetChild(1).GetComponent<Image>().color = pageBottomColor;
            g.transform.GetChild(6).GetComponent<Image>().color = pageBottomColor;

            // import page layout settings
            if (import)
            {
                var pageScript = g.GetComponent<QTQuestionPageManager>();
                pageScript._questionnaireManager = this;
                pageScript.showFullscreenText = showFullscreenT;
                pageScript.fullscreenText = fullscreenT;
                pageScript.showTopPanel = showTopP;
                pageScript.instructionText = instr;
                pageScript.OnValidate();
            }
            
            // if in VR mode set position and scale of the page as needed
            if (displayMode == DisplayMode.VR)
            {
                g.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
                var rectTransform = g.GetComponent<RectTransform>();
                rectTransform.anchoredPosition3D = Vector3.zero;
                rectTransform.sizeDelta = new Vector2(_pageWidth, pageHeight);
                rectTransform.localScale = Vector3.one;
            }
            else if (displayMode == DisplayMode.Desktop || displayMode == DisplayMode.Mobile)
            {
                g.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
            }
            g.GetComponent<Canvas>().worldCamera = Camera.main;
            
            // set contentSizeFitter according to the current orientation
            switch (orientation)
            {
                case Orientation.Horizontal:
                    g.transform.GetChild(0).GetChild(0).GetChild(0).GetComponent<ContentSizeFitter>().horizontalFit =
                        ContentSizeFitter.FitMode.PreferredSize;
                    break;
                case Orientation.Vertical:
                    g.transform.GetChild(0).GetChild(0).GetChild(0).GetComponent<ContentSizeFitter>().horizontalFit =
                        ContentSizeFitter.FitMode.MinSize;
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
            
            // set the visibility of all page elements
            var viewportTransform = g.transform.GetChild(0).GetComponent<RectTransform>();
            if (!import)
            {
                var offsetMax = viewportTransform.offsetMax;
                offsetMax = showTopPanel ? new Vector2(offsetMax.x, -56) : new Vector2(offsetMax.x, 0);
                viewportTransform.offsetMax = offsetMax;
                g.transform.GetChild(6).gameObject.SetActive(showTopPanel); // top panel
                g.transform.GetChild(7).gameObject.SetActive(showTopPanel); // instruction text
            }
            var offsetMin = viewportTransform.offsetMin;
            offsetMin = showBottomPanel ? new Vector2(offsetMin.x, 53) : new Vector2(offsetMin.x, 0);
            viewportTransform.offsetMin = offsetMin;
            g.transform.GetChild(1).gameObject.SetActive(showBottomPanel); // background
            g.transform.GetChild(2).gameObject.SetActive(showBottomPanel); // next button
            g.transform.GetChild(3).gameObject.SetActive(showBottomPanel ? showPageNumber : showBottomPanel); // page number
            g.transform.GetChild(5).gameObject.SetActive(showBottomPanel ? showPrevButton : showBottomPanel); // prev button

            // update the displayed page numbers for all pages
            RecreatePageNumbers();
            ShowPage(questionPages.Count - 1);

            if (import) currentImportedPage = g;
        }

#if UNITY_EDITOR
        /// <summary>
        /// Copy the selected page including its items.
        /// </summary>
        public void CopyPage()
        {
            var g = Instantiate(questionPages[selectedPage], transform, spawnObjectsInWorldSpace) as GameObject;
            questionPages.Add(g);
            g.name = "QuestionPage-" + questionPages.Count;
            RecreatePageNumbers();
            ShowPage(questionPages.Count - 1);
        }

        /// <summary>
        /// Deletes an entire page including its items.
        /// </summary>
        public void DeletePage(int listCount = -1, int sel = -1)
        {
            if (questionPages.Count <= 0 || selectedPage >= questionPages.Count || selectedPage < 0) return;
            if (listCount == -1 && sel == -1)
            {
                listCount = questionPages.Count - 1;
                sel = selectedPage;
            }
            if (listCount < questionPages.Count) // item was removed
            {
                DestroyImmediate(questionPages[sel]);
                questionPages.RemoveAt(sel);
                BuildHeaderItems();
                if (questionPages.Count <= 0) return;
                RecreatePageNumbers();
                ShowPage(questionPages.Count - 1);
            }
        }

        /// <summary>
        /// Reorders pages according to the order of the reorderable list inside the editor.
        /// </summary>
        public void ReorderPage(int listCount, int sel)
        {
            if (listCount == questionPages.Count)
            {
                questionPages[sel].transform.SetSiblingIndex(sel);
                RecreatePageNumbers();
                BuildHeaderItems();
            }
        }
#endif

        /// <summary>
        /// Runtime method to show the previous page
        /// </summary>
        public void PrevPage()
        {
            if (_currentPage > 0)
            {
                _currentPage--;
                ShowPage(_currentPage);
                if(_currentPage == 0)
                    _visiblePage.transform.GetChild(5).gameObject.SetActive(false);
                
                var pageSizeDelta = _visiblePage.GetComponent<RectTransform>().sizeDelta;
                if(displayMode == DisplayMode.VR) 
                    background.transform.localScale = new Vector3(pageSizeDelta.x, pageSizeDelta.y, 20);
            }
        }
        
        /// <summary>
        /// Runtime method to show the next page or to finish and submit the current questionnaire run.
        /// </summary>
        public void NextPage()
        {
            if (AnswersPending())
            {
                _visiblePage.transform.GetChild(4).GetComponent<TextMeshProUGUI>().enabled = true;
            }
            else
            {
                _visiblePage.transform.GetChild(4).GetComponent<TextMeshProUGUI>().enabled = false;
                if (_currentPage < questionPages.Count-1) // show only next page
                {
                    _currentPage++;
                    ShowPage(_currentPage);
                    _visiblePage.transform.GetChild(5).gameObject.SetActive(showPrevButton);
                    var pageSizeDelta = _visiblePage.GetComponent<RectTransform>().sizeDelta;
                    if(displayMode == DisplayMode.VR) 
                        background.transform.localScale = new Vector3(pageSizeDelta.x, pageSizeDelta.y, 20);
                }
                else if (_currentPage == questionPages.Count-1) // finish and submit the questionnaire
                {
                    // Check for BO Manager and collect questionnaire answers even when results file writing is disabled.
                    // Use type lookup instead of tag lookup to avoid hard dependency on a specific scene tag.
                    var bomanager = FindObjectOfType<BoForUnityManager>();
                    var shouldCollectForBo = bomanager != null;

                    if (generateResultsFile)
                    {
                        qtMetaData.currentResponseId++;
                        currentResponseId++;
                        if (runsPerUser > 1 && qtMetaData.currentUserRun < runsPerUser)
                        {
                            qtMetaData.currentUserRun++;
                            if (qtMetaData.currentUserRun > 1)
                            {
                                // revert the ++ operation if the response_id should stay the same
                                qtMetaData.currentResponseId--;
                                currentResponseId--;
                            }
                        }

                        // Keep runtime run counter aligned with persisted metadata.
                        currentRun = qtMetaData.currentUserRun;
                    }

                    if (generateResultsFile || shouldCollectForBo)
                    {
                        finishedTimestamp = DateTime.UtcNow + "";
                        WriteResults();
                    }

                    ResetQuestionnaire();
                    HideQuestionnaire();
                    running = false;
                    onQuestionnaireFinished?.Invoke();

                    // check for BO Manager and start optimization as the questionnaire has finished
                    if (bomanager != null)
                    {
                        bomanager.OptimizationStart();
                        // In external-signal mode, queue progression so the manager advances
                        // once new parameters arrive from Python.
                        if (bomanager.iterationAdvanceMode == BoForUnityManager.IterationAdvanceMode.ExternalSignal)
                        {
                            bomanager.RequestNextIteration();
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Recreates the page numbers for all pages and set the text of the 'next' button according to the current page order.
        /// </summary>
        private void RecreatePageNumbers()
        {
            for (var i = 0; i < questionPages.Count; i++)
            {
                questionPages[i].transform.GetChild(3).GetComponent<TextMeshProUGUI>().text = (i+1) + " / " + questionPages.Count;
                questionPages[i].transform.GetChild(2).GetChild(0).GetComponent<TextMeshProUGUI>().text = "Next";
                questionPages[i].transform.GetChild(5).gameObject.SetActive(showBottomPanel ? showPrevButton && i > 0 : showBottomPanel);
            }
            questionPages[questionPages.Count - 1].transform.GetChild(2).GetChild(0).GetComponent<TextMeshProUGUI>().text = "Finish";
        }

        /// <summary>
        /// Shows the desired page by the given page number. Hides all other pages.
        /// </summary>
        public void ShowPage(int pageNumber)
        {
            if (questionPages.Count == 0 || pageNumber < 0 || pageNumber >= questionPages.Count) return;
            
            foreach (var page in questionPages)
            {
                page.SetActive(false);
            }
            _visiblePage = questionPages[pageNumber];
            _visiblePage.SetActive(true);
            
            selectedPage = pageNumber;
            _currentPage = pageNumber;
        }

        /// <summary>
        /// Iterates over all question item header names and adds them to the resultsHeaderItems list.
        /// </summary>
        public void BuildHeaderItems()
        {
            if (overwriteResultsHeaderItems) return;
            
            resultsHeaderItems.Clear();
        
            foreach (var page in questionPages)
            {
                try
                {
                    foreach (var question in page.GetComponent<QTQuestionPageManager>().questionItems)
                    {
                        if (question.CompareTag("QTText") || question.CompareTag("QTImage") || question.CompareTag("QTButton") || question.CompareTag("QTVideo")) continue;

                        if (question.CompareTag("QTCheckboxesGrid"))
                        {
                            var grid = question.transform.GetChild(1);
                            var questionHeader = GetQuestionHeaderFromObjectName(question.name);
                            var rowHeaders = new List<string>();
                            for (var i = 0; i < grid.childCount; i++)
                            {
                                var child = grid.GetChild(i);
                                if (child.CompareTag("QTGridRowHeader"))
                                {
                                    rowHeaders.Add(GetGridRowHeaderLabel(child, rowHeaders.Count + 1));
                                }
                            }

                            var checkboxGridScript = question.GetComponent<QTCheckboxesGrid>();
                            var configuredRowCount = checkboxGridScript != null && checkboxGridScript.rowTexts != null
                                ? checkboxGridScript.rowTexts.Count
                                : 0;
                            var emittedRowCount = configuredRowCount > 0 ? configuredRowCount : rowHeaders.Count;

                            for (var rowIndex = 0; rowIndex < emittedRowCount; rowIndex++)
                            {
                                var header = rowIndex < rowHeaders.Count
                                    ? rowHeaders[rowIndex]
                                    : $"{questionHeader}_{rowIndex + 1}";
                                resultsHeaderItems.Add(TryHeaderName(header));
                            }
                        }
                        else if (question.CompareTag("QTMultipleChoiceGrid"))
                        {
                            var grid = question.transform.GetChild(1);
                            var rowHeaderIndex = 1;
                            for (var i = 0; i < grid.childCount; i++)
                            {
                                var child = grid.GetChild(i);
                                if (!child.CompareTag("QTGridRowHeader"))
                                    continue;

                                resultsHeaderItems.Add(TryHeaderName(GetGridRowHeaderLabel(child, rowHeaderIndex)));
                                rowHeaderIndex++;
                            }
                        }
                        else
                        {
                            resultsHeaderItems.Add(TryHeaderName(GetQuestionHeaderFromObjectName(question.name)));
                        }
                    }
                }
                catch (Exception)
                {
                    // ignored
                }
            }
        }

        private static void TryAddBoObjectiveValue(
            BoForUnityManager boForUnity,
            string headerName,
            string rawValue,
            string sourceName)
        {
            if (boForUnity == null || boForUnity.optimizer == null)
            {
                return;
            }
            if (boForUnity.objectives == null || boForUnity.objectives.Count == 0)
            {
                return;
            }

            if (string.IsNullOrWhiteSpace(headerName))
            {
                Debug.LogWarning($"Skipping BO objective update from '{sourceName}' because header name is empty.");
                return;
            }

            var headerMatchesObjective = false;
            foreach (var objective in boForUnity.objectives)
            {
                var objectiveKey = objective?.key?.Trim();
                if (objective != null &&
                    !string.IsNullOrWhiteSpace(objectiveKey) &&
                    ContainsObjectiveKeyMatch(headerName, objectiveKey))
                {
                    headerMatchesObjective = true;
                    break;
                }
            }

            if (!headerMatchesObjective)
            {
                return;
            }

            if (!float.TryParse(rawValue, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
            {
                Debug.LogWarning(
                    $"Objective value for '{headerName}' from '{sourceName}' is not numeric ('{rawValue}'). " +
                    "Submitting NaN so this iteration uses BO fallback handling instead of reusing a stale value."
                );
                boForUnity.optimizer.AddObjectiveValue(headerName, float.NaN);
                return;
            }

            boForUnity.optimizer.AddObjectiveValue(headerName, value);
        }

        private static bool ContainsObjectiveKeyMatch(string source, string objectiveKey)
        {
            if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(objectiveKey))
                return false;

            int start = 0;
            while (start < source.Length)
            {
                int idx = source.IndexOf(objectiveKey, start, StringComparison.OrdinalIgnoreCase);
                if (idx < 0)
                    return false;

                int end = idx + objectiveKey.Length;
                if (IsObjectiveBoundary(source, idx) && IsObjectiveBoundary(source, end))
                    return true;

                start = idx + 1;
            }

            return false;
        }

        private static bool IsObjectiveBoundary(string text, int boundaryIndex)
        {
            if (boundaryIndex <= 0 || boundaryIndex >= text.Length)
                return true;

            char left = text[boundaryIndex - 1];
            char right = text[boundaryIndex];
            if (!char.IsLetterOrDigit(left) || !char.IsLetterOrDigit(right))
                return true;

            if (char.IsLetter(left) && char.IsLetter(right) && char.IsLower(left) && char.IsUpper(right))
                return true;

            if (char.IsLetter(left) && char.IsDigit(right))
                return true;
            if (char.IsDigit(left) && char.IsLetter(right))
                return true;

            return false;
        }

        private static string GetNameTokenOrDefault(string source, int tokenIndex, string fallback = "NULL")
        {
            if (string.IsNullOrEmpty(source))
            {
                return fallback;
            }

            var tokens = source.Split('_');
            if (tokenIndex >= 0 && tokenIndex < tokens.Length && !string.IsNullOrEmpty(tokens[tokenIndex]))
            {
                return tokens[tokenIndex];
            }

            return fallback;
        }

        private static string GetQuestionHeaderFromObjectName(string objectName)
        {
            if (string.IsNullOrWhiteSpace(objectName))
                return "NULL";

            var nameComponents = objectName.Split('_');
            if (nameComponents.Length <= 1)
                return objectName;

            var header = string.Join("_", nameComponents.Skip(1).Where(part => !string.IsNullOrEmpty(part)));
            return string.IsNullOrWhiteSpace(header) ? objectName : header;
        }

        private static string GetQuestionHeaderName(GameObject question)
        {
            if (question == null)
                return "NULL";
            return GetQuestionHeaderFromObjectName(question.name);
        }

        private BoForUnityManager GetBoForUnityManager()
        {
            if (_cachedBoForUnityManager == null)
            {
                _cachedBoForUnityManager = FindObjectOfType<BoForUnityManager>();
            }

            return _cachedBoForUnityManager;
        }

        private string BuildPriorRatingHintKey(GameObject question)
        {
            if (question == null)
                return string.Empty;

            string questionnaireToken = string.IsNullOrWhiteSpace(resultsFileName) ? gameObject.name : resultsFileName.Trim();
            var page = question.GetComponentInParent<QTQuestionPageManager>();
            string pageToken = page != null && !string.IsNullOrWhiteSpace(page.name) ? page.name : "Page";
            string headerToken = GetQuestionHeaderName(question);
            int questionIndex = -1;
            if (page != null && page.questionItems != null)
            {
                questionIndex = page.questionItems.IndexOf(question);
            }
            if (questionIndex < 0)
            {
                questionIndex = question.transform.GetSiblingIndex();
            }

            return questionnaireToken + "::" + pageToken + "::" + question.tag + "::" +
                   questionIndex.ToString(CultureInfo.InvariantCulture) + "::" + headerToken;
        }

        private void TryStorePriorSliderRatingHint(BoForUnityManager boForUnity, GameObject question, string rawValue)
        {
            if (boForUnity == null || question == null)
                return;

            if (!float.TryParse(rawValue, NumberStyles.Float, CultureInfo.InvariantCulture, out float sliderValue))
                return;

            if (float.IsNaN(sliderValue) || float.IsInfinity(sliderValue))
                return;

            string key = BuildPriorRatingHintKey(question);
            if (string.IsNullOrWhiteSpace(key))
                return;

            boForUnity.SetPriorSliderRatingHint(key, sliderValue);
        }

        private void TryClearPriorSliderRatingHint(BoForUnityManager boForUnity, GameObject question)
        {
            if (boForUnity == null || question == null)
                return;

            string key = BuildPriorRatingHintKey(question);
            if (string.IsNullOrWhiteSpace(key))
                return;

            boForUnity.RemovePriorSliderRatingHint(key);
        }

        private void TryStorePriorLinearScaleRatingHint(BoForUnityManager boForUnity, GameObject question, string rawValue)
        {
            if (boForUnity == null || question == null)
                return;

            if (string.IsNullOrWhiteSpace(rawValue) || string.Equals(rawValue, "NULL", StringComparison.OrdinalIgnoreCase))
                return;

            string key = BuildPriorRatingHintKey(question);
            if (string.IsNullOrWhiteSpace(key))
                return;

            boForUnity.SetPriorLinearScaleRatingHint(key, rawValue.Trim());
        }

        private void TryClearPriorLinearScaleRatingHint(BoForUnityManager boForUnity, GameObject question)
        {
            if (boForUnity == null || question == null)
                return;

            string key = BuildPriorRatingHintKey(question);
            if (string.IsNullOrWhiteSpace(key))
                return;

            boForUnity.RemovePriorLinearScaleRatingHint(key);
        }

        private void ApplyPriorSliderRatingHint(GameObject question, QTSlider sliderQuestion)
        {
            if (sliderQuestion == null)
                return;

            BoForUnityManager boForUnity = GetBoForUnityManager();
            if (boForUnity == null || !boForUnity.enablePriorSliderRatingHint)
            {
                sliderQuestion.HidePriorRatingHint();
                return;
            }

            string key = BuildPriorRatingHintKey(question);
            if (!boForUnity.TryGetPriorSliderRatingHint(key, out float priorValue))
            {
                sliderQuestion.HidePriorRatingHint();
                return;
            }

            sliderQuestion.ApplyPriorRatingHint(
                enabled: true,
                priorValue: priorValue,
                alpha: boForUnity.priorSliderRatingHintAlpha
            );
        }

        private void ApplyPriorLinearScaleRatingHint(GameObject question, QTLinearScale linearScaleQuestion)
        {
            if (linearScaleQuestion == null)
                return;

            BoForUnityManager boForUnity = GetBoForUnityManager();
            if (boForUnity == null || !boForUnity.enablePriorSliderRatingHint)
            {
                linearScaleQuestion.HidePriorRatingHint();
                return;
            }

            string key = BuildPriorRatingHintKey(question);
            if (!boForUnity.TryGetPriorLinearScaleRatingHint(key, out string priorValue))
            {
                linearScaleQuestion.HidePriorRatingHint();
                return;
            }

            linearScaleQuestion.ApplyPriorRatingHint(
                enabled: true,
                priorAnswerValue: priorValue,
                alpha: boForUnity.priorSliderRatingHintAlpha
            );
        }

        private static string GetGridRowHeaderLabel(Transform rowHeaderTransform, int fallbackIndex)
        {
            if (rowHeaderTransform == null)
                return $"[Row{fallbackIndex}]";

            string label = null;
            if (rowHeaderTransform.childCount > 0)
            {
                var textComponent = rowHeaderTransform.GetChild(0).GetComponent<TextMeshProUGUI>();
                if (textComponent != null)
                {
                    label = textComponent.text;
                }
            }

            if (string.IsNullOrWhiteSpace(label))
            {
                label = rowHeaderTransform.name;
            }

            return "[" + label + "]";
        }

        private static void AppendEmittedValue(
            List<string> emittedValues,
            List<string> emittedHeaders,
            string value,
            string header,
            HashSet<string> usedHeaders)
        {
            if (emittedValues == null || emittedHeaders == null)
                return;

            emittedValues.Add(value ?? "NULL");
            emittedHeaders.Add(GetUniqueHeaderName(header, usedHeaders));
        }

        private static string GetUniqueHeaderName(string header, HashSet<string> usedHeaders)
        {
            string baseHeader = string.IsNullOrWhiteSpace(header) ? "NULL" : header.Trim();
            if (usedHeaders == null)
                return baseHeader;

            if (!usedHeaders.Contains(baseHeader))
            {
                usedHeaders.Add(baseHeader);
                return baseHeader;
            }

            int suffix = 1;
            while (true)
            {
                string candidate = baseHeader + "_" + suffix.ToString(CultureInfo.InvariantCulture);
                if (!usedHeaders.Contains(candidate))
                {
                    usedHeaders.Add(candidate);
                    return candidate;
                }
                suffix++;
            }
        }

        private string ResolveObjectiveHeaderName(string fallbackHeader, int emittedIndex)
        {
            if (!overwriteResultsHeaderItems || resultsHeaderItems == null)
                return fallbackHeader;
            if (emittedIndex < 0 || emittedIndex >= resultsHeaderItems.Count)
                return fallbackHeader;

            string configuredHeader = resultsHeaderItems[emittedIndex];
            return string.IsNullOrWhiteSpace(configuredHeader) ? fallbackHeader : configuredHeader.Trim();
        }

        private static string EscapeCsvCell(string value)
        {
            if (value == null)
                return string.Empty;

            bool mustQuote =
                value.IndexOf(',') >= 0 ||
                value.IndexOf('"') >= 0 ||
                value.IndexOf('\n') >= 0 ||
                value.IndexOf('\r') >= 0;

            if (!mustQuote)
                return value;

            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        /// <summary>
        /// Writes the results of the current questionnaire run in the specified results file.
        /// </summary>
        private void WriteResults()
        {
            var rowCells = new List<string> { currentResponseId.ToString(CultureInfo.InvariantCulture) };
            if (customUserId)
            {
                rowCells.Add(userId);
                if (generateResultsFile && runsPerUser == 1)
                    userId = "";
            }
            if (runsPerUser > 1)
            {
                rowCells.Add(currentRun.ToString(CultureInfo.InvariantCulture));
            }
            if (generateStartTimestamp)
            {
                rowCells.Add(startedTimestamp);
            }
            if (generateFinishTimestamp)
            {
                rowCells.Add(finishedTimestamp);
            }

            //-----------
            // Check if BO for Unity Manager is present in the scene
            var boManager = false;
            var boForUnity = FindObjectOfType<BoForUnityManager>();
            if (boForUnity != null)
            {
                boManager = true;
            }
            HashSet<string> boHeadersUsed = boManager
                ? new HashSet<string>(StringComparer.Ordinal)
                : null;
            //-----------
            
            var currVal = ""; // this is the value that is assigned to each question item value in the following
            var emittedQuestionValueCount = 0;
            
            foreach (var page in questionPages)
            {
                page.SetActive(true); // re-enable all pages otherwise the items cannot be found
                
                foreach (var question in page.GetComponent<QTQuestionPageManager>().questionItems)
                {
                    if (question.CompareTag("QTText") || question.CompareTag("QTImage") ||
                        question.CompareTag("QTButton") || question.CompareTag("QTVideo"))
                    {
                        continue;
                    }

                    var emittedValues = new List<string>();
                    var emittedHeaders = new List<string>();
                    var questionHeader = GetQuestionHeaderName(question);
                    switch (question.tag)
                    {
                        case "QTLinearScale":
                            var toggleGroup = question.transform.GetChild(1).GetComponent<ToggleGroup>();
                            var activeLinearToggle = toggleGroup.ActiveToggles().FirstOrDefault();
                            if (activeLinearToggle == null)
                            {
                                currVal = "NULL";
                                rowCells.Add("NULL");
                            }
                            else
                            {
                                currVal = GetNameTokenOrDefault(activeLinearToggle.gameObject.name, 0, activeLinearToggle.gameObject.name);
                                rowCells.Add(currVal);
                            }
                            AppendEmittedValue(emittedValues, emittedHeaders, currVal, questionHeader, boHeadersUsed);
                            if (!string.Equals(currVal, "NULL", StringComparison.OrdinalIgnoreCase))
                            {
                                TryStorePriorLinearScaleRatingHint(boForUnity, question, currVal);
                            }
                            else
                            {
                                TryClearPriorLinearScaleRatingHint(boForUnity, question);
                            }
                            break;
                        case "QTCheckboxes":
                            var checkboxesResults = "";
                            for (var c = 0; c < question.transform.GetChild(1).childCount; c++)
                            {
                                var checkbox = question.transform.GetChild(1).GetChild(c).GetComponent<Toggle>();
                                if (checkbox.isOn)
                                {
                                    if (checkbox.CompareTag("QTOptionOther"))
                                    {
                                        checkboxesResults += "" + checkbox.transform.GetChild(1).GetComponent<TMP_InputField>().text + ";";
                                    }
                                    else
                                    {
                                        checkboxesResults += GetNameTokenOrDefault(checkbox.name, 0, checkbox.name) + ";";
                                    }
                                }
                            }
                            currVal = checkboxesResults.TrimEnd(';');
                            rowCells.Add(currVal);
                            AppendEmittedValue(emittedValues, emittedHeaders, currVal, questionHeader, boHeadersUsed);
                            break;
                        case "QTSlider":
                            var sliderQuestion = question.GetComponent<QTSlider>();
                            var sliderComponent = question.transform.GetChild(1).GetComponent<UnityEngine.UI.Slider>();
                            if (sliderQuestion != null && !sliderQuestion.HasRuntimeAnswer())
                            {
                                currVal = "NULL";
                            }
                            else
                            {
                                currVal = sliderComponent.value.ToString(CultureInfo.InvariantCulture);
                            }
                            rowCells.Add(currVal);
                            AppendEmittedValue(emittedValues, emittedHeaders, currVal, questionHeader, boHeadersUsed);
                            if (!string.Equals(currVal, "NULL", StringComparison.OrdinalIgnoreCase))
                            {
                                TryStorePriorSliderRatingHint(boForUnity, question, currVal);
                            }
                            else
                            {
                                TryClearPriorSliderRatingHint(boForUnity, question);
                            }
                            break;
                        case "QTMultipleChoice":
                            var toggleGroupMc = question.transform.GetChild(1).GetComponent<ToggleGroup>();
                            var toggledOption = toggleGroupMc.ActiveToggles().FirstOrDefault();
                            if (toggledOption == null)
                            {
                                currVal = "NULL";
                                rowCells.Add("NULL");
                            }
                            else if (toggledOption.CompareTag("QTOptionOther"))
                            {
                                currVal = toggledOption.transform.GetChild(1).GetComponent<TMP_InputField>().text;
                                rowCells.Add(currVal);
                            }
                            else
                            {
                                currVal = GetNameTokenOrDefault(toggledOption.gameObject.name, 0, toggledOption.gameObject.name);
                                rowCells.Add(currVal);
                            }
                            AppendEmittedValue(emittedValues, emittedHeaders, currVal, questionHeader, boHeadersUsed);
                            break;
                        case "QTTextInput":
                            currVal = question.transform.GetChild(1).GetComponent<TMP_InputField>().text;
                            rowCells.Add(currVal);
                            AppendEmittedValue(emittedValues, emittedHeaders, currVal, questionHeader, boHeadersUsed);
                            break;
                        case "QTDropdown":
                            var dropdown = question.transform.GetChild(1).GetComponent<TMP_Dropdown>();
                            var tmp = dropdown.value;
                            if (dropdown.options == null || dropdown.options.Count == 0 || tmp < 0 || tmp >= dropdown.options.Count)
                            {
                                currVal = "NULL";
                                rowCells.Add("NULL");
                            }
                            else
                            {
                                currVal = dropdown.options[tmp].text;
                                rowCells.Add(currVal);
                            }
                            AppendEmittedValue(emittedValues, emittedHeaders, currVal, questionHeader, boHeadersUsed);
                            break;
                        case "QTCheckboxesGrid":
                            var checkboxGrid = question.transform.GetChild(1);
                            var optionCounter = 0;
                            var checkboxGridScript = question.GetComponent<QTCheckboxesGrid>();
                            var rowHeaders = new List<string>();
                            for (var i = 0; i < checkboxGrid.childCount; i++)
                            {
                                var child = checkboxGrid.GetChild(i);
                                if (child.CompareTag("QTGridRowHeader"))
                                {
                                    rowHeaders.Add(GetGridRowHeaderLabel(child, rowHeaders.Count + 1));
                                }
                            }
                            var columnCount = checkboxGridScript != null && checkboxGridScript.columnTexts != null
                                ? checkboxGridScript.columnTexts.Count
                                : 0;
                            var rowCount = checkboxGridScript != null && checkboxGridScript.rowTexts != null
                                ? checkboxGridScript.rowTexts.Count
                                : rowHeaders.Count;
                            string GetCheckboxGridRowHeader(int rowIndex)
                            {
                                if (rowIndex >= 0 && rowIndex < rowHeaders.Count)
                                    return rowHeaders[rowIndex];
                                return $"{questionHeader}_{rowIndex + 1}";
                            }

                            if (columnCount <= 0)
                            {
                                for (var r = 0; r < rowCount; r++)
                                {
                                    currVal = "";
                                    rowCells.Add(currVal);
                                    AppendEmittedValue(emittedValues, emittedHeaders, currVal, GetCheckboxGridRowHeader(r), boHeadersUsed);
                                }
                                break;
                            }

                            var rowSelections = "";
                            var rowOutputIndex = 0;
                            for (var i = 0; i < checkboxGrid.childCount; i++)
                            {
                                var currChild = checkboxGrid.GetChild(i);
                                if (currChild.CompareTag("QTGridOption"))
                                {
                                    if (currChild.GetComponent<Toggle>().isOn)
                                    {
                                        currVal = GetNameTokenOrDefault(currChild.name, 1);
                                        rowSelections += "" + currVal + ";";
                                    }
                                    optionCounter++;
                                }
                                if (optionCounter == columnCount)
                                {
                                    optionCounter = 0;
                                    if (rowOutputIndex >= rowCount)
                                    {
                                        rowSelections = "";
                                        continue;
                                    }

                                    currVal = rowSelections.TrimEnd(';');
                                    rowCells.Add(currVal);
                                    AppendEmittedValue(
                                        emittedValues,
                                        emittedHeaders,
                                        currVal,
                                        GetCheckboxGridRowHeader(rowOutputIndex),
                                        boHeadersUsed);
                                    rowSelections = "";
                                    rowOutputIndex++;
                                }
                            }

                            while (rowOutputIndex < rowCount)
                            {
                                currVal = "";
                                rowCells.Add(currVal);
                                AppendEmittedValue(
                                    emittedValues,
                                    emittedHeaders,
                                    currVal,
                                    GetCheckboxGridRowHeader(rowOutputIndex),
                                    boHeadersUsed);
                                rowOutputIndex++;
                            }
                            break;
                        case "QTMultipleChoiceGrid":
                            var grid = question.transform.GetChild(1);
                            for (var i = 0; i < grid.childCount; i++)
                            {
                                var currChild = grid.GetChild(i);
                                if (currChild.CompareTag("QTGridRowHeader"))
                                {
                                    string rowHeader = GetGridRowHeaderLabel(currChild, emittedValues.Count + 1);
                                    var toggleGroupRow = currChild.GetComponent<ToggleGroup>();
                                    var activeRowToggle = toggleGroupRow.ActiveToggles().FirstOrDefault();
                                    if (activeRowToggle == null)
                                    {
                                        currVal = "NULL";
                                        rowCells.Add("NULL");
                                    }
                                    else
                                    {
                                        currVal = GetNameTokenOrDefault(activeRowToggle.gameObject.name, 1);
                                        rowCells.Add(currVal);
                                    }
                                    AppendEmittedValue(emittedValues, emittedHeaders, currVal, rowHeader, boHeadersUsed);
                                }
                            }
                            break;
                        default:
                            currVal = "NULL";
                            rowCells.Add("NULL");
                            AppendEmittedValue(emittedValues, emittedHeaders, currVal, questionHeader, boHeadersUsed);
                            break;
                    }
                    
                    // add the current question item value as objective function value
                    // addObjectiveValue("Trust", 3f);
                    if (boManager)
                    {
                        for (var i = 0; i < emittedValues.Count; i++)
                        {
                            string objectiveHeader = i < emittedHeaders.Count ? emittedHeaders[i] : questionHeader;
                            objectiveHeader = ResolveObjectiveHeaderName(objectiveHeader, emittedQuestionValueCount);
                            TryAddBoObjectiveValue(
                                boForUnity,
                                objectiveHeader,
                                emittedValues[i],
                                question.name
                            );
                            emittedQuestionValueCount++;
                        }
                    }
                }
            }

            if (additionalCsvItems != null)
            {
                foreach (var aci in additionalCsvItems)
                {
                    if (aci == null)
                    {
                        Debug.LogWarning("Skipping null entry in additionalCsvItems while writing questionnaire results.");
                        continue;
                    }

                    if (aci.itemValue != null && aci.itemValue.isAssigned)
                    {
                        currVal = aci.itemValue.Get() + "";
                        rowCells.Add(currVal);
                    }
                    else
                    {
                        rowCells.Add("NULL");
                    }
                }
            }

            try
            {
                if (generateResultsFile)
                {
                    using (var writer = new StreamWriter(userPath, true))
                    {
                        writer.WriteLine(string.Join(",", rowCells.Select(EscapeCsvCell)));
                    }
                    Debug.Log("Write results for: " + resultsFileName + (newFileEachStart ? "_" + currentResponseId : "") +
                              (runsPerUser > 1 ? " run: " + currentRun : "") + ".\n@ " + userPath);
                    if (currentRun == runsPerUser)
                    {
                        qtMetaData.currentUserRun = 0;
                        currentRun = 0;
                        userId = "";
                    }
                    SaveMetaData(currentResponseId, currentRun);
                }
            }
            catch (Exception)
            {
                Debug.Log("Write results failed!");
            }
        }

        /// <summary>
        /// Resets all question items to enable a new questionnaire run without restarting the application.
        /// </summary>
        public void ResetQuestionnaire()
        {
            _currentPage = 0;
            _visiblePage = questionPages[_currentPage];
            foreach (var page in questionPages)
            {
                page.SetActive(true);
                foreach (var question in page.GetComponent<QTQuestionPageManager>().questionItems)
                {
                    switch (question.tag)
                    {
                        case "QTLinearScale":
                            var linearScaleQuestion = question.GetComponent<QTLinearScale>();
                            try
                            {
                                question.transform.GetChild(1).GetComponent<ToggleGroup>().ActiveToggles().FirstOrDefault().isOn = false;
                            }
                            catch (Exception) { }
                            ApplyPriorLinearScaleRatingHint(question, linearScaleQuestion);
                            break;
                        case "QTCheckboxes":
                            for (var c = 0; c < question.transform.GetChild(1).childCount; c++)
                            {
                                var checkbox = question.transform.GetChild(1).GetChild(c);
                                checkbox.GetComponent<Toggle>().isOn = false;
                                if (checkbox.CompareTag("QTOptionOther"))
                                    checkbox.transform.GetChild(1).GetComponent<TMP_InputField>().text = "";
                            }
                            break;
                        case "QTSlider":
                            var slider = question.transform.GetChild(1).GetComponent<UnityEngine.UI.Slider>();
                            var sliderQuestion = question.GetComponent<QTSlider>();
                            if (sliderQuestion != null)
                            {
                                sliderQuestion.ResetRuntimeAnswerState();
                            }
                            slider.SetValueWithoutNotify(slider.minValue);
                            ApplyPriorSliderRatingHint(question, sliderQuestion);
                            break;
                        case "QTMultipleChoice":
                            try
                            {
                                var activeToggle = question.transform.GetChild(1).GetComponent<ToggleGroup>().ActiveToggles().FirstOrDefault();
                                activeToggle.isOn = false;
                                activeToggle.transform.GetChild(1).GetComponent<TMP_InputField>().text = "";
                            } catch (Exception) { }
                            break;
                        case "QTTextInput":
                            question.transform.GetChild(1).GetComponent<TMP_InputField>().text = "";
                            break;
                        case "QTDropdown":
                            question.transform.GetChild(1).GetComponent<TMP_Dropdown>().value = 0;
                            break;
                        case "QTCheckboxesGrid":
                            var checkboxGrid = question.transform.GetChild(1);
                            for (var c = 0; c < checkboxGrid.childCount; c++)
                            {
                                var currChild = checkboxGrid.GetChild(c);
                                if (currChild.CompareTag("QTGridOption"))
                                {
                                    currChild.GetComponent<Toggle>().isOn = false;
                                }
                            }
                            break;
                        case "QTMultipleChoiceGrid":
                            var grid = question.transform.GetChild(1);
                            for (var c = 0; c < grid.childCount; c++)
                            {
                                var currChild = grid.GetChild(c);
                                if (currChild.CompareTag("QTGridRowHeader"))
                                {
                                    try
                                    {
                                        currChild.GetComponent<ToggleGroup>().ActiveToggles().FirstOrDefault().isOn = false;
                                    }
                                    catch (Exception) { }
                                }
                            }
                            break;
                    }
                }
                page.SetActive(false);
            }
            _visiblePage.SetActive(true);
        }
        
        /// <summary>
        /// Checks if there are any answers pending which are marked as required. Returns true if any item is still pending.
        /// </summary>
        public bool AnswersPending()
        {
            var pending = false;
            foreach (var question in _visiblePage.GetComponent<QTQuestionPageManager>().questionItems)
            {
                try
                {
                    switch (question.tag)
                    {
                        case "QTLinearScale":
                            if (question.GetComponent<QTLinearScale>().answerRequired)
                            {
                                var test = question.transform.GetChild(1).GetComponent<ToggleGroup>().ActiveToggles().FirstOrDefault().gameObject;
                            }
                            question.GetComponent<Image>().color = Color.white;
                            break;
                        case "QTCheckboxes":
                            if (question.GetComponent<QTCheckboxes>().answerRequired)
                            {
                                question.GetComponent<Image>().color = new Color(1, 0.316f, 0.316f, 0.2235f);
                                var checkBoxPending = true;
                                for (var c = 0; c < question.transform.GetChild(1).childCount; c++)
                                {
                                    if (question.transform.GetChild(1).GetChild(c).GetComponent<Toggle>().isOn)
                                    {
                                        question.GetComponent<Image>().color = Color.white;
                                        checkBoxPending = false;
                                    }
                                }
                                if (!pending) // only set pending if it is not true already!
                                    pending = checkBoxPending;
                            }
                            break;
                        case "QTSlider":
                            var sliderQuestion = question.GetComponent<QTSlider>();
                            if (sliderQuestion.answerRequired)
                            {
                                if (!sliderQuestion.HasRuntimeAnswer())
                                {
                                    question.GetComponent<Image>().color = new Color(1, 0.316f, 0.316f, 0.2235f);
                                    pending = true;
                                }
                                else
                                {
                                    question.GetComponent<Image>().color = Color.white;
                                }
                            }
                            break;
                        case "QTMultipleChoice":
                            if (question.GetComponent<QTMultipleChoice>().answerRequired)
                            {
                                var test = question.transform.GetChild(1).GetComponent<ToggleGroup>().ActiveToggles().FirstOrDefault().gameObject;
                            }
                            question.GetComponent<Image>().color = Color.white;
                            break;
                        case "QTTextInput":
                            if (question.GetComponent<QTTextInput>().answerRequired)
                            {
                                if (question.transform.GetChild(1).GetComponent<TMP_InputField>().text.Equals(""))
                                {
                                    question.GetComponent<Image>().color = new Color(1, 0.316f, 0.316f, 0.2235f);
                                    pending = true;
                                }
                                else
                                {
                                    question.GetComponent<Image>().color = Color.white;
                                }
                            }
                            break;
                        case "QTDropdown":
                            if (question.GetComponent<QTDropdown>().answerRequired)
                            {
                                var dropdown = question.transform.GetChild(1).GetComponent<TMP_Dropdown>();
                                var selectedIndex = dropdown.value;
                                var hasValidSelection = dropdown.options != null &&
                                                        selectedIndex >= 0 &&
                                                        selectedIndex < dropdown.options.Count &&
                                                        !string.IsNullOrWhiteSpace(dropdown.options[selectedIndex].text);
                                if (!hasValidSelection)
                                {
                                    question.GetComponent<Image>().color = new Color(1, 0.316f, 0.316f, 0.2235f);
                                    pending = true;
                                }
                                else
                                {
                                    question.GetComponent<Image>().color = Color.white;
                                    //question.GetComponent<Image>().color = new Color(0.7924f, 0.7924f, 0.7924f, 0.2235f);
                                }
                            }
                            break;
                        case "QTCheckboxesGrid":
                            if (question.GetComponent<QTCheckboxesGrid>().answerRequired)
                            {
                                var checkBoxPending = false;
                                var checkboxesGrid = question.GetComponent<QTCheckboxesGrid>();
                                var cGrid = question.transform.GetChild(1);
                                var rows = checkboxesGrid.rowTexts.Count;
                                var cols = checkboxesGrid.columnTexts.Count;

                                for (var r = cols+1; r < (rows+1) * (cols+1); r += cols+1)
                                {
                                    var rowPending = true;
                                    for (var o = r+1; o <= r + cols; o++)
                                    {
                                        if (cGrid.GetChild(o).GetComponent<Toggle>().isOn)
                                        {
                                            rowPending = false;
                                        }
                                    }

                                    if (rowPending)
                                        checkBoxPending = true;
                                }

                                question.GetComponent<Image>().color = checkBoxPending ? new Color(1, 0.316f, 0.316f, 0.2235f) : Color.white;
                                if (!pending) // only set pending if it is not true already!
                                    pending = checkBoxPending;
                            }
                            break;
                        case "QTMultipleChoiceGrid":
                            if (question.GetComponent<QTMultipleChoiceGrid>().answerRequired)
                            {
                                var grid = question.transform.GetChild(1);
                                for (var c = 0; c < grid.childCount; c++)
                                {
                                    var currChild = grid.GetChild(c);
                                    if (currChild.CompareTag("QTGridRowHeader"))
                                    {
                                        var test = currChild.GetComponent<ToggleGroup>().ActiveToggles().FirstOrDefault().gameObject;
                                    }
                                }
                            }
                            question.GetComponent<Image>().color = Color.white;
                            break;
                    }
                }
                catch (Exception)
                {
                    question.GetComponent<Image>().color = new Color(1, 0.316f, 0.316f, 0.2235f);
                    pending = true;
                }
            }
            return pending;
        }

        /// <summary>
        /// Shows the current questionnaire. Can be used to show the questionnaire after it was paused. The answers already made in the current run will remain.
        /// </summary>
        public void ShowQuestionnaire()
        {
            foreach (var page in questionPages)
            {
                page.SetActive(false);
            }
            
            questionPages[_currentPage].SetActive(true);
            
            if(displayMode == DisplayMode.VR && running)
                background.SetActive(true);
        }

        /// <summary>
        /// Hides the current questionnaire. Can be used to pause a questionnaire run. The answers already made in the current run will remain.
        /// </summary>
        public void HideQuestionnaire()
        {
            foreach (var page in questionPages)
            {
                page.SetActive(false);
            }
            
            if(displayMode == DisplayMode.VR && running)
                background.SetActive(false);
        }

#if UNITY_EDITOR
        /// <summary>
        /// Changes the canvas settings of all pages if the display mode was changed.
        /// </summary>
        private void ChangeRendering()
        {
            if (spawnObjectsInWorldSpace)
            {
                foreach (var page in questionPages)
                {
                    page.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
                    var rectTransform = page.GetComponent<RectTransform>();
                    rectTransform.anchoredPosition3D = Vector3.zero;
                    rectTransform.pivot = new Vector2(0.5f, 0.5f);
                    rectTransform.localScale = Vector3.one;
                }
                
                if (!useCustomTransform)
                {
                    transform.position = new Vector3(0, 0, distanceToCamera);
                    transform.localScale = new Vector3(pageScale * pageScaleFactor, pageScale * pageScaleFactor, pageScale * pageScaleFactor);
                }
            }
            else
            {
                foreach (var page in questionPages)
                {
                    page.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
                    var rectTransform = page.GetComponent<RectTransform>();
                    rectTransform.position = Vector3.zero;
                    rectTransform.sizeDelta = new Vector2(_pageWidth, pageHeight);
                }
                
                if (!useCustomTransform)
                {
                    transform.position = Vector3.zero;
                    transform.localScale = Vector3.one;
                }
            }
            SetDynamicHeight(displayMode == DisplayMode.Desktop || displayMode == DisplayMode.Mobile);
        }

        private void LateUpdate()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            SetDynamicHeight(displayMode == DisplayMode.Desktop || displayMode == DisplayMode.Mobile);
        }

        /// <summary>
        /// Set the height of each page dynamically to fit its content.
        /// </summary>
        private void SetDynamicHeight(bool isDesktop)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            foreach (var page in questionPages)
            {
                if (dynamicHeight)
                {
                    // if dynamic height is true then the ScrollRect is not needed as the content always fits the vertical space
                    page.transform.GetChild(0).GetComponent<ScrollRect>().enabled = page.GetComponent<QTQuestionPageManager>().questionItems.Count == 0; // set only false if page is not empty
                    page.transform.GetChild(0).GetChild(0).GetComponent<RectTransform>().sizeDelta = new Vector2(0, 0);
                    page.transform.GetChild(0).GetChild(1).GetComponent<RectTransform>().sizeDelta = new Vector2(0, 0);
                    RecalculateHeight(page, isDesktop);
                }
                else
                {
                    // if dynamic height is false (re-)enable the scroll rect because it might be needed
                    page.transform.GetChild(0).GetComponent<ScrollRect>().enabled = true;
                    page.transform.GetChild(0).GetChild(1).GetComponent<RectTransform>().sizeDelta = new Vector2(20, 0);
                    page.GetComponent<RectTransform>().sizeDelta = orientation == Orientation.Horizontal ? new Vector2(1600, pageHeight) : new Vector2(1000, pageHeight);
                    
                    // update the reference res because in screenspace mode the size is controlled by the canvas scaler
                    if (isDesktop)
                        page.GetComponent<CanvasScaler>().referenceResolution = new Vector2(1600f, pageHeight);
                }
            }
        }

        /// <summary>
        /// Recalculate the required space of a question page.
        /// </summary>
        private void RecalculateHeight(GameObject page, bool isDesktop = false)
        {
            if (!dynamicHeight) return;

            // calculate the sum of all question item heights for each page
            var requiredHeight = 0f;
            foreach (var item in page.GetComponent<QTQuestionPageManager>().questionItems)
            {
                requiredHeight += item.GetComponent<RectTransform>().sizeDelta.y + 8f;
            }
            requiredHeight = requiredHeight == 0f ? 900 : requiredHeight + 52;
            page.GetComponent<RectTransform>().sizeDelta = new Vector2(orientation == Orientation.Horizontal ? 1600f : 1000f, requiredHeight);
            
            // update the reference res because in screenspace mode the size is controlled by the canvas scaler
            if (isDesktop)
            {
                requiredHeight = requiredHeight < 900 ? 900 : requiredHeight;
                page.GetComponent<CanvasScaler>().referenceResolution = new Vector2(1600f, requiredHeight);
            }
        }

        /// <summary>
        /// Sets the width and height if each page according to the current orientation.
        /// </summary>
        private void SetPageOrientation()
        {
            foreach (var page in questionPages)
            {
                switch (orientation)
                {
                    case Orientation.Horizontal:
                        page.GetComponent<RectTransform>().sizeDelta = new Vector2(1600, dynamicHeight ? page.GetComponent<RectTransform>().sizeDelta.y : pageHeight);
                        break;
                    case Orientation.Vertical:
                        page.GetComponent<RectTransform>().sizeDelta = new Vector2(1000, dynamicHeight ? page.GetComponent<RectTransform>().sizeDelta.y : pageHeight);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            SetDynamicHeight(displayMode == DisplayMode.Desktop || displayMode == DisplayMode.Mobile);
        }

        /// <summary>
        /// Resizes each item panel to fit the current selected orientation.
        /// </summary>
        private void ResizeItemPanels()
        {
            foreach (var page in questionPages)
            {
                switch (orientation)
                {
                    case Orientation.Horizontal:
                        page.transform.GetChild(0).GetChild(0).GetChild(0).GetComponent<ContentSizeFitter>().horizontalFit =
                            ContentSizeFitter.FitMode.PreferredSize;
                        break;
                    case Orientation.Vertical:
                        page.transform.GetChild(0).GetChild(0).GetChild(0).GetComponent<ContentSizeFitter>().horizontalFit =
                            ContentSizeFitter.FitMode.MinSize;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }

                foreach (var question in page.GetComponent<QTQuestionPageManager>().questionItems)
                {
                    switch (question.tag)
                    {
                        case "QTLinearScale":
                            var parentWidth = orientation == Orientation.Vertical ? 950 : 1400;
                            var options = question.GetComponent<QTLinearScale>().options;
                            foreach (var option in options)
                            {
                                option.GetComponent<LayoutElement>().minWidth = parentWidth / options.Count;
                            }
                            break;
                        case "QTCheckboxesGrid":
                            question.GetComponent<QTCheckboxesGrid>().ResizeGrid(orientation == Orientation.Vertical ? 928 : 1378);
                            break;
                        case "QTMultipleChoiceGrid":
                            question.GetComponent<QTMultipleChoiceGrid>().ResizeGrid(orientation == Orientation.Vertical ? 928 : 1378);
                            break;
                    }
                    
                }
            }
        }
#endif

        /// <summary>
        /// Sets the current main camera as the event camera of each canvas.
        /// </summary>
        private void SetEventCamera()
        {
            foreach (var page in questionPages)
            {
                page.GetComponent<Canvas>().worldCamera = Camera.main;
            }
        }

#if UNITY_EDITOR
        /// <summary>
        /// Changes the overall colors according to the selected color scheme. Takes the enum index of the desired color scheme as input.
        /// </summary>
        public void ChangeColorScheme(int i)
        {
            switch (i)
            {
                case 0:
                    pageBackgroundColor = new Color(0.9245f, 0.9245f, 0.9245f, 1); 
                    pageBottomColor = new Color(0.76f, 0.88f, 1, 1);
                    highlightColor = new Color(0, 0.67f, 1, 1);
                    break;
                case 1: 
                    pageBackgroundColor = new Color(1, 0.83f, 0.83f, 1); 
                    pageBottomColor = new Color(1, 0.5f, 0.5f, 1);
                    highlightColor = new Color(1, 0.27f, 0.24f, 1);
                    break;
                case 2:
                    pageBackgroundColor = new Color(0.83f, 1, 0.83f, 1); 
                    pageBottomColor = new Color(0.5f, 1, 0.5f, 1);
                    highlightColor = new Color(0.67f, 1, 0, 1);
                    break;
                case 3:
                    pageBackgroundColor = new Color(0.83f, 0.83f, 1, 1); 
                    pageBottomColor = new Color(0.4f, 0.5f, 1, 1);
                    highlightColor = new Color(0, 0.67f, 1, 1);
                    break;
                case 4:
                    // do nothing, as system uses inspector color fields
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            pageBackgroundColor.a = 1 - sliderValue * 0.01f;

            try // in case question page is not recognized early enough at startup
            {
                foreach (var page in questionPages)
                {
                    page.GetComponent<Image>().color = pageBackgroundColor;
                    page.transform.GetChild(1).GetComponent<Image>().color = pageBottomColor;
                    page.transform.GetChild(6).GetComponent<Image>().color = pageBottomColor;
                }
            }
            catch (Exception) { }

            var highlighters = GameObject.FindGameObjectsWithTag("QTSelectionHighlight");
            foreach (var highlighter in highlighters)
            {
                highlighter.GetComponent<Image>().color = highlightColor;
            }
        }
#endif

        /// <summary>
        /// Automatically add a counter to a header item name if the original name is already used.
        /// </summary>
        private string TryHeaderName(string toTry)
        {
            if (!resultsHeaderItems.Contains(toTry))
            {
                return toTry;
            }
            
            var alreadyExists = true;
            var resultString = "";
            var currTry = 1;
            while (alreadyExists)
            {
                if (!resultsHeaderItems.Contains(toTry + "_" + currTry))
                {
                    resultString = toTry + "_" + currTry;
                    alreadyExists = false;
                }
                currTry++;
            }
            return resultString;
        }
        
        private string TryResultFileName(string toTry)
        {
            if (!ContainsName(toTry, 2))
            {
                return toTry;
            }

            var alreadyExists = true;
            var resultString = "";
            var currTry = 1;
            while (alreadyExists)
            {
                if (!ContainsName(toTry + "_" + currTry, 1))
                {
                    resultString = toTry + "_" + currTry;
                    alreadyExists = false;
                }
                currTry++;
            }
            return resultString;
        }

        private bool ContainsName(string toTest, int maxCount)
        {
            var count = 0;
            foreach (var q in _qtManager.questionnaires)
            {
                if (!q.resultsFileName.Equals(toTest)) continue;
                count++;
                if(count == maxCount) return true;
            }
            return false;
        }
        
#if UNITY_EDITOR
        /// <summary>
        /// Add a define symbol to the scriptingDefineSymbols in the project settings if the VR device type was changed.
        /// </summary>
        private void AddDefineIfNeeded(string define, string[] toRemove)
        {
            // Get selected target defines.
            BuildTargetGroup buildTargetGroup = EditorUserBuildSettings.selectedBuildTargetGroup;
            string defines = PlayerSettings.GetScriptingDefineSymbolsForGroup(buildTargetGroup);
            
            foreach (var d in toRemove)
            {
                defines = defines.Replace(";" + d, "");
            }
            
            // Only if not defined already.
            if (define.Equals("Other") || defines.Contains(define))
            {
                // Overwrite defines
                PlayerSettings.SetScriptingDefineSymbolsForGroup(buildTargetGroup, (defines));
            }
            else if (define.Equals("HTC_Vive"))
            {
                if (File.Exists("Assets/HTC.UnityPlugin/Pointer3D/RaycastMethod/CanvasRaycastTarget.cs"))
                {
                    // Append.
                    PlayerSettings.SetScriptingDefineSymbolsForGroup(buildTargetGroup, (defines + ";" + define));
                }
                else
                {
                    Debug.LogWarning("<b>Assets/HTC.UnityPlugin/Pointer3D/RaycastMethod/CanvasRaycastTarget.cs</b> could not be found!" +
                                     "\nPlease import <b>Vive Input Utility</b>. When you fixed the problem, reselect <b>HTC_Vive</b> from the dropdown.");
                }
            }
            else if (define.Equals("Oculus"))
            {
                if (File.Exists("Assets/Oculus/VR/Scripts/Util/OVRRaycaster.cs"))
                {
                    // Append.
                    PlayerSettings.SetScriptingDefineSymbolsForGroup(buildTargetGroup, (defines + ";" + define));
                }
                else
                {
                    Debug.LogWarning("<b>Assets/Oculus/VR/Scripts/Util/OVRRaycaster.cs</b> could not be found!" +
                                     "\nPlease import <b>Oculus Integration</b> from the Asset Store. When you fixed the problem, reselect <b>Oculus</b> from the dropdown.");
                }
            }
        }
#endif

        /// <summary>
        /// Runs the VR device library dependant code to convert the standard unity canvases to work in the context of a VR raycast interaction.
        /// </summary>
        private void SetupVRSettings()
        {
#if HTC_Vive
            try
            {
                foreach (var page in questionPages)
                {
                    var vivePointers = GameObject.Find("VivePointers").transform;
                    vivePointers.GetChild(0).GetChild(0).GetChild(0).GetComponent<ViveRaycaster>()
                        .dragThreshold = 0.1f; // right controller
                    vivePointers.GetChild(1).GetChild(0).GetChild(0).GetComponent<ViveRaycaster>()
                        .dragThreshold = 0.1f; // left controller
                    page.GetComponent<GraphicRaycaster>().enabled = false;
                    page.AddComponent<CanvasRaycastTarget>();
                    foreach (var item in page.GetComponent<QTQuestionPageManager>().questionItems)
                    {
                        if (item.CompareTag("QTDropdown"))
                        {
                            item.transform.GetChild(1).GetChild(2).gameObject.AddComponent<CanvasRaycastTarget>();
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("Could not setup device setting for the specified VR device! (Vive)");
            }
#endif
#if Oculus
           try
            {
                foreach (var page in questionPages)
                {
                    page.GetComponent<GraphicRaycaster>().enabled = false;
                    page.AddComponent<OVRRaycaster>().blockingObjects = OVRRaycaster.BlockingObjects.All;
                    foreach (var item in page.GetComponent<QTQuestionPageManager>().questionItems)
                    {
                        if (item.CompareTag("QTDropdown"))
                        {
                            item.transform.GetChild(1).GetChild(2).gameObject.AddComponent<OVRRaycaster>();
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("Could not setup device setting for the specified VR device! (Oculus)");
            }
#endif
        }
        
        private static string LoadMetaData()
        {
            if (File.Exists(Application.persistentDataPath + "/QTMetaData.txt"))
            {
                return File.ReadAllText(Application.persistentDataPath + "/QTMetaData.txt", Encoding.UTF8);
            }
            
            SaveMetaData(0, 0);
            return "0;0";
        }

        private static void SaveMetaData(int r_id, int run)
        {
            File.WriteAllText(Application.persistentDataPath + "/QTMetaData.txt", r_id + ";" + run, Encoding.UTF8);
        }
        
        private static bool IsApproximate(Quaternion q1, Quaternion q2, float precision)
        {
            return Mathf.Abs(Quaternion.Dot(q1, q2)) >= 1 - precision;
        }
    }
}
