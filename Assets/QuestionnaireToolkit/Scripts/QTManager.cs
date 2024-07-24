using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace QuestionnaireToolkit.Scripts
{
    [ExecuteInEditMode, DisallowMultipleComponent]
    public class QTManager : MonoBehaviour
    {
        public enum DisplayMode { Desktop, VR, Mobile }
        public enum Orientation { Horizontal, Vertical }
        public enum DeviceType { Other, Vive, Oculus }
        public enum ColorScheme { Default, Red, Green, Blue, Custom }
        
        [Tooltip("Set the display mode for all questionnaires according to the target platform.")]
        public DisplayMode displayMode = DisplayMode.Desktop;
        [Tooltip("Orientation of a question page. Horizontal aspect ratio: 16:9, Vertical aspect ratio: 10:16.")]
        public Orientation orientation = Orientation.Horizontal;
        public DeviceType deviceType = DeviceType.Other;
        public float pageHeight = 900;
        public bool dynamicHeight;
        [Tooltip("(VR only) If true, then you can set the position/rotation of a questionnaire in world space. Otherwise the distance below will be used and a questionnaire will be positioned in the center of the view.")]
        public bool useCustomTransform;
        [Tooltip("(VR only) Distance to main camera in meters.")]
        public float distanceToCamera = 1;
        [Tooltip("(VR only) Determines the size of a question page in meters.")]
        public float pageScaleFactor = 1;
        public ColorScheme colorScheme = ColorScheme.Default;
        public float sliderValue = 20;
        public bool showTopPanel = true;
        [Tooltip("If the bottom panel is deactivated a questionnaire cannot be finished! Use only for basic UI elements!")]
        public bool showBottomPanel = true;
        public bool showPageNumber = true;
        [Tooltip("Enables the ability to go back to the previous page.")]
        public bool showPrevButton;

        [Header("Questionnaire Management")]
        [SerializeField]
        public List<QTQuestionnaireManager> questionnaires = new List<QTQuestionnaireManager>();

        private const float pageScale = 0.001f;
        private const float AspectRatio = 0.5625f;
        private bool _bottomPanelState = true;
        private bool _pageNumberState = true;
        private bool _prevButtonState;
        private bool _topPanelState = true;
        
        private int _oldDisplayMode;
        private int _oldOrientation;
        private int _oldColorScheme ;
        private int _oldDeviceType;
        private bool _oldDynamicHeight;
        private bool _oldUseCustomTransform;
        private float _oldPageHeight = 900;
        private float _oldDistanceToCamera = 1.5f;
        private float _oldPageScaleFactor = 1f;
        private float _oldSliderValue = 20;
        
        private Color _oldBackgroundColor;
        private Color _oldBottomColor;
        private Color _oldHighlightColor;
        public Color pageBackgroundColor = new Color(0.9245f, 0.9245f, 0.9245f, 1);
        public Color pageBottomColor = new Color(0.76f, 0.88f, 1, 1);
        public Color highlightColor = new Color(0, 0.67f, 1, 1);

        [HideInInspector]
        public int selectedQuestionnaire ;
        private QTQuestionnaireManager _visibleQuestionnaire;

        [HideInInspector]
        public int currentRunningQuestionnaireIndex;
        private bool _inspectorChanged;

        private void Start()
        {
#if UNITY_EDITOR
            TagsAndLayers.RefreshQtTags(); // keep tag list fresh!
            
            if (!EditorApplication.isPlayingOrWillChangePlaymode)
            {
                try
                {
                    if (GameObject.FindGameObjectsWithTag("QTManager").Length > 1)
                    {
                        Debug.Log("Only one active QTManager allowed in the scene!");
                        DestroyImmediate(gameObject);
                        return;
                    }

                    if (transform.childCount == 0 && GameObject.FindGameObjectsWithTag("QTQuestionnaireManager").Length >= 1)
                    {
                        Debug.Log("QTManager cannot exist besides individual Questionnaires!");
                        DestroyImmediate(gameObject);
                        return;
                    }
                    
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
            }
#endif
        }

        
#if UNITY_EDITOR
        public void OnValidate()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;

            //EditorApplication.delayCall += () => { };
            
            // update device type config
                if (_oldDeviceType != (int) deviceType)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.deviceType = (QTQuestionnaireManager.DeviceType)(int)deviceType;
                    }
                    _oldDeviceType = (int) deviceType;
                    _inspectorChanged = true;
                }
                
                // update custom page height
                if (_oldPageHeight != pageHeight && _oldOrientation == (int) orientation)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.pageHeight = pageHeight;
                    }
                    _oldPageHeight = pageHeight;
                    _inspectorChanged = true;
                }
                
                // update distance to camera
                if (_oldDistanceToCamera != distanceToCamera)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.distanceToCamera = distanceToCamera;
                    }
                    _oldDistanceToCamera = distanceToCamera;
                    _inspectorChanged = true;
                }
                
                // update page scale factor
                if (_oldPageScaleFactor != pageScaleFactor)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.pageScaleFactor = pageScaleFactor;
                    }
                    _oldPageScaleFactor = pageScaleFactor;
                    _inspectorChanged = true;
                }
                
                // update UseCustomTransform
                if (_oldUseCustomTransform != useCustomTransform)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.useCustomTransform = useCustomTransform;
                    }
                    _oldUseCustomTransform = useCustomTransform;
                    _inspectorChanged = true;
                }
                
                // update dynamic page height
                if (_oldDynamicHeight != dynamicHeight)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.dynamicHeight = dynamicHeight;
                    }
                    _oldDynamicHeight = dynamicHeight;
                    _inspectorChanged = true;
                }

                // update display mode settings
                if (_oldDisplayMode != (int)displayMode)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.displayMode = (QTQuestionnaireManager.DisplayMode)(int)displayMode;
                    }
                    _oldDisplayMode = (int) displayMode;
                    _inspectorChanged = true;
                }

                // update page orientation
                if (_oldOrientation != (int)orientation)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.orientation = (QTQuestionnaireManager.Orientation)(int)orientation;
                    }
                    _oldOrientation = (int) orientation;
                    _inspectorChanged = true;
                }

                // update overall color scheme
                if (_oldColorScheme != (int)colorScheme)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.colorScheme = (QTQuestionnaireManager.ColorScheme)(int)colorScheme;
                    }
                    _oldColorScheme = (int) colorScheme;
                    _inspectorChanged = true;
                }

                // update custom colors
                if (_oldBackgroundColor != pageBackgroundColor || _oldBottomColor != pageBottomColor || _oldHighlightColor != highlightColor)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.pageBackgroundColor = pageBackgroundColor;
                        q.pageBottomColor = pageBottomColor;
                        q.highlightColor = highlightColor;
                    }
                    _oldBackgroundColor = pageBackgroundColor;
                    _oldBottomColor = pageBottomColor;
                    _oldHighlightColor = highlightColor;
                    _inspectorChanged = true;
                }
                
                // update page transparency slider value
                if ((int)_oldSliderValue != (int)sliderValue)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.sliderValue = sliderValue;
                    }
                    _oldSliderValue = sliderValue;
                    _inspectorChanged = true;
                }
                
                // update top panel visuals
                if (_topPanelState != showTopPanel)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.showTopPanel = showTopPanel;
                    }
                    _topPanelState = showTopPanel;
                    _inspectorChanged = true;
                }
                
                // update bottom panel visuals
                if (_bottomPanelState != showBottomPanel)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.showBottomPanel = showBottomPanel;
                    }
                    _bottomPanelState = showBottomPanel;
                    _inspectorChanged = true;
                }

                // update visibility of page number
                if (_pageNumberState != showPageNumber)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.showPageNumber = showPageNumber;
                    }
                    _pageNumberState = showPageNumber;
                    _inspectorChanged = true;
                }
                
                // update visibility of the previous button
                if (_prevButtonState != showPrevButton)
                {
                    foreach (var q in questionnaires)
                    {
                        if (q.overrideManagerSettings) continue;
                        q.showPrevButton = showPrevButton;
                    }
                    _prevButtonState = showPrevButton;
                    _inspectorChanged = true;
                }

                if (!_inspectorChanged) return;
                foreach (var q in questionnaires)
                {
                    if (q.overrideManagerSettings) continue;
                    q.OnValidate();
                }
                _inspectorChanged = false;
        }

        /// <summary>
        /// Creates a new questionnaire.
        /// </summary>
        public void CreateQuestionnaire()
        {
            // load and instantiate the questionnaire prefab
            var o = Resources.Load("QuestionnaireToolkitPrefabs/QTQuestionnaireManager");
            var g = Instantiate(o, transform) as GameObject;
            questionnaires.Add(g.GetComponent<QTQuestionnaireManager>());
            g.name = "Questionnaire-" + questionnaires.Count;
            g.GetComponent<QTQuestionnaireManager>().resultsFileName = g.name;

            var q = g.GetComponent<QTQuestionnaireManager>();
            
            q.displayMode = (QTQuestionnaireManager.DisplayMode)(int)displayMode;
            q.deviceType = (QTQuestionnaireManager.DeviceType)(int)deviceType;
            q.orientation = (QTQuestionnaireManager.Orientation)(int)orientation;
            q.colorScheme = (QTQuestionnaireManager.ColorScheme)(int)colorScheme;
            q.pageHeight = pageHeight;
            q.pageScaleFactor = pageScaleFactor;
            q.dynamicHeight = dynamicHeight;
            q.useCustomTransform = useCustomTransform;
            q.distanceToCamera = distanceToCamera;

            q.pageBackgroundColor = pageBackgroundColor;
            q.pageBottomColor = pageBottomColor;
            q.highlightColor = highlightColor;
            q.sliderValue = sliderValue;
            
            q.showBottomPanel = showBottomPanel;
            q.showPageNumber = showPageNumber;
            q.showPrevButton = showPrevButton;

            q.ChangeColorScheme((int) q.colorScheme);
            
            ShowSelectedQuestionnaire(questionnaires.Count - 1);
        }

        /// <summary>
        /// Copies the selected questionnaire including its pages.
        /// </summary>
        public void CopyQuestionnaire()
        {
            var g = Instantiate(questionnaires[selectedQuestionnaire].gameObject, transform) as GameObject;
            questionnaires.Add(g.GetComponent<QTQuestionnaireManager>());
            g.name = "Questionnaire-" + questionnaires.Count;
            g.GetComponent<QTQuestionnaireManager>().resultsFileName = g.name;
            ShowSelectedQuestionnaire(questionnaires.Count - 1);
        }

        /// <summary>
        /// Deletes an entire questionnaire including its pages.
        /// </summary>
        public void DeleteQuestionnaire(int listCount = -1, int sel = -1)
        {
            if (questionnaires.Count <= 0 || selectedQuestionnaire >= questionnaires.Count || selectedQuestionnaire < 0) return;
            if (listCount == -1 && sel == -1)
            {
                listCount = questionnaires.Count - 1;
                sel = selectedQuestionnaire;
            }
            if (listCount < questionnaires.Count) // item was removed
            {
                DestroyImmediate(questionnaires[sel].gameObject);
                questionnaires.RemoveAt(sel);
                
                if (questionnaires.Count <= 0) return;
                
                ShowSelectedQuestionnaire(questionnaires.Count - 1);
            }
        }

        /// <summary>
        /// Reorders questionnaires according to the order of the reorderable list.
        /// </summary>
        public void ReorderQuestionnaires(int listCount, int sel)
        {
            if (listCount == questionnaires.Count)
            {
                questionnaires[sel].transform.SetSiblingIndex(sel);
            }
        }

        /// <summary>
        /// Hides all questionnaires except the one that should be displayed.
        /// </summary>
        public void ShowSelectedQuestionnaire(int sel)
        {
            if (questionnaires.Count == 0) return;
            
            foreach (var q in questionnaires)
            {
                q.HideQuestionnaire();
                if (!q.name.EndsWith(" [ACTIVE]")) continue;
                q.name = q.name.Substring(0, q.name.LastIndexOf(" [ACTIVE]", StringComparison.Ordinal));
            }
            
            _visibleQuestionnaire = questionnaires[sel];
            _visibleQuestionnaire.ShowQuestionnaire();
            _visibleQuestionnaire.name += " [ACTIVE]";

            selectedQuestionnaire = sel;
        }
#endif

        public GameObject FindParentWithTag(GameObject childObject, string parentTag)
        {
            Transform t = childObject.transform;
            while (t.parent != null)
            {
                if (t.parent.CompareTag(parentTag))
                {
                    return t.parent.gameObject;
                }
                t = t.parent.transform;
            }
            return null; // Could not find a parent with given tag.
        }
    }
}
