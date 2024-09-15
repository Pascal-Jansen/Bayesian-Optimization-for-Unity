using System;
using System.Collections.Generic;
using System.Linq;
using BOforUnity.Scripts;
using QuestionnaireToolkit.Scripts;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions.Must;
using UnityEngine.SceneManagement;
using UnityEngine.Serialization;
using UnityEngine.UI;
using PythonStarter = BOforUnity.Scripts.PythonStarter;

namespace BOforUnity
{
    public class BoForUnityManager : MonoBehaviour
    {
        public PythonStarter pythonStarter;
        public Optimizer optimizer;
        public MainThreadDispatcher mainThreadDispatcher;
        public SocketNetwork socketNetwork;

        public SceneAsset optimizerScene;
        public SceneAsset simulationScene;
        public SceneAsset questionnaireScene;
        public SceneAsset finalScene;
        
        private static BoForUnityManager _instance;
        
        //-----------------------------------------------
        // DESIGN PARAMETERS and DESIGN OBJECTIVES
        public List<ParameterEntry> parameters = new List<ParameterEntry>();
        public List<ObjectiveEntry> objectives = new List<ObjectiveEntry>();
        //-----------------------------------------------
        
        //-----------------------------------------------
        // ITERATION CONTROLLER
        [SerializeField]
        public int currentIteration;  // Current iteration value.
        public int totalIterations;
        public bool perfectRating;   // Flag indicating perfect rating.
        public bool perfectRatingStart;  // Flag indicating the start of perfect rating.
        public int perfectRatingIteration;
        public bool initialized = false;
        public bool simulationRunning = false;
        private bool _waitingForPythonProcess = false;
        
        // BO Hyper-parameters
        public int batchSize = 1;
        public int numRestarts = 10;
        public int rawSamples = 1024;
        public int nIterations = 10;
        public int mcSamples = 512;
        public int nInitial = 5; // in typical MOBO problems, this should be 2(d+1), where d is the number of objectives
        public int seed = 3;
        
        public bool warmStart = false;
        public bool perfectRatingInInitialRounds = false;
        public string initialParametersDataPath;
        public string initialObjectivesDataPath;

        public string userId = "-1";
        public string conditionId = "-1";
        public string groupId = "-1";
        //-----------------------------------------------
        
        //-----------------------------------------------
        private void Awake()
        {
            // If there is already an instance of this object, destroy the new one
            if (_instance != null)
            {
                Destroy(gameObject);
                return;
            }
            // Mark this object as the single instance and make it persistent
            _instance = this;
            DontDestroyOnLoad(gameObject);
            
            pythonStarter = gameObject.GetComponent<PythonStarter>();
            optimizer = gameObject.GetComponent<Optimizer>();
            mainThreadDispatcher = gameObject.GetComponent<MainThreadDispatcher>();
            socketNetwork = gameObject.GetComponent<SocketNetwork>();

            currentIteration = 1; // starting to count at 1
            totalIterations = nInitial + nIterations; // set how many iterations the optimizer should run for
        }
        
        void Start()
        {
            loadingObj.SetActive(true);
            nextButton.SetActive(false);

            initialized = false;
            _waitingForPythonProcess = true;
            perfectRating = false;
            perfectRatingStart = false;
            simulationRunning = true; // the simulation to true to prevent 
        }
        
        void Update()
        {
            if (_waitingForPythonProcess && pythonStarter.isPythonProcessRunning && pythonStarter.isSystemStarted)
            {
                SwitchLoading(false, true);
                _waitingForPythonProcess = false;
            }
        }
        //-----------------------------------------------
        
        
        // CONTROLLER SCENE
        //-----------------------------------------------
        public TMP_Text outputText;
        public GameObject loadingObj;
        public GameObject nextButton;

        public GameObject welcomePanel;
        
        public bool optimizationRunning = false;
        public bool optimizationFinished = false;

        //Starts a new iteration
        public void ButtonNextIteration()
        {
            SwitchLoading(true, false);

            switch (_waitingForPythonProcess)
            {
                // Do nothing if Python process did not yet started
                case true:
                    return;
                // Check if the button should trigger the initialization
                case false when !initialized:
                    // initialize the Socket to receive the first parameter values
                    PythonInitializationDone();
                    // load the optimizing scene after the button in the welcome scene was pressed
                    SceneManager.LoadSceneAsync(optimizerScene.name);
                    return;
            }

            var isPerfect = false;
            // Check if perfect rating can be achieved in the initial rounds (also known as: "sampling phase")
            if(perfectRatingInInitialRounds == true)
            {
                isPerfect = IsPerfectRating();
            }
            // Else wait until the initial rounds are over and then check for perfect rating
            else if (perfectRatingInInitialRounds == false && currentIteration > nInitial)
            {
                isPerfect = IsPerfectRating();
            }
            
            // Check if there should be another iteration of the optimization
            if (currentIteration <= totalIterations && !isPerfect)
            {
                Debug.Log("--------------------------------------Current Iteration: " + currentIteration);

                simulationRunning = true; // waiting for the simulation to finish
                
                SceneManager.LoadSceneAsync(simulationScene.name); // reload scene
            }
            else if (currentIteration > totalIterations || isPerfect)
            {
                if (isPerfect)
                {
                    Debug.Log(">>>>> Perfect Rating");
                }
                
                Debug.Log("<<<<<<< Exiting the loop ... ");
                Debug.Log("------------------------------------------------");

                simulationRunning = false; // the simulation must be finished
                
                socketNetwork.SocketQuit();

                SceneManager.LoadSceneAsync(finalScene.name);
                
                SetInfoText("The simulation has finished!\nYou can now close the application.");
                SwitchLoading(false, false);
            }
        }
        
        public void OptimizationStart()
        {
            Debug.Log("Optimization START");
            socketNetwork.SendObjectives(); // send the current objective values to the Python process
            optimizationRunning = true;
            simulationRunning = false;

            SwitchLoading(true, false);
            SetInfoText("The system is loading, please wait ...");

            SceneManager.LoadSceneAsync(optimizerScene.name); // change scene to the loading scene
        }
        
        public void OptimizationDone()
        {
            Debug.Log("Optimization DONE");
            //optimizer.UpdateDesignParameters(); // apply the parameter value of the current iteration
            optimizationRunning = false;
            
            SwitchLoading(false, true);
            SetInfoText("The system has finished loading.\nYou can now proceed.");
            
            currentIteration++; // increase iteration counter
        }
        
        public void InitializationDone()
        {
            Debug.Log("Initialization DONE");
            //optimizer.UpdateDesignParameters(); // apply the parameter value of the current iteration
            optimizationRunning = false;
            
            initialized = true;
            SwitchLoading(false, true);
            SetInfoText("The system has been started successfully!\nYou can now start the study.");
        }
        
        private void PythonInitializationDone()
        {
            Debug.Log("Python Process Initialization DONE");
            // Initialize the optimizer and socket connection ... only for Debug
            // optimizer.DebugOptimizer();
            // Start Optimization to receive the initialized parameter values for the first iteration
            userId = welcomePanel.transform.GetChild(0).GetChild(3).GetChild(1).GetComponent<TMP_InputField>().text;
            socketNetwork.InitSocket();
        }
        
        private bool IsPerfectRating()
        {
            foreach (var ob in objectives)
            {
                if (ob.value.values.Count == 0)
                {
                    return false;
                }
                
                if (ob.value.smallerIsBetter ?
                        ob.value.values.Average() > ob.value.lowerBound : 
                        ob.value.values.Average() < ob.value.upperBound)
                {
                    return false; // the rating is imperfect!
                }
            }
            
            Debug.Log("Could be perfect rating ...");
            
            switch (perfectRatingStart)
            {
                case false:
                    perfectRatingStart = true;
                    perfectRating = false;
                    perfectRatingIteration = currentIteration; // remember the current iteration for this perfect rating
                    break;
                case true when currentIteration - perfectRatingIteration == 1:
                    Debug.Log("It is a perfect rating (i.e., perfect two times in a row)!");
                    perfectRatingStart = false;
                    perfectRating = true; // the rating was perfect after two consecutive iterations
                    return true;
                default:
                    perfectRatingStart = false; // the perfect rating was more than one iteration ago
                    perfectRating = false;
                    break;
            }
            return false;
        }
        
        private void SetInfoText(string infoText)
        {
            try
            {
                outputText.text = infoText;
            }
            catch (Exception e)
            {
                // ignored
            }
        }
        
        private void SwitchLoading(bool loadingActive, bool buttonActive)
        {
            try
            {
                loadingObj.SetActive(loadingActive);
                nextButton.SetActive(buttonActive);
            }
            catch (Exception e)
            {
                // ignored
            }
        }
        
        public void EndApplication()
        {
#if UNITY_EDITOR
            EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
        }
        //-----------------------------------------------
        
        
        //-----------------------------------------------
        // SETTINGS SCRIPT

        //[Header("Server Client Communication for LogFiles")]
        //public bool serverClientCommunication;
        //public string downloadURLGroupID = "https://barakuda.de/longitudinal_uploads/GroupID_Database.csv";
        //public string downloadURLlongitudinalID = "https://barakuda.de/longitudinal_uploads/LongitudinalID_Database.csv";
        //public string lockFileURL = "https://barakuda.de/longitudinal_uploads/S_Lock_file.lock";
        //public string uploadURL = "https://barakuda.de/longitudinal_save_csv.php";
        //public string groupDatabaseName = "GroupID_Database.csv";
        //public string longitudinalDatabaseName = "LongitudinalID_Database.csv";
        //public string lockFileName = "S_Lock_file.lock";

        [Header("Location of Python executable")]
        public bool localPython;
        public string pythonPath;
       /* 
        public bool getServerClientCommunication() { return serverClientCommunication; }
        public void setServerClientCommunication(bool a) { serverClientCommunication = a; }
        
        public string getDownloadURLGroupID() { return downloadURLGroupID; }
        public void setDownloadURLGroupID(string a) { downloadURLGroupID = a; }
        
        public string getDownloadURLLongitudinalID() { return downloadURLlongitudinalID; }
        public void setDownloadURLLongitudinalID(string a) { downloadURLlongitudinalID = a; }

        public string getLockFileUrl() { return lockFileURL; }
        public void setLockFileUrl(string a) { lockFileURL = a; }

        public string getUploadURL() { return uploadURL; }
        public void setUploadURL(string a) { uploadURL = a; }

        public string getGroupDatabaseName() { return groupDatabaseName; }
        public void setGroupDatabaseName(string a) { groupDatabaseName = a; }
        
        public string getLongitudinalDatabaseName() { return longitudinalDatabaseName; }
        public void setLongitudinalDatabaseName(string a) { longitudinalDatabaseName = a; }

        public string getLockFileName() { return lockFileName; }
        public void setLockFileName(string a) { lockFileName = a; }
        */

       //-----------------------------------------------
    }
    
            // ------------------
        // the objective entries:
        // ------------------
        [System.Serializable]
        public class ObjectiveEntry
        {
            public string key;
            public ObjectiveArgs value;
            public ObjectiveEntry(string key, ObjectiveArgs value)
            {
                this.key = key;
                this.value = value;
            }
        }
        
        [System.Serializable]
        public class ObjectiveArgs
        {
            /// <summary>
            /// optSeqOrder: an integer that represents the order of this objective in a sequence of objectives
            /// values: a List of floats that stores the values obtained for this objective in a sequence of trials.
            /// lowerBound: a float that represents the lower bound of the acceptable range of values for this objective.
            /// upperBound: a float that represents the upper bound of the acceptable range of values for this objective
            /// smallerIsBetter: a bool that specifies whether a smaller value is considered better for this objective.
            /// hasMultipleValues: a bool that specifies whether this objective should have multiple values.
            /// </summary>
            [HideInInspector] public int optSeqOrder;
            public int numberOfSubMeasures;
            public List<float> values = new List<float>();
            public float lowerBound = 0.0f;
            public float upperBound = 0.0f;
            public bool smallerIsBetter = false;

            /// <summary>
            /// ObjectiveArgs(): a constructor that creates an empty instance of the ObjectiveArgs class.
            /// </summary>
            public ObjectiveArgs() { }

            /// <summary>
            /// ObjectiveArgs(lowerBound, upperBound, smallerIsBetter): a constructor that creates an instance of the
            /// ObjectiveArgs class and sets the lower and upper bounds of the acceptable range of values, as well as the
            /// smallerIsBetter flag.
            /// </summary>
            /// <param name="lowerBound"></param>
            /// <param name="upperBound"></param>
            /// <param name="smallerIsBetter"></param>
            /// <param name="numberOfSubMeasures"></param>
            public ObjectiveArgs(float lowerBound, float upperBound, bool smallerIsBetter, int numberOfSubMeasures)
            {
                this.lowerBound = lowerBound;
                this.upperBound = upperBound;
                this.smallerIsBetter = smallerIsBetter;
                this.numberOfSubMeasures = numberOfSubMeasures;
            }

            /// <summary>
            /// GetInitInfoStr(): a method that returns a string representing the initial configuration of this objective, including
            /// the lower and upper bounds and the smallerIsBetter flag.
            /// </summary>
            /// <returns></returns>
            public string GetInitInfoStr()
            {
                return $"{lowerBound},{upperBound},{(smallerIsBetter ? 1 : 0)}/";
            }
        }
        // ------------------
        
        
        // ------------------
        // the parameter entries:
        // ------------------
        [System.Serializable]
        public class ParameterEntry
        {
            public string key;
            public ParameterArgs value;
            public ParameterEntry(string key, ParameterArgs value)
            {
                this.key = key;
                this.value = value;
            }
        }
        
        [System.Serializable]
        public class ParameterArgs
        {
            /// <summary>
            /// optSeqOrder: an integer that represents the order of this parameter in a sequence of parameters.
            /// isDiscrete: a bool that specifies whether the parameter takes discrete (quantized) values.
            /// lowerBound: a float that represents the lower bound of the acceptable range of values for this parameter.
            /// upperBound: a float that represents the upper bound of the acceptable range of values for this parameter.
            /// step: a float that represents the increment between two consecutive values for a discrete parameter
            /// Value: a float that represents the current value of this parameter.
            /// reference: a float reference that can be used to keep track of the previous value of this parameter, if needed.
            /// </summary>
            [HideInInspector] public int optSeqOrder;
            public bool isDiscrete = false;
            public float lowerBound = 0.0f;
            public float upperBound = 0.0f;
            public float step = 1.0f;
            public float Value = 0.0f;

            // Reference to another GameObject's script
            public MonoBehaviour scriptReference;
            public string variableName;
            public string gameObjectName;
            public string scriptName;
            
            /// <summary>
            /// ParameterArgs(): a constructor that creates an empty instance of the ParameterArgs class.
            /// </summary>
            public ParameterArgs() { }

            /// <summary>
            /// ParameterArgs(lowerBound, upperBound): a constructor that creates an instance of the ParameterArgs class and sets
            /// the lower and upper bounds of the acceptable range of values for this parameter.
            /// </summary>
            /// <param name="lowerBound"></param>
            /// <param name="upperBound"></param>
            public ParameterArgs(float lowerBound, float upperBound)
            {
                this.lowerBound = lowerBound;
                this.upperBound = upperBound;
            }

            /// <summary>
            /// ParameterArgs(lowerBound, upperBound, step): a constructor that creates an instance of the ParameterArgs class and sets
            /// the lower and upper bounds of the acceptable range of values for this parameter, as well as the step size for a discrete
            /// parameter.
            /// </summary>
            /// <param name="lowerBound"></param>
            /// <param name="upperBound"></param>
            /// <param name="step"></param>
            public ParameterArgs(float lowerBound, float upperBound, float step)
            {
                this.lowerBound = lowerBound;
                this.upperBound = upperBound;
                this.step = step;
                isDiscrete = true;
            }

            /// <summary>
            /// GetInitInfoStr(): a method that returns a string representing the initial configuration of this parameter, including the lower
            /// and upper bounds and the step size.
            /// </summary>
            /// <returns></returns>
            public string GetInitInfoStr()
            {
                return $"{lowerBound},{upperBound},{step}/";
            }
        }
}