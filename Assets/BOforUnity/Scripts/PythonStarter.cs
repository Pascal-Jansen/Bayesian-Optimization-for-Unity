using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Threading;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;
using Debug = UnityEngine.Debug;


// This class manages the interaction with a Python script, which is responsible for System initialization
// and communication during a Unity application. It handles the launching of the Python process, monitors its
// status, and updates the user interface based on the System's state. Additionally, it facilitates scene
// transitions based on the application's configuration.
namespace BOforUnity.Scripts
{
    public class PythonStarter : MonoBehaviour
    {
        private string pythonExecutable;
        private Process pythonProcess;

        public bool isPythonProcessRunning;
        public bool isSystemStarted = false;
        
        private string outputFilePath;  
        private StreamWriter outputFileWriter;

        //private GameObject initSampleObj;

        private BoForUnityManager _bomanager;

        private bool _exitMessageShown = false;

        /*
        private void Awake()
        {
            
        }*/

        private void Start()
        {
            _bomanager = gameObject.GetComponent<BoForUnityManager>();
            
            if(_bomanager.getLocalPython() == true)
            {
                pythonExecutable = _bomanager.getPythonPath();
            }
            else
            {
                pythonExecutable = GetPythonExecutablePath();
            }

            Debug.Log("Python Executable Path: " + pythonExecutable);
            Debug.Log("Python Executable Exists: " + File.Exists(GetPythonExecutablePath()));

            _bomanager.loadingObj.SetActive(true);
            _bomanager.nextButton.SetActive(false);
            
            //initSampleObj = GameObject.Find("InitialSampleB");

            // Set an environment variable to allow for multiple instances of a dynamic link library.
            Environment.SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE");

            // Determine the Python script to execute based on the group and iteration configuration.
            string moboScriptName = "mobo.py";

            // Construct the full path to the Python script based on the platform.
#if UNITY_EDITOR
            //Debug.Log(Application.streamingAssetsPath);
            string fullPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization", moboScriptName);
#elif UNITY_STANDALONE_WIN
        //string fullPath = Path.Combine(GetapplicationPath(), "BayesianOptimization", "mobo.py");
        string bayesianOptimizationPath = Path.Combine(Application.streamingAssetsPath, BOData", "BayesianOptimization");
        string fullPath = Path.Combine(bayesianOptimizationPath, moboScriptName);
#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
Debug.Log(Application.streamingAssetsPath);
        string bayesianOptimizationPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization");
        string fullPath = Path.Combine(bayesianOptimizationPath, moboScriptName);
#endif
            
            // Log the full path to the Python script.
            UnityEngine.Debug.Log("Mobo Path: " + fullPath);
            UnityEngine.Debug.Log("Mobo Exists: " + File.Exists(fullPath));

            outputFilePath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization", "output.txt");
            
            // Create a StreamWriter for the output file.
            outputFileWriter = new StreamWriter(outputFilePath);
            
            // Create a Python process running the mobo.py
            CreateProcess(fullPath);
            
#if UNITY_EDITOR
            // Subscribe to the play mode state change event
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
#endif
        }

        private void Update()
        {
            if (pythonProcess != null && pythonProcess.HasExited && !_exitMessageShown)
            {
                _exitMessageShown = true;
                Debug.Log(">>>>> Python Process has EXITED!");
                if (_bomanager.simulationRunning) // if the simulation is still running show an error message
                {
                    _bomanager.outputText.text = "The system could not be started...\nPlease restart the application.";
                    _bomanager.loadingObj.SetActive(false);
                }
            }
        }

        private void CreateProcess(string fullPath)
        {
            StartCoroutine(RestartPythonProcessCoroutine(fullPath));
        }

        private IEnumerator RestartPythonProcessCoroutine(string fullPath)
        {
            yield return new WaitForSeconds(4f); // Ensure any existing process is stopped before starting a new one

            pythonProcess = new Process();
            pythonProcess.StartInfo.FileName = pythonExecutable;
            pythonProcess.StartInfo.Arguments = fullPath;
            pythonProcess.StartInfo.WorkingDirectory = GetApplicationPath();
            pythonProcess.StartInfo.UseShellExecute = false;
            pythonProcess.StartInfo.CreateNoWindow = true;
            pythonProcess.StartInfo.RedirectStandardOutput = true;
            pythonProcess.StartInfo.RedirectStandardError = true;
            pythonProcess.EnableRaisingEvents = true;

            pythonProcess.OutputDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    outputFileWriter.WriteLine(e.Data);
                    outputFileWriter.Flush();
                    Debug.LogWarning("Python Output: " + e.Data);

                    if (e.Data.IndexOf("Server starts, waiting for connection...", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        isSystemStarted = true;
                    }
                }
            };
            pythonProcess.ErrorDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    Debug.LogError("Python Error: " + e.Data);
                }
                /*
                if (e.Data == "OSError: [Errno 48] Address already in use")
                {
                    
                }*/
            };
            pythonProcess.Exited += (sender, args) => Debug.LogWarning("Python process exited with code: " + pythonProcess.ExitCode);

            try
            {
                pythonProcess.Start();
                isPythonProcessRunning = true;
                pythonProcess.BeginOutputReadLine();
                pythonProcess.BeginErrorReadLine();
                Debug.Log("Python process started successfully.");
            }
            catch (Exception ex)
            {
                Debug.Log("Failed to start Python process: " + ex.Message);
                isPythonProcessRunning = false;
                _bomanager.outputText.text = "The system could not be started...\nPlease restart the application.";
            }
        }

        public void StopPythonProcess()
        {
            if (pythonProcess != null)
            {
                if (!pythonProcess.HasExited)
                {
                    pythonProcess.Kill();
                    pythonProcess.WaitForExit();
                }

                pythonProcess.Dispose();
                pythonProcess = null;
            }
        }

        private void OnDestroy()
        {
            StopPythonProcess();
            if (outputFileWriter != null)
            {
                outputFileWriter.Close();
            }

    #if UNITY_EDITOR
            // Unsubscribe from the play mode state change event
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
    #endif
        }

        private void OnApplicationQuit()
        {
            StopPythonProcess();
            if (outputFileWriter != null)
            {
                outputFileWriter.Close();
            }
        }

    #if UNITY_EDITOR
        private void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingPlayMode || state == PlayModeStateChange.ExitingEditMode)
            {
                StopPythonProcess();
            }
        }
    #endif


        // Get the application path based on the current platform.
        private string GetApplicationPath()
        {
            string applicationPath = "";
#if UNITY_EDITOR
            applicationPath = Path.Combine(Application.dataPath, "StreamingAssets", "BOData");
#elif UNITY_STANDALONE_WIN
            applicationPath = Path.Combine(Application.dataPath, "StreamingAssets", "BOData");
#elif UNITY_STANDALONE_OSX
            applicationPath = Path.Combine(Application.dataPath, "StreamingAssets", "BOData");
#endif
            return applicationPath;
        }
        
        // Get the path to the Python executable based on the current platform.
        private string GetPythonExecutablePath()
        {
            string pythonPath = "";

#if UNITY_STANDALONE_WIN
            pythonPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Python311", "python.exe");
#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            pythonPath = "/usr/local/bin/python3";
#endif

            if (File.Exists(pythonPath))
            {
                return pythonPath;
            }

            return "";
        }

    }
}



