using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TMPro;
#if UNITY_EDITOR
using UnityEditor;
#endif
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

        private BoForUnityManager _bomanager;

        private bool _exitMessageShown = false;

        // ── Python dependency install status (shown in UI while running) ─────
        [Header("Python Install Status")]
        public string pythonInstallStatus = "Idle";
        public bool pythonInstallRunning = false;
        public bool pythonInstallSucceeded = false;

        // Python compatibility policy for automatic selection and setup.
        private const int SupportedPythonMajor = 3;
        private const int MinSupportedPythonMinor = 9;
        private const int BundledPythonMinor = 13;
        private const string BundledPythonVersionLabel = "3.13.7";

        private void Start()
        {
            _bomanager = gameObject.GetComponent<BoForUnityManager>();
            if (_bomanager == null)
            {
                Debug.LogError("PythonStarter requires a BoForUnityManager component on the same GameObject.");
                enabled = false;
                return;
            }

            if (_bomanager.loadingObj != null) _bomanager.loadingObj.SetActive(true);
            if (_bomanager.nextButton != null) _bomanager.nextButton.SetActive(false);

            // Run setup async, then start Python process only after pip finished
            StartCoroutine(SetupThenLaunchCoroutine());

#if UNITY_EDITOR
            // Subscribe to the play mode state change event
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
#endif
        }

        private IEnumerator SetupThenLaunchCoroutine()
        {
            if (_bomanager == null)
            {
                Debug.LogError("PythonStarter setup aborted: BoForUnityManager is missing.");
                yield break;
            }

            // use either the user specified python path or find the path automatically
            if (_bomanager.getLocalPython())
            {
                if (_bomanager.getPythonPath() == "")
                {
                    Debug.LogError("No Python path found -> You must specify a Python path in the BOforUnityManager inspector's python settings!");
                    _bomanager.simulationRunning = false;
                    if (_bomanager.outputText != null)
                    {
                        _bomanager.outputText.text =
                            "No Python executable path is configured.\nSet it in BoForUnityManager and restart.";
                    }
                    if (_bomanager.loadingObj != null)
                        _bomanager.loadingObj.SetActive(false);
                    yield break;
                }
                pythonExecutable = _bomanager.getPythonPath();
            }
            else
            {
                pythonExecutable = GetPythonExecutablePath();

                // If auto mode cannot find a supported Python, install the bundled fallback runtime once.
                if (!TryGetPythonVersion(pythonExecutable, out Version autoDetectedVersion, out _) ||
                    !IsSupportedPythonVersion(autoDetectedVersion))
                {
                    string detectedDescription = string.IsNullOrWhiteSpace(pythonExecutable)
                        ? "No Python executable was found"
                        : autoDetectedVersion != null
                            ? $"Detected Python {autoDetectedVersion}"
                            : $"Could not detect Python version at {pythonExecutable}";
                    pythonInstallStatus =
                        $"{detectedDescription}. Installing bundled Python {BundledPythonVersionLabel} " +
                        "(may require admin confirmation)...";

                    var bundledInstallTask = TryInstallBundledPythonAsync();
                    while (!bundledInstallTask.IsCompleted)
                    {
                        if (_bomanager.outputText != null)
                            _bomanager.outputText.text = pythonInstallStatus;
                        yield return null;
                    }

                    if (bundledInstallTask.Result)
                    {
                        pythonExecutable = GetPythonExecutablePath();
                        Debug.Log("Bundled Python installation succeeded. New executable path: " + pythonExecutable);
                    }
                    else
                    {
                        Debug.LogError(
                            "Bundled Python installation did not complete successfully. " +
                            "Aborting startup to avoid running with an unintended interpreter."
                        );
                        if (_bomanager != null)
                        {
                            _bomanager.simulationRunning = false;
                            if (_bomanager.outputText != null)
                            {
                                _bomanager.outputText.text =
                                    "Python installation was cancelled or failed.\n" +
                                    $"Please install Python {SupportedPythonMajor}.{MinSupportedPythonMinor} or newer and restart.";
                            }
                            if (_bomanager.loadingObj != null)
                                _bomanager.loadingObj.SetActive(false);
                        }
                        yield break;
                    }
                }
            }

            Debug.Log("Python Executable Path: " + pythonExecutable);
            Debug.Log("Python Executable Exists: " + (!string.IsNullOrEmpty(pythonExecutable) && File.Exists(pythonExecutable)));

            // Show status in the UI while installing
            pythonInstallStatus = "Preparing Python environment…";
            pythonInstallRunning = true;

            // Install requirements on a background thread and wait
            var installTask = InstallRequirementsForPythonAsync(pythonExecutable);
            while (!installTask.IsCompleted)
            {
                // Mirror status to UI if available
                if (_bomanager.outputText != null)
                    _bomanager.outputText.text = pythonInstallStatus;
                yield return null;
            }
            pythonInstallSucceeded = installTask.Result;
            pythonInstallRunning = false;

            if (!pythonInstallSucceeded)
            {
                Debug.LogError("Python dependency setup failed: " + pythonInstallStatus);
                if (_bomanager != null)
                {
                    _bomanager.simulationRunning = false;
                    if (_bomanager.outputText != null)
                        _bomanager.outputText.text =
                            "Python setup failed.\nCheck Console output and requirements.txt, then restart.";
                    if (_bomanager.loadingObj != null)
                        _bomanager.loadingObj.SetActive(false);
                }
                yield break;
            }

            if (_bomanager.outputText != null)
                _bomanager.outputText.text = "Python dependencies ready.";

            // Set an environment variable to allow for multiple instances of a dynamic link library.
            Environment.SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE");

            int effectiveParameterCount = CountDistinctValidParameterKeys(_bomanager?.parameters);
            int effectiveObjectiveCount = CountDistinctValidObjectiveKeys(_bomanager?.objectives);
            if (effectiveParameterCount < 1 || effectiveObjectiveCount < 1)
            {
                Debug.LogError(
                    $"Invalid optimization configuration. Effective parameters={effectiveParameterCount}, " +
                    $"effective objectives={effectiveObjectiveCount}. At least one of each is required."
                );
                if (_bomanager != null)
                {
                    _bomanager.simulationRunning = false;
                    if (_bomanager.outputText != null)
                    {
                        _bomanager.outputText.text =
                            "Invalid optimization configuration.\nEnsure at least one valid parameter and objective key are set.";
                    }
                    if (_bomanager.loadingObj != null)
                        _bomanager.loadingObj.SetActive(false);
                }
                yield break;
            }

            var parameterKeys = GetDistinctValidParameterKeys(_bomanager?.parameters);
            var objectiveKeys = GetDistinctValidObjectiveKeys(_bomanager?.objectives);
            var overlap = parameterKeys
                .Intersect(objectiveKeys, StringComparer.OrdinalIgnoreCase)
                .OrderBy(k => k, StringComparer.OrdinalIgnoreCase)
                .ToList();
            if (overlap.Count > 0)
            {
                Debug.LogError(
                    "Python startup aborted because parameter and objective keys overlap: " +
                    string.Join(", ", overlap)
                );
                if (_bomanager != null)
                {
                    _bomanager.simulationRunning = false;
                    if (_bomanager.outputText != null)
                    {
                        _bomanager.outputText.text =
                            "Invalid BO configuration.\n" +
                            "Parameter and objective keys must be distinct.";
                    }
                    if (_bomanager.loadingObj != null)
                        _bomanager.loadingObj.SetActive(false);
                }
                yield break;
            }

            int configuredParameterCount = _bomanager?.parameters?.Count ?? 0;
            int configuredObjectiveCount = _bomanager?.objectives?.Count ?? 0;
            if (configuredParameterCount != effectiveParameterCount || configuredObjectiveCount != effectiveObjectiveCount)
            {
                Debug.LogError(
                    "Python startup aborted due to invalid/duplicate parameter or objective entries. " +
                    $"Configured counts: parameters={configuredParameterCount}, objectives={configuredObjectiveCount}; " +
                    $"effective counts: parameters={effectiveParameterCount}, objectives={effectiveObjectiveCount}."
                );
                if (_bomanager != null)
                {
                    _bomanager.simulationRunning = false;
                    if (_bomanager.outputText != null)
                    {
                        _bomanager.outputText.text =
                            "Invalid BO configuration.\n" +
                            "Remove duplicate or empty parameter/objective keys and restart.";
                    }
                    if (_bomanager.loadingObj != null)
                        _bomanager.loadingObj.SetActive(false);
                }
                yield break;
            }

            // Determine the Python script to execute based on backend + objective mode.
            string optimizerScriptName;
            if (!TryResolveOptimizerScriptName(_bomanager, effectiveObjectiveCount, out optimizerScriptName, out string scriptError))
            {
                Debug.LogError("Python startup aborted: " + scriptError);
                if (_bomanager != null)
                {
                    _bomanager.simulationRunning = false;
                    if (_bomanager.outputText != null)
                        _bomanager.outputText.text = "Invalid optimizer settings.\n" + scriptError;
                    if (_bomanager.loadingObj != null)
                        _bomanager.loadingObj.SetActive(false);
                }
                yield break;
            }

            // Construct the full path to the Python script based on the platform.
#if UNITY_EDITOR
            string fullPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization", optimizerScriptName);
#elif UNITY_STANDALONE_WIN
            string bayesianOptimizationPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization");
            string fullPath = Path.Combine(bayesianOptimizationPath, optimizerScriptName);
#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            string bayesianOptimizationPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization");
            string fullPath = Path.Combine(bayesianOptimizationPath, optimizerScriptName);
#else
            string fullPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization", optimizerScriptName);
#endif

            // Log the full path to the Python script.
            Debug.Log("Optimizer script path: " + fullPath);
            Debug.Log("Optimizer script exists: " + File.Exists(fullPath));

            outputFilePath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization", "output.txt");
            outputFileWriter = new StreamWriter(outputFilePath);

            // Start Python process only after pip finished
            CreateProcess(fullPath);
        }

        private static int CountDistinctValidParameterKeys(IList<BOforUnity.ParameterEntry> parameters)
        {
            if (parameters == null)
                return 0;

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < parameters.Count; i++)
            {
                var parameter = parameters[i];
                if (parameter == null || parameter.value == null || string.IsNullOrWhiteSpace(parameter.key))
                    continue;

                seen.Add(parameter.key.Trim());
            }
            return seen.Count;
        }

        private static HashSet<string> GetDistinctValidParameterKeys(IList<BOforUnity.ParameterEntry> parameters)
        {
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            if (parameters == null)
                return seen;

            for (int i = 0; i < parameters.Count; i++)
            {
                var parameter = parameters[i];
                if (parameter == null || parameter.value == null || string.IsNullOrWhiteSpace(parameter.key))
                    continue;

                seen.Add(parameter.key.Trim());
            }

            return seen;
        }

        private static int CountDistinctValidObjectiveKeys(IList<BOforUnity.ObjectiveEntry> objectives)
        {
            if (objectives == null)
                return 0;

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < objectives.Count; i++)
            {
                var objective = objectives[i];
                if (objective == null || objective.value == null || string.IsNullOrWhiteSpace(objective.key))
                    continue;

                seen.Add(objective.key.Trim());
            }
            return seen.Count;
        }

        private static HashSet<string> GetDistinctValidObjectiveKeys(IList<BOforUnity.ObjectiveEntry> objectives)
        {
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            if (objectives == null)
                return seen;

            for (int i = 0; i < objectives.Count; i++)
            {
                var objective = objectives[i];
                if (objective == null || objective.value == null || string.IsNullOrWhiteSpace(objective.key))
                    continue;

                seen.Add(objective.key.Trim());
            }

            return seen;
        }

        private static bool TryResolveOptimizerScriptName(
            BOforUnity.BoForUnityManager manager,
            int effectiveObjectiveCount,
            out string scriptName,
            out string error)
        {
            scriptName = null;
            error = null;

            if (manager == null)
            {
                error = "BoForUnityManager is missing.";
                return false;
            }

            if (manager.optimizerBackend == BOforUnity.BoForUnityManager.OptimizerBackend.BoTorch)
            {
                scriptName = effectiveObjectiveCount > 1 ? "mobo.py" : "bo.py";
                return true;
            }

            if (manager.optimizerBackend == BOforUnity.BoForUnityManager.OptimizerBackend.CABOP)
            {
                if (manager.cabopObjectiveMode == BOforUnity.BoForUnityManager.CabopObjectiveMode.SingleObjective)
                {
                    if (effectiveObjectiveCount != 1)
                    {
                        error =
                            "CABOP SingleObjective mode requires exactly one configured objective.";
                        return false;
                    }

                    scriptName = "cabop_bo.py";
                    return true;
                }

                if (manager.cabopObjectiveMode == BOforUnity.BoForUnityManager.CabopObjectiveMode.MultiObjectiveScalarized)
                {
                    if (effectiveObjectiveCount < 2)
                    {
                        error =
                            "CABOP MultiObjectiveScalarized mode requires at least two configured objectives.";
                        return false;
                    }

                    scriptName = "cabop_mobo.py";
                    return true;
                }
            }

            error = "Unsupported optimizer backend/objective mode combination.";
            return false;
        }

        private void Update()
        {
            // Live status during install
            if (pythonInstallRunning && _bomanager != null && _bomanager.outputText != null)
            {
                _bomanager.outputText.text = pythonInstallStatus;
            }

            if (pythonProcess != null && pythonProcess.HasExited && !_exitMessageShown)
            {
                _exitMessageShown = true;
                Debug.Log(">>>>> Python Process has EXITED!");
                if (_bomanager != null && _bomanager.simulationRunning) // if the simulation is still running show an error message
                {
                    if (_bomanager.outputText != null)
                        _bomanager.outputText.text = "The system could not be started...\nPlease restart the application.";
                    if (_bomanager.loadingObj != null)
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
            yield return new WaitForSeconds(0.25f); // small delay

            pythonProcess = new Process();
            pythonProcess.StartInfo.FileName = pythonExecutable;
            pythonProcess.StartInfo.Arguments = $"\"{fullPath}\"";
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
            };
            pythonProcess.Exited += (sender, args) =>
            {
                isPythonProcessRunning = false;
                isSystemStarted = false;
                Debug.LogWarning("Python process exited with code: " + pythonProcess.ExitCode);
            };

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
                isSystemStarted = false;
                if (_bomanager?.outputText != null)
                    _bomanager.outputText.text = "The system could not be started...\nPlease restart the application.";
                if (_bomanager?.loadingObj != null) _bomanager.loadingObj.SetActive(false);
            }
        }

        public void StopPythonProcess()
        {
            if (pythonProcess != null)
            {
                try
                {
                    if (!pythonProcess.HasExited)
                    {
                        pythonProcess.Kill();
                        pythonProcess.WaitForExit();
                    }
                }
                catch { /* ignore */ }

                pythonProcess.Dispose();
                pythonProcess = null;
            }

            isPythonProcessRunning = false;
            isSystemStarted = false;
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
#else
            applicationPath = Path.Combine(Application.dataPath, "StreamingAssets", "BOData");
#endif
            return applicationPath;
        }

        /// <summary>
        /// Finds and returns the path to the newest installed Python executable.
        /// </summary>
        /// <returns>Full path to the Python executable, or an empty string if not found.</returns>
        private string GetPythonExecutablePath()
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            var candidates = new List<string>();

            // 1. Get Python executables from the PATH using the 'where' command.
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = "where",
                    Arguments = "python",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                };

                using (Process proc = Process.Start(psi))
                {
                    string output = proc.StandardOutput.ReadToEnd();
                    proc.WaitForExit();
                    Debug.Log("Output of 'where python': " + output);

                    string[] lines = output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        if (line.IndexOf("WindowsApps", StringComparison.OrdinalIgnoreCase) >= 0)
                        {
                            Debug.Log("Skipping candidate from WindowsApps: " + line);
                            continue;
                        }
                        if (File.Exists(line))
                        {
                            if (!candidates.Contains(line))
                            {
                                candidates.Add(line);
                                Debug.Log("Added candidate from PATH: " + line);
                            }
                        }
                        else
                        {
                            Debug.Log("Candidate from PATH does not exist: " + line);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Error finding python using 'where': " + ex.Message);
            }

            // 2. Search the Local Programs folder.
            string localProgramsPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs");
            Debug.Log("Checking Local Programs directory: " + localProgramsPath);
            if (Directory.Exists(localProgramsPath))
            {
                try
                {
                    string[] pythonDirs = Directory.GetDirectories(localProgramsPath, "Python*", SearchOption.TopDirectoryOnly);
                    foreach (var dir in pythonDirs)
                    {
                        string candidate = Path.Combine(dir, "python.exe");
                        if (File.Exists(candidate) && !candidates.Contains(candidate))
                        {
                            candidates.Add(candidate);
                            Debug.Log("Added candidate from Local Programs: " + candidate);
                        }
                        else
                        {
                            string[] subdirs = Directory.GetDirectories(dir, "*", SearchOption.TopDirectoryOnly);
                            foreach (var subdir in subdirs)
                            {
                                candidate = Path.Combine(subdir, "python.exe");
                                if (File.Exists(candidate) && !candidates.Contains(candidate))
                                {
                                    candidates.Add(candidate);
                                    Debug.Log("Added candidate from Local Programs subdir: " + candidate);
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("Error searching Local Programs directory: " + ex.Message);
                }
            }
            else
            {
                Debug.Log("Local Programs directory not found: " + localProgramsPath);
            }

            // 3. Search the Program Files directory.
            string programFilesPath = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            Debug.Log("Checking Program Files directory: " + programFilesPath);
            if (Directory.Exists(programFilesPath))
            {
                try
                {
                    string[] pythonDirs = Directory.GetDirectories(programFilesPath, "Python*", SearchOption.TopDirectoryOnly);
                    foreach (var dir in pythonDirs)
                    {
                        string candidate = Path.Combine(dir, "python.exe");
                        if (File.Exists(candidate) && !candidates.Contains(candidate))
                        {
                            candidates.Add(candidate);
                            Debug.Log("Added candidate from Program Files: " + candidate);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("Error searching Program Files directory: " + ex.Message);
                }
            }
            else
            {
                Debug.Log("Program Files directory not found: " + programFilesPath);
            }

            Debug.Log("Total candidates found: " + candidates.Count);

            return SelectNewestSupportedPythonCandidate(candidates, "");

#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            // macOS
            List<string> candidates = new List<string>();
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = "which",
                    Arguments = "-a python3",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                };
                using (Process proc = Process.Start(psi))
                {
                    string output = proc.StandardOutput.ReadToEnd();
                    proc.WaitForExit();
                    string[] lines = output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        if (File.Exists(line))
                        {
                            candidates.Add(line);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Error finding python using 'which': " + ex.Message);
            }

            return SelectNewestSupportedPythonCandidate(candidates, "");

#elif UNITY_STANDALONE_LINUX
            // Linux
            List<string> candidates = new List<string>();
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = "which",
                    Arguments = "-a python3",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                };
                using (Process proc = Process.Start(psi))
                {
                    string output = proc.StandardOutput.ReadToEnd();
                    proc.WaitForExit();
                    string[] lines = output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        if (!string.IsNullOrEmpty(line) && File.Exists(line))
                        {
                            candidates.Add(line);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Error finding python using 'which': " + ex.Message);
            }

            return SelectNewestSupportedPythonCandidate(candidates, "python3");
#else
            return "python";
#endif
        }

        private string SelectNewestSupportedPythonCandidate(IEnumerable<string> candidates, string fallback)
        {
            string newestSupportedPython = "";
            Version newestSupportedVersion = new Version(0, 0, 0);
            string newestAnyPython = "";
            Version newestAnyVersion = new Version(0, 0, 0);

            foreach (string candidate in candidates.Where(c => !string.IsNullOrWhiteSpace(c)).Distinct())
            {
                if (!TryGetPythonVersion(candidate, out Version version, out string rawOutput))
                {
                    Debug.LogWarning("Unable to parse Python version from candidate: " + candidate +
                                     " output: " + rawOutput?.Trim());
                    continue;
                }

                Debug.Log("Candidate: " + candidate + " has version: " + version);

                if (version > newestAnyVersion)
                {
                    newestAnyVersion = version;
                    newestAnyPython = candidate;
                }

                if (IsSupportedPythonVersion(version) && version > newestSupportedVersion)
                {
                    newestSupportedVersion = version;
                    newestSupportedPython = candidate;
                }
            }

            if (!string.IsNullOrEmpty(newestSupportedPython))
            {
                Debug.Log("Newest supported Python candidate selected: " + newestSupportedPython);
                return newestSupportedPython;
            }

            if (!string.IsNullOrEmpty(newestAnyPython))
            {
                Debug.LogWarning("No supported Python candidate was found. Newest installed candidate is unsupported: " +
                                 newestAnyPython + " (" + newestAnyVersion + ")");
                return newestAnyPython;
            }

            Debug.LogWarning("No Python candidate was found. Falling back to: " + fallback);
            return fallback;
        }

        private static bool IsSupportedPythonVersion(Version version)
        {
            return version != null &&
                   version.Major == SupportedPythonMajor &&
                   version.Minor >= MinSupportedPythonMinor;
        }

        private static string NormalizePythonVersionToken(string token)
        {
            if (string.IsNullOrWhiteSpace(token))
                return string.Empty;

            string t = token.Trim();
            int plus = t.IndexOf('+');
            if (plus >= 0)
                t = t.Substring(0, plus);
            return t;
        }

        private bool TryGetPythonVersion(string pythonPath, out Version version, out string rawOutput)
        {
            version = null;
            rawOutput = null;

            if (string.IsNullOrWhiteSpace(pythonPath))
                return false;

            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments = "--version",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using (var p = Process.Start(psi))
                {
                    string stdout = p.StandardOutput.ReadToEnd();
                    string stderr = p.StandardError.ReadToEnd();
                    p.WaitForExit();

                    rawOutput = string.IsNullOrWhiteSpace(stdout) ? stderr : stdout;
                    if (string.IsNullOrWhiteSpace(rawOutput))
                        return false;

                    string trimmed = rawOutput.Trim();
                    if (!trimmed.StartsWith("Python", StringComparison.OrdinalIgnoreCase))
                        return false;

                    string afterPrefix = trimmed.Substring("Python".Length).Trim();
                    string[] parts = afterPrefix.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 0)
                        return false;

                    string versionToken = NormalizePythonVersionToken(parts[0]);
                    return Version.TryParse(versionToken, out version);
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Error checking Python version: " + ex.Message);
                return false;
            }
        }

        private static bool TryGetBundledPythonPath(out string bundledPath)
        {
            bundledPath = null;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            bundledPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                $"Python{SupportedPythonMajor}{BundledPythonMinor}",
                "python.exe"
            );
#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            bundledPath = $"/Library/Frameworks/Python.framework/Versions/{SupportedPythonMajor}.{BundledPythonMinor}/bin/python3";
#elif UNITY_STANDALONE_LINUX
            bundledPath = $"/usr/bin/python{SupportedPythonMajor}.{BundledPythonMinor}";
#endif
            return !string.IsNullOrWhiteSpace(bundledPath) && File.Exists(bundledPath);
        }

        private Task<bool> TryInstallBundledPythonAsync()
        {
            return Task.Run(() =>
            {
                try
                {
#if UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
                    string pkgPath = Path.Combine(
                        Application.streamingAssetsPath,
                        "BOData",
                        "Installation",
                        "MacOs",
                        "Data",
                        "Installation_Objects",
                        "python-3.13.7-macos11.pkg"
                    );
                    if (!File.Exists(pkgPath))
                    {
                        pythonInstallStatus = $"Bundled Python installer package not found: {pkgPath}";
                        return false;
                    }

                    pythonInstallStatus =
                        $"Installing bundled Python {BundledPythonVersionLabel} from packaged installer " +
                        "(macOS admin prompt may appear)...";
                    int rc = RunMacPkgInstallerWithAdminPrompt(pkgPath);
                    if (rc != 0)
                    {
                        pythonInstallStatus = $"Bundled Python installation failed ({rc}).";
                        return false;
                    }

                    // Verify target path now exists.
                    if (!TryGetBundledPythonPath(out string bundledPath))
                    {
                        pythonInstallStatus = "Bundled Python install completed, but executable path was not found.";
                        return false;
                    }

                    if (!TryGetPythonVersion(bundledPath, out Version bundledVersion, out _)
                        || !IsSupportedPythonVersion(bundledVersion))
                    {
                        pythonInstallStatus = "Bundled Python path exists, but version check failed.";
                        return false;
                    }

                    pythonInstallStatus = $"Bundled Python {bundledVersion} installed.";
                    return true;
#elif UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
                    string installBat = Path.Combine(
                        Application.streamingAssetsPath,
                        "BOData",
                        "Installation",
                        "Windows",
                        "installation_python.bat"
                    );
                    if (!File.Exists(installBat))
                    {
                        pythonInstallStatus = $"Bundled Python installer not found: {installBat}";
                        return false;
                    }

                    // May trigger UAC depending on system policy and current permissions.
                    pythonInstallStatus = $"Installing bundled Python {BundledPythonVersionLabel} (Windows UAC prompt may appear)...";
                    int rc = RunProcessBlocking("cmd.exe", $"/c \"\"{installBat}\"\"");
                    if (rc != 0)
                    {
                        pythonInstallStatus = $"Bundled Python installation failed ({rc}).";
                        return false;
                    }
                    pythonInstallStatus = "Bundled Python installer finished.";
                    return true;
#elif UNITY_STANDALONE_LINUX
                    pythonInstallStatus =
                        "Automatic bundled Python installation is not supported from Unity on Linux. " +
                        "Run Assets/StreamingAssets/BOData/Installation/Linux/install_python.sh manually.";
                    return false;
#else
                    pythonInstallStatus = "Automatic bundled Python installation is not supported on this platform.";
                    return false;
#endif
                }
                catch (Exception ex)
                {
                    pythonInstallStatus = $"Bundled Python install error: {ex.Message}";
                    return false;
                }
            });
        }

#if UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
        private int RunMacPkgInstallerWithAdminPrompt(string pkgPath)
        {
            string tempPkgPath = Path.Combine(Path.GetTempPath(), $"bo4unity_python_{Guid.NewGuid():N}.pkg");
            string tempScriptPath = Path.Combine(Path.GetTempPath(), $"bo4unity_install_{Guid.NewGuid():N}.applescript");
            try
            {
                // Copy installer payload into a neutral temp location to avoid Desktop/Documents access restrictions.
                File.Copy(pkgPath, tempPkgPath, overwrite: true);
                string escapedTempPkgPath = tempPkgPath.Replace("\\", "\\\\").Replace("\"", "\\\"");
                string applescript =
                    $"do shell script \"cd / && /usr/sbin/installer -pkg \\\"{escapedTempPkgPath}\\\" -target /\" with administrator privileges";
                File.WriteAllText(tempScriptPath, applescript);
                return RunProcessBlocking("/usr/bin/osascript", $"\"{tempScriptPath}\"", "/");
            }
            finally
            {
                try
                {
                    if (File.Exists(tempScriptPath))
                        File.Delete(tempScriptPath);
                }
                catch
                {
                    // Best effort cleanup for temporary AppleScript file.
                }
                try
                {
                    if (File.Exists(tempPkgPath))
                        File.Delete(tempPkgPath);
                }
                catch
                {
                    // Best effort cleanup for copied installer payload.
                }
            }
        }
#endif

        // ── Async: install requirements with the given Python on a worker thread ──
        private Task<bool> InstallRequirementsForPythonAsync(string pythonPath)
        {
            return Task.Run(() =>
            {
                try
                {
                    if (string.IsNullOrEmpty(pythonPath))
                    {
                        pythonInstallStatus = "Install skipped: no Python found.";
                        return false;
                    }

                    if (!TryGetPythonVersion(pythonPath, out Version pyVersion, out string rawVersion))
                    {
                        pythonInstallStatus = "Could not detect Python version.";
                        return false;
                    }
                    if (!IsSupportedPythonVersion(pyVersion))
                    {
                        pythonInstallStatus =
                            $"Unsupported Python version {pyVersion}. " +
                            $"Supported: {SupportedPythonMajor}.{MinSupportedPythonMinor}+.";
                        return false;
                    }

                    Debug.Log($"Using Python {pyVersion} at {pythonPath}. Raw version output: {rawVersion?.Trim()}");

                    // Build path to requirements.txt inside StreamingAssets
                    string reqPath = Path.Combine(Application.streamingAssetsPath, "BOData", "Installation", "requirements.txt");
                    if (!File.Exists(reqPath))
                    {
                        pythonInstallStatus = "requirements.txt not found. Skipping install.";
                        return false;
                    }

                    pythonInstallStatus = "Ensuring pip…";
                    int rc = RunProcessBlocking(pythonPath, "-m ensurepip --upgrade");
                    if (rc != 0)
                    {
                        pythonInstallStatus = $"ensurepip failed ({rc}).";
                        return false;
                    }

                    pythonInstallStatus = "Upgrading pip…";
                    rc = RunProcessBlocking(pythonPath, "-m pip install --upgrade pip");
                    if (rc != 0)
                    {
                        pythonInstallStatus = $"pip upgrade failed ({rc}).";
                        return false;
                    }

                    pythonInstallStatus = "Installing Python dependencies…";
                    rc = RunProcessBlocking(pythonPath, $"-m pip install --user -r \"{reqPath}\"");
                    if (rc != 0)
                    {
                        pythonInstallStatus = $"requirements install failed ({rc}).";
                        return false;
                    }

                    pythonInstallStatus = "Dependencies installed.";
                    return true;
                }
                catch (Exception ex)
                {
                    pythonInstallStatus = $"Python setup error: {ex.Message}";
                    return false;
                }
            });
        }

        // Run a process and return exit code (blocking on the worker thread)
        private int RunProcessBlocking(string fileName, string arguments)
        {
            return RunProcessBlocking(fileName, arguments, null);
        }

        private int RunProcessBlocking(string fileName, string arguments, string workingDirectoryOverride)
        {
            try
            {
                string workingDir = string.IsNullOrWhiteSpace(workingDirectoryOverride)
                    ? GetSafeWorkingDirectory()
                    : workingDirectoryOverride;
                var psi = new ProcessStartInfo
                {
                    FileName = fileName,
                    Arguments = arguments,
                    WorkingDirectory = workingDir,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };
                psi.Environment["PIP_NO_INPUT"] = "1";
                psi.Environment["PYTHONIOENCODING"] = "utf-8";

                using (var p = Process.Start(psi))
                {
                    string stdout = p.StandardOutput.ReadToEnd();
                    string stderr = p.StandardError.ReadToEnd();
                    p.WaitForExit();

                    if (!string.IsNullOrEmpty(stdout))
                        Debug.Log(TrimMultiline($"[{Path.GetFileName(fileName)} {arguments}] {stdout}"));
                    if (!string.IsNullOrEmpty(stderr))
                        Debug.Log(TrimMultiline($"[{Path.GetFileName(fileName)} {arguments}] {stderr}"));

                    return p.ExitCode;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Error running: {fileName} {arguments}\n{ex.Message}");
                return -1;
            }
        }

        private string GetSafeWorkingDirectory()
        {
            try
            {
                string appDataPath = Application.dataPath;
                if (!string.IsNullOrWhiteSpace(appDataPath) && Directory.Exists(appDataPath))
                    return appDataPath;
            }
            catch
            {
                // Fallback below.
            }

            string userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (!string.IsNullOrWhiteSpace(userHome) && Directory.Exists(userHome))
                return userHome;

            string tempPath = Path.GetTempPath();
            if (!string.IsNullOrWhiteSpace(tempPath) && Directory.Exists(tempPath))
                return tempPath;

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            return Environment.SystemDirectory;
#else
            return "/";
#endif
        }

        private static string TrimMultiline(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            s = s.Trim();
            const int max = 500;
            return s.Length <= max ? s : s.Substring(0, max) + " …";
        }
    }
}
