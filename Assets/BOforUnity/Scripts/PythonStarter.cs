using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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

        private void Start()
        {
            _bomanager = gameObject.GetComponent<BoForUnityManager>();

            _bomanager.loadingObj.SetActive(true);
            _bomanager.nextButton.SetActive(false);

            // Run setup async, then start Python process only after pip finished
            StartCoroutine(SetupThenLaunchCoroutine());

#if UNITY_EDITOR
            // Subscribe to the play mode state change event
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
#endif
        }

        private IEnumerator SetupThenLaunchCoroutine()
        {
            // use either the user specified python path or find the path automatically
            if (_bomanager.getLocalPython())
            {
                if (_bomanager.getPythonPath() == "")
                {
                    Debug.LogError("No Python path found -> You must specify a Python path in the BOforUnityManager inspector's python settings!");
                    yield break;
                }
                pythonExecutable = _bomanager.getPythonPath();
            }
            else
            {
                pythonExecutable = GetPythonExecutablePath();
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

            if (_bomanager.outputText != null)
            {
                _bomanager.outputText.text = pythonInstallSucceeded
                    ? "Python dependencies ready."
                    : "Python setup incomplete. Continuing…";
            }

            // Set an environment variable to allow for multiple instances of a dynamic link library.
            Environment.SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE");

            // Determine the Python script to execute
            string moboScriptName = "mobo.py";

            // Construct the full path to the Python script based on the platform.
#if UNITY_EDITOR
            string fullPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization", moboScriptName);
#elif UNITY_STANDALONE_WIN
            string bayesianOptimizationPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization");
            string fullPath = Path.Combine(bayesianOptimizationPath, moboScriptName);
#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            string bayesianOptimizationPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization");
            string fullPath = Path.Combine(bayesianOptimizationPath, moboScriptName);
#else
            string fullPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization", moboScriptName);
#endif

            // Log the full path to the Python script.
            UnityEngine.Debug.Log("Mobo Path: " + fullPath);
            UnityEngine.Debug.Log("Mobo Exists: " + File.Exists(fullPath));

            outputFilePath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization", "output.txt");
            outputFileWriter = new StreamWriter(outputFilePath);

            // Start Python process only after pip finished
            CreateProcess(fullPath);
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
                if (_bomanager?.outputText != null)
                    _bomanager.outputText.text = "The system could not be started...\nPlease restart the application.";
                if (_bomanager != null) _bomanager.loadingObj.SetActive(false);
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

            // 4. Evaluate each candidate to determine the newest version.
            string newestPython = "";
            Version newestVersion = new Version(0, 0, 0);
            foreach (var candidate in candidates)
            {
                try
                {
                    ProcessStartInfo psiVer = new ProcessStartInfo
                    {
                        FileName = candidate,
                        Arguments = "--version",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };

                    using (Process p = Process.Start(psiVer))
                    {
                        string verOutput = p.StandardOutput.ReadToEnd();
                        if (string.IsNullOrEmpty(verOutput))
                        {
                            verOutput = p.StandardError.ReadToEnd();
                        }
                        p.WaitForExit();

                        if (!string.IsNullOrEmpty(verOutput))
                        {
                            string trimmed = verOutput.Trim();
                            if (trimmed.StartsWith("Python"))
                            {
                                string versionString = trimmed.Substring("Python".Length).Trim();
                                if (Version.TryParse(versionString, out Version ver))
                                {
                                    Debug.Log("Candidate: " + candidate + " has version: " + ver);
                                    if (ver > newestVersion)
                                    {
                                        newestVersion = ver;
                                        newestPython = candidate;
                                    }
                                }
                                else
                                {
                                    Debug.LogWarning("Unable to parse version from candidate: " + candidate + " output: " + trimmed);
                                }
                            }
                            else
                            {
                                Debug.LogWarning("Candidate output did not start with 'Python': " + candidate + " output: " + trimmed);
                            }
                        }
                        else
                        {
                            Debug.LogWarning("No version output from candidate: " + candidate);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("Error checking python version for candidate: " + candidate + " - " + ex.Message);
                }
            }

            Debug.Log("Newest Python candidate selected: " + newestPython);
            return newestPython;

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

            string newestPython = "";
            Version newestVersion = new Version(0, 0, 0);
            foreach (var candidate in candidates)
            {
                try
                {
                    ProcessStartInfo psiVer = new ProcessStartInfo
                    {
                        FileName = candidate,
                        Arguments = "--version",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };
                    using (Process p = Process.Start(psiVer))
                    {
                        string verOutput = p.StandardOutput.ReadToEnd();
                        if (string.IsNullOrEmpty(verOutput))
                        {
                            verOutput = p.StandardError.ReadToEnd();
                        }
                        p.WaitForExit();
                        if (!string.IsNullOrEmpty(verOutput))
                        {
                            string trimmed = verOutput.Trim();
                            if (trimmed.StartsWith("Python"))
                            {
                                string versionString = trimmed.Substring("Python".Length).Trim();
                                if (Version.TryParse(versionString, out Version ver))
                                {
                                    if (ver > newestVersion)
                                    {
                                        newestVersion = ver;
                                        newestPython = candidate;
                                    }
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("Error checking python version for candidate: " + candidate + " - " + ex.Message);
                }
            }
            return newestPython;

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

            string newestPython = "";
            Version newestVersion = new Version(0, 0, 0);
            foreach (var candidate in candidates)
            {
                try
                {
                    ProcessStartInfo psiVer = new ProcessStartInfo
                    {
                        FileName = candidate,
                        Arguments = "--version",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };
                    using (Process p = Process.Start(psiVer))
                    {
                        string verOutput = p.StandardOutput.ReadToEnd();
                        if (string.IsNullOrEmpty(verOutput))
                        {
                            verOutput = p.StandardError.ReadToEnd();
                        }
                        p.WaitForExit();
                        if (!string.IsNullOrEmpty(verOutput))
                        {
                            string trimmed = verOutput.Trim();
                            if (trimmed.StartsWith("Python"))
                            {
                                string versionString = trimmed.Substring("Python".Length).Trim();
                                if (Version.TryParse(versionString, out Version ver))
                                {
                                    if (ver > newestVersion)
                                    {
                                        newestVersion = ver;
                                        newestPython = candidate;
                                    }
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("Error checking python version for candidate: " + candidate + " - " + ex.Message);
                }
            }
            return !string.IsNullOrEmpty(newestPython) ? newestPython : "python3";
#else
            return "python";
#endif
        }

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
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = fileName,
                    Arguments = arguments,
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

        private static string TrimMultiline(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            s = s.Trim();
            const int max = 500;
            return s.Length <= max ? s : s.Substring(0, max) + " …";
        }
    }
}