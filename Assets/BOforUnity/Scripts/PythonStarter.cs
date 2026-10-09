using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using QuestionnaireToolkit.Scripts;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using Debug = UnityEngine.Debug;

// Prepares the Python environment the optimizer backends run in and launches the backend script.
//
// Setup (off the main thread, with timeouts):
//   1. Find an interpreter: the configured one ("Manually Installed Python"), or the best installed Python 3.13
//      (bundled 3.13 first; newer minors only after every tested one failed). When none is found, the bundled
//      Python 3.13 is installed on Windows/macOS (after checking the installer's SHA-256).
//   2. Pick where the packages go: into the configured interpreter if it already is a venv/conda environment,
//      otherwise into a private venv at persistentDataPath/BOData/python-venv (falls back to `pip install --user`
//      if a venv cannot be created, e.g. Debian without python3-venv).
//   3. Verify the pinned versions from requirements.txt on every start (importlib.metadata, no torch import);
//      run pip when they do not match or requirements.txt changed. The full pip output goes to
//      persistentDataPath/BOData/pip-install.log.
// Launch: the backend runs with that environment's interpreter; its console output goes to
// persistentDataPath/BOData/BayesianOptimization/output.txt (previous run kept as output.prev.txt) and is archived
// into the run's LogData folder at the end of the session.
namespace BOforUnity.Scripts
{
    public class PythonStarter : MonoBehaviour
    {
        // ── Public state (read by BoForUnityManager, shown in the inspector) ──────────────────
        public bool isPythonProcessRunning;
        public bool isSystemStarted = false;

        [Header("Python Install Status")]
        public string pythonInstallStatus = "Idle";
        public bool pythonInstallRunning = false;
        public bool pythonInstallSucceeded = false;

        // ── Python compatibility policy ────────────────────────────────────────────────────────
        private const int SupportedPythonMajor = 3;
        private const int MinSupportedPythonMinor = 13;
        // Newest minor version the pinned dependency stack is tested with (the CI full-stack job runs 3.13).
        // Newer interpreters are only tried after every tested one failed: the pinned wheels may not exist for them.
        private const int MaxTestedPythonMinor = 13;
        private const int BundledPythonMinor = 13;
        private const string BundledPythonVersionLabel = "3.13.7";

        // SHA-256 of the bundled installers as shipped in Installation/. Checked before they run with admin rights,
        // so a truncated copy or a Git LFS pointer file is reported instead of executed.
        private const string MacPythonPkgFile = "python-3.13.7-macos11.pkg";
        private const string MacPythonPkgSha256 = "f7e8c8d63ab0a4e736b5864aa369098b16af622042c079addb2f1a08400560c5";
        private const string WindowsPythonInstallerFile = "python-3.13.7.exe";
        private const string WindowsPythonInstallerSha256 = "b12e2e82461ac8e51fc43289050bc8eb937a32d84ce4d242e2c88258c37cf2bb";
        private const string WindowsVcRedistFile = "VC_redist.x64.exe";
        private const string WindowsVcRedistSha256 = "cc0ff0eb1dc3f5188ae6300faef32bf5beeba4bdd6e8e445a9184072096b713b";

        // PyPI's Linux torch wheels are CUDA builds that pull several GB of nvidia-* packages; GP-sized workloads run
        // on the CPU. With PyTorch's CPU index added, pip picks 2.14.1+cpu, which satisfies the "torch==2.14.1" pin.
        private const string LinuxCpuTorchIndexUrl = "https://download.pytorch.org/whl/cpu";

        private static readonly TimeSpan ProbeTimeout = TimeSpan.FromSeconds(30);
        private static readonly TimeSpan VenvTimeout = TimeSpan.FromMinutes(10);
        private static readonly TimeSpan EnsurePipTimeout = TimeSpan.FromMinutes(10);
        private static readonly TimeSpan PipInstallTimeout = TimeSpan.FromMinutes(45);
        private static readonly TimeSpan BundledInstallTimeout = TimeSpan.FromMinutes(30);
        private static readonly TimeSpan SetupTimeout = TimeSpan.FromMinutes(60);

        // Prints version, machine, whether the interpreter already is a virtual/conda environment, and the real path
        // of the executable (symlinks such as /usr/local/bin/python3.13 lead to the same interpreter).
        // Single quotes only, so it survives argument quoting on every platform.
        private const string InterpreterProbeScript =
            "import sys,os,platform;print('BO4U|%d|%d|%d|%s|%d|%s' % (sys.version_info[0],sys.version_info[1]," +
            "sys.version_info[2],platform.machine(),int(sys.prefix!=getattr(sys,'base_prefix',sys.prefix) or " +
            "os.path.isdir(os.path.join(sys.prefix,'conda-meta'))),os.path.realpath(sys.executable)))";

        // Lists the installed distributions from their metadata (no package is imported, so no torch start-up).
        // Iterated in reverse so the entry first on sys.path wins, as it does for imports. Skipped like pip skips
        // them: "~orch-…" folders an interrupted pip install/uninstall leaves behind (they keep the old version's
        // metadata and could shadow the real entry) and folders without metadata.
        private const string DistributionsProbeScript =
            "import json,re,importlib.metadata as m;print('BO4U-DISTS '+json.dumps({re.sub('[-_.]+','-'," +
            "(d.metadata.get_all('Name') or [''])[0]).lower():d.version for d in reversed(list(m.distributions())) " +
            "if getattr(getattr(d,'_path',None),'name','')[:1]!='~' and (d.read_text('METADATA') or d.read_text('PKG-INFO'))}))";

        // Explicit settings (JsonSerializer.Create ignores JsonConvert.DefaultSettings): a host project's global
        // Json.NET defaults must not change how the package list is read.
        private static readonly JsonSerializer PackageListJson = JsonSerializer.Create(new JsonSerializerSettings
        {
            DateParseHandling = DateParseHandling.None,
            MetadataPropertyHandling = MetadataPropertyHandling.Ignore,
            TypeNameHandling = TypeNameHandling.None,
        });

        private const int TailLines = 60;
        private const int MaxCapturedStdout = 1 << 20;

        // ── Instance state ─────────────────────────────────────────────────────────────────────
        private BoForUnityManager _bomanager;
        private Process pythonProcess;
        private PythonEnvironment _environment;
        private bool _started;
        private bool _shutdownDone;

        private string outputFilePath;
        private StreamWriter outputFileWriter;
        // stdout and stderr callbacks run on different worker threads.
        private readonly object _outputFileLock = new object();
        private DateTime _launchTimeUtc;
        private bool _outputArchived;

        private bool _exitMessageShown;
        private volatile bool _backendExitObserved;
        private volatile bool _deliberateStop;
        private volatile bool _backendEverStarted;
        private volatile string _lastStderrLine;
        private int _backendExitCode = int.MinValue;

        // ── Setup coordination (static: one dependency install at a time per Unity process) ──────
        private static readonly object s_setupGate = new object();
        private static Task<SetupOutcome> s_setupTask;
        private static string s_setupKey;
        private static CancellationTokenSource s_setupCts;
        private static volatile string s_status = "Idle";
        private static readonly object s_childGate = new object();
        private static readonly HashSet<Process> s_children = new HashSet<Process>();
        private static HostOs s_hostOs = HostOs.Unknown;

        private enum HostOs { Unknown, Windows, MacOS, Linux }

        private sealed class Requirement
        {
            public string Name;
            public string NormalizedName;
            public string PinnedVersion; // null: only presence is checked
        }

        private sealed class SetupContext
        {
            public HostOs Os;
            public bool ManualPython;
            public string ManualPythonPath;
            public string RequirementsPath;
            public string WheelsDir;
            public string BundledInstallerRoot;
            public string StateDir;
            public string VenvDir;
            public string PipLogPath;
            public string WorkingDirectory;
            public string Key;
            public bool UseArm64Wrapper;
            public CancellationToken Token;
            public DateTime DeadlineUtc;
            public string RequirementsText;
            public List<Requirement> Requirements;
            public string LockPath;
            public StreamWriter PipLog;
            public readonly object PipLogLock = new object();
        }

        private sealed class PythonCandidate
        {
            public string Path;
            public string Source;
            public Version Version;
            public string Machine;
            public bool IsEnvironment;
            public bool IsBundled;
            public string RealPath;
            public int Order;
        }

        private sealed class PythonEnvironment
        {
            public string Interpreter;      // what the backends run with
            public string BaseInterpreter;  // what was discovered/configured
            public Version Version;
            public string Mode;
            public bool UserSite;
            public bool Arm64Wrapper;
            public string PackageReport;
        }

        private sealed class SetupOutcome
        {
            public bool Success;
            public bool Aborted; // cancelled or out of time: do not try further interpreters
            public string Error;
            public PythonEnvironment Environment;

            public static SetupOutcome Ok(PythonEnvironment env) => new SetupOutcome { Success = true, Environment = env };
            public static SetupOutcome Fail(string error) => new SetupOutcome { Error = error };
            public static SetupOutcome Abort(string error) => new SetupOutcome { Aborted = true, Error = error };
        }

        private sealed class ChildOptions
        {
            public bool UserSite;
            public bool LogToPipLog;
            public bool TrackInLockFile;
            public Action<string> OnLine;
        }

        private sealed class ChildResult
        {
            public int ExitCode = -1;
            public bool TimedOut;
            public bool Cancelled;
            public bool FailedToStart;
            public string StartError;
            public string StdOut = string.Empty;
            public List<string> Tail = new List<string>();
            public List<string> StderrTail = new List<string>();
            public bool Succeeded => !TimedOut && !Cancelled && !FailedToStart && ExitCode == 0;
        }

        private sealed class PackageCheck
        {
            public bool Ok;
            public bool Aborted;
            public string Summary;
            public string Report;
        }

        // ── Unity lifecycle ────────────────────────────────────────────────────────────────────

        private void Start()
        {
            _bomanager = gameObject.GetComponent<BoForUnityManager>();
            if (_bomanager == null)
            {
                Debug.LogError("PythonStarter requires a BoForUnityManager component on the same GameObject.");
                enabled = false;
                return;
            }

            _started = true;
            s_hostOs = DetectHostOs();

            if (_bomanager.loadingObj != null) _bomanager.loadingObj.SetActive(true);
            if (_bomanager.nextButton != null) _bomanager.nextButton.SetActive(false);

#if UNITY_EDITOR
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
            AssemblyReloadEvents.beforeAssemblyReload += OnBeforeAssemblyReload;
#endif

            StartCoroutine(SetupThenLaunchCoroutine());
        }

        private void Update()
        {
            if (pythonInstallRunning)
            {
                pythonInstallStatus = s_status;
                if (_bomanager != null && _bomanager.outputText != null)
                    _bomanager.outputText.text = pythonInstallStatus;
            }

            if (_backendExitObserved && !_exitMessageShown)
            {
                _exitMessageShown = true;
                ReportBackendExit();
            }
        }

        private void OnDestroy()
        {
#if UNITY_EDITOR
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            AssemblyReloadEvents.beforeAssemblyReload -= OnBeforeAssemblyReload;
#endif
            // A duplicate manager discarded in Awake never started anything; it must not stop the live session.
            if (_started)
                ShutdownBackend("PythonStarter destroyed");
        }

        private void OnApplicationQuit()
        {
            if (_started)
                ShutdownBackend("application quit");
        }

#if UNITY_EDITOR
        private void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingPlayMode && _started)
                ShutdownBackend("leaving Play mode");
        }

        private void OnBeforeAssemblyReload()
        {
            if (_started && this != null)
                ShutdownBackend("script reload");
        }
#endif

        /// <summary>
        /// Ends the session: stops a running dependency setup (and its pip child), tells SocketNetwork the backend is
        /// going away on purpose, stops the backend, and archives its console log with the study data.
        /// </summary>
        private void ShutdownBackend(string reason)
        {
            if (_shutdownDone)
                return;
            _shutdownDone = true;

            CancelSetup(reason);

            var socketNetwork = GetComponent<SocketNetwork>();
            if (socketNetwork != null)
                socketNetwork.RequestStop();

            StopPythonProcess();
            CloseOutputFile();
            ArchiveOutputFile();
        }

        // ── Setup and launch ───────────────────────────────────────────────────────────────────

        private IEnumerator SetupThenLaunchCoroutine()
        {
            // Configuration errors are cheap to detect: report them before a long dependency install.
            if (!TryValidateConfiguration(out string optimizerScriptName))
                yield break;

            SetupContext ctx;
            try
            {
                ctx = CreateSetupContext();
            }
            catch (Exception ex)
            {
                Debug.LogError("Python setup could not start: " + ex);
                ShowFailure("Python setup could not start:\n" + ex.Message);
                yield break;
            }

            pythonInstallRunning = true;
            pythonInstallSucceeded = false;
            Task<SetupOutcome> setupTask = StartOrReuseSetup(ctx);
            while (!setupTask.IsCompleted)
            {
                pythonInstallStatus = s_status;
                if (_bomanager.outputText != null)
                    _bomanager.outputText.text = pythonInstallStatus;
                yield return null;
            }
            pythonInstallRunning = false;

            SetupOutcome outcome = setupTask.Status == TaskStatus.RanToCompletion
                ? setupTask.Result
                : SetupOutcome.Fail("Python setup error: " + (setupTask.Exception?.GetBaseException().Message ?? "unknown"));

            if (_shutdownDone)
                yield break;

            if (outcome == null || !outcome.Success)
            {
                string error = outcome?.Error ?? "unknown error";
                pythonInstallStatus = "Python setup failed: " + error;
                Debug.LogError("Python setup failed: " + error + "\nFull pip output (if pip ran): " + ctx.PipLogPath);
                ShowFailure("Python setup failed:\n" + Shorten(error, 400) + "\nSee the Console for details.");
                yield break;
            }

            pythonInstallSucceeded = true;
            _environment = outcome.Environment;
            pythonInstallStatus = "Python dependencies ready.";
            if (_bomanager.outputText != null)
                _bomanager.outputText.text = pythonInstallStatus;

            Debug.Log(
                $"Optimizer Python: {_environment.Interpreter} (Python {_environment.Version}, {_environment.Mode}). " +
                "Install optional packages (openbo, open_clip_torch) with this interpreter: " +
                $"\"{_environment.Interpreter}\" -m pip install …"
            );
            if (!string.IsNullOrEmpty(_environment.PackageReport))
                Debug.Log("Python packages: " + _environment.PackageReport);

            string scriptPath = Path.Combine(Application.streamingAssetsPath, "BOData", "BayesianOptimization", optimizerScriptName);
            Debug.Log("Optimizer script path: " + scriptPath + " (exists: " + File.Exists(scriptPath) + ")");

            yield return new WaitForSeconds(0.25f);
            if (_shutdownDone)
                yield break;
            LaunchBackend(scriptPath, optimizerScriptName);
        }

        private bool TryValidateConfiguration(out string optimizerScriptName)
        {
            optimizerScriptName = null;

            int effectiveParameterCount = CountDistinctValidParameterKeys(_bomanager.parameters);
            int effectiveObjectiveCount = CountDistinctValidObjectiveKeys(_bomanager.objectives);
            if (effectiveParameterCount < 1 || effectiveObjectiveCount < 1)
            {
                Debug.LogError(
                    $"Invalid optimization configuration. Effective parameters={effectiveParameterCount}, " +
                    $"effective objectives={effectiveObjectiveCount}. At least one of each is required."
                );
                ShowFailure("Invalid optimization configuration.\nEnsure at least one valid parameter and objective key are set.");
                return false;
            }

            var overlap = GetDistinctValidParameterKeys(_bomanager.parameters)
                .Intersect(GetDistinctValidObjectiveKeys(_bomanager.objectives), StringComparer.OrdinalIgnoreCase)
                .OrderBy(k => k, StringComparer.OrdinalIgnoreCase)
                .ToList();
            if (overlap.Count > 0)
            {
                Debug.LogError("Python startup aborted because parameter and objective keys overlap: " + string.Join(", ", overlap));
                ShowFailure("Invalid optimizer configuration.\nParameter and objective keys must be distinct.");
                return false;
            }

            int configuredParameterCount = _bomanager.parameters?.Count ?? 0;
            int configuredObjectiveCount = _bomanager.objectives?.Count ?? 0;
            if (configuredParameterCount != effectiveParameterCount || configuredObjectiveCount != effectiveObjectiveCount)
            {
                Debug.LogError(
                    "Python startup aborted due to invalid/duplicate parameter or objective entries. " +
                    $"Configured counts: parameters={configuredParameterCount}, objectives={configuredObjectiveCount}; " +
                    $"effective counts: parameters={effectiveParameterCount}, objectives={effectiveObjectiveCount}."
                );
                ShowFailure("Invalid optimizer configuration.\nRemove duplicate or empty parameter/objective keys and restart.");
                return false;
            }

            if (!TryValidateContextualConfiguration(_bomanager, out string contextError))
            {
                Debug.LogError("Python startup aborted: " + contextError);
                ShowFailure("Invalid contextual optimization settings.\n" + contextError);
                return false;
            }

            if (!TryResolveOptimizerScriptName(_bomanager, effectiveObjectiveCount, out optimizerScriptName, out string scriptError))
            {
                Debug.LogError("Python startup aborted: " + scriptError);
                ShowFailure("Invalid optimizer settings.\n" + scriptError);
                return false;
            }

            // The shared checks (also shown in the inspector and repeated before the init message): reserved or
            // duplicate keys, lower >= upper, raw samples < restarts, ... Refuse before Python is started at all.
            if (!BoConfigValidator.TryValidate(_bomanager, out string configError))
            {
                Debug.LogError("Python startup aborted: invalid BO configuration:\n" + configError);
                ShowFailure("Invalid optimizer configuration.\n" + configError);
                return false;
            }

            return true;
        }

        // Ends the study with the message on screen (state panel, loop stopped, auto-advance cancelled, Python stopped).
        private void ShowFailure(string message)
        {
            if (_bomanager == null)
                return;
            _bomanager.TerminateWithError(message);
        }

        // Main thread only: reads Application paths and the manager's settings.
        private SetupContext CreateSetupContext()
        {
            string persistentDataPath = Application.persistentDataPath;
            string installationDir = Path.Combine(Application.streamingAssetsPath, "BOData", "Installation");
            string wheelsDir = Path.Combine(installationDir, "wheels");
            var ctx = new SetupContext
            {
                Os = s_hostOs,
                ManualPython = _bomanager.getLocalPython(),
                ManualPythonPath = _bomanager.getPythonPath(),
                RequirementsPath = Path.Combine(installationDir, "requirements.txt"),
                WheelsDir = Directory.Exists(wheelsDir) ? wheelsDir : null,
                BundledInstallerRoot = installationDir,
                StateDir = Path.Combine(persistentDataPath, "BOData", "Installation"),
                VenvDir = ResolvePrivateVenvDir(s_hostOs, persistentDataPath),
                PipLogPath = Path.Combine(persistentDataPath, "BOData", "pip-install.log"),
                WorkingDirectory = GetSafeWorkingDirectory(),
            };
            ctx.Key = string.Join("|", ctx.ManualPython, (ctx.ManualPythonPath ?? string.Empty).Trim(), ctx.RequirementsPath, ctx.VenvDir);
            return ctx;
        }

        private static string ResolvePrivateVenvDir(HostOs os, string persistentDataPath)
        {
            string venvDir = Path.Combine(persistentDataPath, "BOData", "python-venv");
            // Windows paths are limited to 259 characters unless long paths are enabled. The deepest file in the pinned
            // stack is ~125 characters below Lib\site-packages, so a long company/product name could break pip;
            // use a short per-project folder under %LOCALAPPDATA% then.
            if (os == HostOs.Windows && venvDir.Length > 100)
            {
                string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                if (!string.IsNullOrEmpty(localAppData))
                {
                    string shortDir = Path.Combine(localAppData, "BOforUnity", "venv-" + ShortHash(persistentDataPath));
                    Debug.Log($"Private Python environment path '{venvDir}' is too long for Windows; using '{shortDir}' instead.");
                    return shortDir;
                }
            }
            return venvDir;
        }

        private static Task<SetupOutcome> StartOrReuseSetup(SetupContext ctx)
        {
            lock (s_setupGate)
            {
                if (s_setupTask != null && !s_setupTask.IsCompleted && s_setupCts != null &&
                    !s_setupCts.IsCancellationRequested && s_setupKey == ctx.Key)
                {
                    Debug.Log("A Python dependency setup is already running; waiting for it instead of starting a second pip install.");
                    return s_setupTask;
                }

                var cts = new CancellationTokenSource();
                ctx.Token = cts.Token;
                s_setupCts = cts;
                s_setupKey = ctx.Key;
                s_status = "Preparing Python environment…";

                Task previous = s_setupTask;
                s_setupTask = previous == null || previous.IsCompleted
                    ? Task.Run(() => RunSetup(ctx))
                    // A cancelled setup may still be unwinding (killing pip): never let two installs overlap.
                    : previous.ContinueWith(_ => RunSetup(ctx), CancellationToken.None,
                        TaskContinuationOptions.None, TaskScheduler.Default);
                return s_setupTask;
            }
        }

        private static void CancelSetup(string reason)
        {
            CancellationTokenSource cts;
            bool running;
            lock (s_setupGate)
            {
                cts = s_setupCts;
                running = s_setupTask != null && !s_setupTask.IsCompleted;
            }

            try { cts?.Cancel(); } catch (ObjectDisposedException) { }
            int killed = KillTrackedChildren();
            if (running || killed > 0)
                Debug.Log($"Python setup stopped ({reason}); terminated {killed} setup process(es).");
        }

        private void LaunchBackend(string scriptPath, string scriptName)
        {
            Process process = null;
            try
            {
                // Everything that can throw (log root creation, output file, process start) stays inside this try:
                // on failure the session reports the error instead of leaving a half-initialised process behind.
                string logRoot = LogDataFolderUtility.LogDataRoot;
                Directory.CreateDirectory(logRoot);
                OpenOutputFile(scriptName, logRoot);

                ProcessStartInfo startInfo = CreatePythonStartInfo(_environment.Interpreter, QuoteArgument(scriptPath), _environment.Arm64Wrapper);
                startInfo.WorkingDirectory = Path.Combine(Application.streamingAssetsPath, "BOData");
                startInfo.RedirectStandardOutput = true;
                startInfo.RedirectStandardError = true;
                // Python writes UTF-8 (PYTHONIOENCODING); decode it as such, not as the console code page.
                startInfo.StandardOutputEncoding = Encoding.UTF8;
                startInfo.StandardErrorEncoding = Encoding.UTF8;
                ConfigurePythonRuntimeEnvironment(startInfo, logRoot, _environment.UserSite);

                process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
                process.OutputDataReceived += OnBackendStdout;
                process.ErrorDataReceived += OnBackendStderr;
                process.Exited += OnBackendExited;

                _deliberateStop = false;
                _backendExitObserved = false;
                _exitMessageShown = false;
                _lastStderrLine = null;
                _launchTimeUtc = DateTime.UtcNow;

                process.Start();
                pythonProcess = process;
                isPythonProcessRunning = true;
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                Debug.Log("Python process started successfully.");
            }
            catch (Exception ex)
            {
                if (process != null)
                {
                    process.OutputDataReceived -= OnBackendStdout;
                    process.ErrorDataReceived -= OnBackendStderr;
                    process.Exited -= OnBackendExited;
                    try { process.Dispose(); } catch (Exception) { }
                }
                pythonProcess = null;
                isPythonProcessRunning = false;
                isSystemStarted = false;
                Debug.LogError("Failed to start the optimizer backend: " + ex);
                ShowFailure("The optimizer backend could not be started:\n" + Shorten(ex.Message, 300));
            }
        }

        private void OnBackendStdout(object sender, DataReceivedEventArgs e)
        {
            if (string.IsNullOrEmpty(e.Data))
                return;
            AppendToOutputFile(e.Data);
            Debug.LogWarning("Python Output: " + e.Data);

            if (e.Data.IndexOf("Server starts, waiting for connection...", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                _backendEverStarted = true;
                isSystemStarted = true;
            }
        }

        private void OnBackendStderr(object sender, DataReceivedEventArgs e)
        {
            if (string.IsNullOrEmpty(e.Data))
                return;
            // Keep tracebacks in output.txt too, next to the stdout they belong to.
            AppendToOutputFile("[stderr] " + e.Data);
            Debug.LogError("Python Error: " + e.Data);

            // The last unindented line of a traceback is the exception and its message
            // (e.g. "OSError: Port 56001 is already in use, ...").
            string line = e.Data;
            if (line.Trim().Length > 0 && !char.IsWhiteSpace(line[0]) &&
                !line.StartsWith("Traceback (most recent call last)", StringComparison.Ordinal))
            {
                _lastStderrLine = line.Trim();
            }
        }

        private void OnBackendExited(object sender, EventArgs args)
        {
            int exitCode = int.MinValue;
            try
            {
                // Read the code from the sender: StopPythonProcess may already have disposed and cleared
                // pythonProcess by the time this runs on a worker thread.
                if (sender is Process exitedProcess)
                    exitCode = exitedProcess.ExitCode;
            }
            catch (Exception)
            {
                // Disposed during shutdown; the code is no longer available.
            }

            isPythonProcessRunning = false;
            isSystemStarted = false;
            if (_deliberateStop)
                return;

            _backendExitCode = exitCode;
            _backendExitObserved = true;
            Debug.LogWarning("Python process exited with code: " + (exitCode == int.MinValue ? "unknown" : exitCode.ToString(CultureInfo.InvariantCulture)));
        }

        private void ReportBackendExit()
        {
            Debug.Log(">>>>> Python Process has EXITED!");
            int exitCode = _backendExitCode;
            bool failed = exitCode != 0 && exitCode != int.MinValue;
            bool sessionRunning = _bomanager != null && _bomanager.simulationRunning;
            // After optimization_finished, or after Unity's stop message (perfect rating), Python ends by itself while
            // the session may continue in Unity (final-design round): that exit is expected, not a failure.
            bool optimizationFinished = _bomanager != null && _bomanager.optimizationFinished;
            if (!failed && (!sessionRunning || optimizationFinished))
                return;

            string reason = _lastStderrLine;
            string logHint = string.IsNullOrEmpty(outputFilePath) ? string.Empty : "\nFull log: " + outputFilePath;
            if (!string.IsNullOrEmpty(reason))
                Debug.LogError($"The optimizer backend exited (code {exitCode}): {reason}{logHint}");

            if (_bomanager == null)
                return;
            if (optimizationFinished)
            {
                // The optimization itself is complete: report the error (its final log writes may be incomplete),
                // but do not abort the final-design round or replace the completion message.
                Debug.LogError($"The optimizer backend exited with code {exitCode} after the optimization finished.{logHint}");
                return;
            }

            string text;
            if (string.IsNullOrEmpty(reason))
            {
                text = _backendEverStarted
                    ? "The optimizer backend stopped unexpectedly.\nPlease restart the application." + logHint
                    : "The system could not be started...\nPlease restart the application." + logHint;
            }
            else
            {
                text = (_backendEverStarted ? "The optimizer backend stopped:\n" : "The optimizer backend could not be started:\n") +
                       Shorten(reason, 300) + logHint;
            }

            _bomanager.TerminateWithError(text);
            // When SocketNetwork noticed the closed connection first, the loop already ended with its generic
            // connection message (TerminateWithError then does nothing): show the backend's own reason instead.
            if (!string.IsNullOrEmpty(reason) && _bomanager.outputText != null)
                _bomanager.outputText.text = text;
        }

        public void StopPythonProcess()
        {
            Process process = pythonProcess;
            pythonProcess = null;
            if (process != null)
            {
                bool exitedOnItsOwn = false;
                try { exitedOnItsOwn = process.HasExited; }
                catch (Exception) { /* not started or already disposed */ }

                if (exitedOnItsOwn)
                {
                    // Python ended before anyone stopped it (e.g. SocketNetwork reports the closed connection and
                    // calls this): keep the exit visible so Update can show the backend's own error.
                    if (!_deliberateStop && !_backendExitObserved)
                    {
                        try { _backendExitCode = process.ExitCode; } catch (Exception) { }
                        _backendExitObserved = true;
                    }
                }
                else
                {
                    _deliberateStop = true;
                    try
                    {
                        process.Kill();
                        process.WaitForExit(5000);
                    }
                    catch (Exception) { /* exited in between */ }
                }

                try { process.Dispose(); } catch (Exception) { }
            }

            isPythonProcessRunning = false;
            isSystemStarted = false;
        }

        // ── Backend console log (output.txt) ───────────────────────────────────────────────────

        private void OpenOutputFile(string scriptName, string logRoot)
        {
            outputFilePath = GetPythonOutputFilePath();
            string directory = Path.GetDirectoryName(outputFilePath);
            Directory.CreateDirectory(directory);

            // Keep the previous session's log: it often holds the reason the last run failed.
            if (File.Exists(outputFilePath))
            {
                try
                {
                    File.Copy(outputFilePath, Path.Combine(directory, "output.prev.txt"), true);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("Could not keep the previous output.txt as output.prev.txt: " + ex.Message);
                }
            }

            lock (_outputFileLock)
            {
                outputFileWriter = new StreamWriter(outputFilePath, false, new UTF8Encoding(false));
                outputFileWriter.WriteLine($"[BOforUnity] {DateTime.Now:yyyy-MM-dd HH:mm:ss}  backend {scriptName}, log root {logRoot}");
                if (_environment != null)
                {
                    outputFileWriter.WriteLine(
                        $"[BOforUnity] Python {_environment.Version} at {_environment.Interpreter} ({_environment.Mode})" +
                        (_environment.Arm64Wrapper ? ", launched via /usr/bin/arch -arm64" : string.Empty));
                    if (!string.IsNullOrEmpty(_environment.PackageReport))
                        outputFileWriter.WriteLine("[BOforUnity] packages: " + _environment.PackageReport);
                }
                outputFileWriter.Flush();
            }
        }

        private void AppendToOutputFile(string line)
        {
            lock (_outputFileLock)
            {
                if (outputFileWriter == null)
                    return;
                try
                {
                    outputFileWriter.WriteLine(line);
                    outputFileWriter.Flush();
                }
                catch (Exception)
                {
                    // Disk full or file closed during shutdown: the Console still has the line.
                }
            }
        }

        private void CloseOutputFile()
        {
            lock (_outputFileLock)
            {
                try { outputFileWriter?.Close(); } catch (Exception) { }
                outputFileWriter = null;
            }
        }

        /// <summary>
        /// Copies output.txt into this session's run folder (LogData/&lt;user&gt;/&lt;condition&gt;/run_N, or the condition
        /// folder when the backend never created a run folder), so the backend's console log is archived with the
        /// study data. Main thread only. Never creates log folders.
        /// </summary>
        private void ArchiveOutputFile()
        {
            if (_outputArchived || _launchTimeUtc == default || string.IsNullOrEmpty(outputFilePath) ||
                !File.Exists(outputFilePath) || _bomanager == null)
                return;
            _outputArchived = true;

            try
            {
                string conditionDir = Path.Combine(
                    LogDataFolderUtility.LogDataRoot,
                    LogDataFolderUtility.NormalizeLogFolderToken(_bomanager.UserId),
                    LogDataFolderUtility.NormalizeLogFolderToken(_bomanager.ConditionId));
                if (!Directory.Exists(conditionDir))
                {
                    Debug.Log("Python console log not archived: no log folder was created for this run (" + conditionDir + ").");
                    return;
                }

                string runDir = FindRunFolderOfThisSession(conditionDir, _launchTimeUtc);
                string target = runDir != null
                    ? Path.Combine(runDir, "output.txt")
                    : Path.Combine(conditionDir, "output_" + _launchTimeUtc.ToLocalTime().ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture) + ".txt");
                File.Copy(outputFilePath, target, true);
                Debug.Log("Python console log archived to " + target);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Could not archive the Python console log: " + ex.Message);
            }
        }

        private static string FindRunFolderOfThisSession(string conditionDir, DateTime launchTimeUtc)
        {
            // Backends write to <condition>/run[_N]; CABOP to <condition>/CABOP/<mode>/run[_N].
            DateTime threshold = launchTimeUtc.AddSeconds(-5);
            string best = null;
            DateTime bestTime = DateTime.MinValue;
            var pending = new Queue<KeyValuePair<string, int>>();
            pending.Enqueue(new KeyValuePair<string, int>(conditionDir, 0));
            while (pending.Count > 0)
            {
                var entry = pending.Dequeue();
                string[] children;
                try { children = Directory.GetDirectories(entry.Key); }
                catch (Exception) { continue; }

                foreach (string child in children)
                {
                    string name = Path.GetFileName(child);
                    if (Regex.IsMatch(name, @"^run(_\d+)?$"))
                    {
                        DateTime written = Directory.GetLastWriteTimeUtc(child);
                        if (written >= threshold && written > bestTime)
                        {
                            best = child;
                            bestTime = written;
                        }
                    }
                    else if (entry.Value < 2)
                    {
                        pending.Enqueue(new KeyValuePair<string, int>(child, entry.Value + 1));
                    }
                }
            }
            return best;
        }

        private static string GetPythonOutputFilePath()
        {
            return Path.Combine(Application.persistentDataPath, "BOData", "BayesianOptimization", "output.txt");
        }

        private static string GetPythonInitRootPath()
        {
            return Path.Combine(Application.streamingAssetsPath, "BOData", "InitData");
        }

        private static void ConfigurePythonRuntimeEnvironment(ProcessStartInfo startInfo, string logRootPath, bool userSite)
        {
            startInfo.Environment["BO_LOG_ROOT"] = logRootPath;
            startInfo.Environment["BO_INIT_ROOT"] = GetPythonInitRootPath();
            startInfo.Environment["PYTHONIOENCODING"] = "utf-8";
            // UTF-8 mode makes open() default to UTF-8 instead of the Windows code page, so the
            // CSV logs are written in the encoding pandas and FinalDesignSelector read them with
            // (otherwise a key or ID such as "Größe" aborts the run when the CSV is read back).
            startInfo.Environment["PYTHONUTF8"] = "1";
            // Keep StreamingAssets clean: without this, importing backend helper
            // modules would create __pycache__ folders inside the Unity project.
            startInfo.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
            // Allow multiple copies of the OpenMP runtime (torch and numpy can both bring one).
            startInfo.Environment["KMP_DUPLICATE_LIB_OK"] = "TRUE";
            ApplySitePolicy(startInfo, userSite);
            Debug.Log("BO log root: " + logRootPath);
        }

        // Packages in another project's `pip install --user` would shadow the environment's own pinned versions;
        // only the --user fallback needs the user site.
        private static void ApplySitePolicy(ProcessStartInfo startInfo, bool userSite)
        {
            if (userSite)
                startInfo.Environment.Remove("PYTHONNOUSERSITE");
            else
                startInfo.Environment["PYTHONNOUSERSITE"] = "1";
        }

        private static ProcessStartInfo CreatePythonStartInfo(string interpreter, string arguments, bool arm64Wrapper)
        {
            var startInfo = new ProcessStartInfo
            {
                // An Intel (Rosetta) Unity would start the universal2 Python as x86_64, for which the pinned torch has
                // no macOS wheels; /usr/bin/arch starts it natively instead.
                FileName = arm64Wrapper ? "/usr/bin/arch" : interpreter,
                Arguments = arm64Wrapper ? "-arm64 " + QuoteArgument(interpreter) + " " + arguments : arguments,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            return startInfo;
        }

        // ── Setup pipeline (worker thread; static so it never touches Unity objects) ──────────────

        private static SetupOutcome RunSetup(SetupContext ctx)
        {
            ctx.DeadlineUtc = DateTime.UtcNow + SetupTimeout;
            try
            {
                if (ctx.Token.IsCancellationRequested)
                    return SetupOutcome.Abort("Python setup was cancelled.");

                if (!File.Exists(ctx.RequirementsPath))
                    return SetupOutcome.Fail("requirements.txt not found: " + ctx.RequirementsPath);
                ctx.RequirementsText = File.ReadAllText(ctx.RequirementsPath);
                ctx.Requirements = ParseRequirements(ctx.RequirementsText);
                Directory.CreateDirectory(ctx.StateDir);

                if (ctx.Os == HostOs.MacOS && IsRunningUnderRosetta(ctx))
                {
                    ctx.UseArm64Wrapper = true;
                    Debug.Log("Unity runs as x86_64 (Rosetta) on Apple Silicon; Python is started as arm64 via /usr/bin/arch -arm64.");
                }

                SetupOutcome outcome = ctx.ManualPython ? SetupConfiguredPython(ctx) : SetupDiscoveredPython(ctx);
                if (!outcome.Success && DateTime.UtcNow >= ctx.DeadlineUtc && !ctx.Token.IsCancellationRequested)
                {
                    outcome = SetupOutcome.Abort(
                        $"Python setup did not finish within {SetupTimeout.TotalMinutes:0} minutes and was stopped. " +
                        "Check the network connection (the first install downloads several hundred MB) and press Play again. " +
                        "Full pip output: " + ctx.PipLogPath);
                }
                return outcome;
            }
            catch (Exception ex)
            {
                Debug.LogError("Python setup error: " + ex);
                return SetupOutcome.Fail("Python setup error: " + ex.Message);
            }
            finally
            {
                ClosePipLog(ctx);
            }
        }

        private static SetupOutcome SetupConfiguredPython(SetupContext ctx)
        {
            string path = NormalizeConfiguredPythonPath(ctx.ManualPythonPath, ctx.Os);
            if (string.IsNullOrEmpty(path))
            {
                return SetupOutcome.Fail(
                    "No Python executable path is configured. Set 'Path of Python Executable' in the " +
                    "BoForUnityManager's Python Settings, or uncheck 'Manually Installed Python'.");
            }
            if (!File.Exists(path))
                return SetupOutcome.Fail("The configured Python executable does not exist: " + path);

            SetStatus("Checking the configured Python…");
            PythonCandidate candidate = ProbeInterpreter(ctx, path, "configured path");
            if (ctx.Token.IsCancellationRequested)
                return SetupOutcome.Abort("Python setup was cancelled.");
            if (candidate == null)
                return SetupOutcome.Fail("The configured Python could not be run: " + path + " (see the Console for the error).");
            if (!IsSupportedPythonVersion(candidate.Version))
            {
                return SetupOutcome.Fail(
                    $"The configured Python is version {candidate.Version}; Python {SupportedPythonMajor}.{MinSupportedPythonMinor} " +
                    $"or newer is required (tested: {SupportedPythonMajor}.{MaxTestedPythonMinor}). Path: {path}");
            }
            if (ctx.Os == HostOs.MacOS && !IsArm64(candidate.Machine))
                return SetupOutcome.Fail(MacArchitectureMessage(candidate));
            if (candidate.Version.Minor > MaxTestedPythonMinor)
            {
                Debug.LogWarning(
                    $"The configured Python {candidate.Version} is newer than the newest tested version " +
                    $"({SupportedPythonMajor}.{MaxTestedPythonMinor}). If the dependency install fails, use Python " +
                    $"{SupportedPythonMajor}.{MaxTestedPythonMinor}.");
            }

            return PrepareEnvironment(ctx, candidate, inPlaceAllowed: true);
        }

        private static SetupOutcome SetupDiscoveredPython(SetupContext ctx)
        {
            SetStatus("Looking for Python…");
            List<PythonCandidate> candidates = DiscoverInterpreters(ctx);
            if (ctx.Token.IsCancellationRequested)
                return SetupOutcome.Abort("Python setup was cancelled.");

            bool bundledInstallAttempted = false;
            if (candidates.Count == 0)
            {
                bundledInstallAttempted = true;
                if (!TryInstallBundledPython(ctx, $"No Python {SupportedPythonMajor}.{MinSupportedPythonMinor} or newer was found", out string installError))
                    return ctx.Token.IsCancellationRequested ? SetupOutcome.Abort(installError) : SetupOutcome.Fail(installError);
                candidates = DiscoverInterpreters(ctx);
                if (candidates.Count == 0)
                {
                    return SetupOutcome.Fail(
                        $"Python {BundledPythonVersionLabel} was installed, but no usable interpreter was found afterwards. " +
                        "Restart Unity and press Play again.");
                }
            }

            var failures = new List<string>();
            foreach (PythonCandidate candidate in candidates)
            {
                if (ctx.Token.IsCancellationRequested)
                    return SetupOutcome.Abort("Python setup was cancelled.");

                SetupOutcome outcome = PrepareEnvironment(ctx, candidate, inPlaceAllowed: false);
                if (outcome.Success || outcome.Aborted)
                    return outcome;

                failures.Add($"Python {candidate.Version} at {candidate.Path}: {outcome.Error}");
                Debug.LogWarning($"Python {candidate.Version} at {candidate.Path} could not be prepared; trying the next interpreter. Reason: {outcome.Error}");
            }

            // Only interpreters newer than the tested version were found, and none worked: fall back to the bundled 3.13.
            if (!bundledInstallAttempted && !candidates.Any(IsTestedVersion) && IsBundledInstallSupported(ctx.Os))
            {
                Debug.LogWarning(
                    $"Only Python versions newer than the tested {SupportedPythonMajor}.{MaxTestedPythonMinor} were found and none " +
                    $"could be prepared; installing the bundled Python {BundledPythonVersionLabel}.");
                if (TryInstallBundledPython(ctx, "No tested Python interpreter could be prepared", out string installError))
                {
                    PythonCandidate bundled = DiscoverInterpreters(ctx).FirstOrDefault(c => c.IsBundled);
                    if (bundled != null)
                    {
                        SetupOutcome outcome = PrepareEnvironment(ctx, bundled, inPlaceAllowed: false);
                        if (outcome.Success || outcome.Aborted)
                            return outcome;
                        failures.Add($"Python {bundled.Version} at {bundled.Path}: {outcome.Error}");
                    }
                }
                else
                {
                    failures.Add(installError);
                }
            }

            return SetupOutcome.Fail(failures.Count == 1
                ? failures[0]
                : "No Python interpreter could be prepared:\n- " + string.Join("\n- ", failures));
        }

        private static bool IsTestedVersion(PythonCandidate candidate)
        {
            return candidate.Version.Minor <= MaxTestedPythonMinor;
        }

        private static SetupOutcome PrepareEnvironment(SetupContext ctx, PythonCandidate candidate, bool inPlaceAllowed)
        {
            var env = new PythonEnvironment
            {
                BaseInterpreter = candidate.Path,
                Version = candidate.Version,
                Arm64Wrapper = ctx.UseArm64Wrapper,
            };

            // One install at a time per project, across Unity instances and across script reloads.
            using (InstallLock installLock = AcquireInstallLock(ctx))
            {
                if (installLock == null)
                {
                    return SetupOutcome.Abort(ctx.Token.IsCancellationRequested
                        ? "Python setup was cancelled."
                        : "Timed out waiting for another Unity instance to finish installing the Python dependencies.");
                }

                if (inPlaceAllowed && candidate.IsEnvironment)
                {
                    // pip refuses --user inside a virtual environment; install into the environment itself.
                    env.Interpreter = candidate.Path;
                    env.Mode = "the configured virtual/conda environment";
                }
                else
                {
                    string venvDir = GetPrivateVenvDirFor(ctx, candidate);
                    string venvPython = EnsurePrivateVenv(ctx, candidate, venvDir, out string venvError);
                    if (ctx.Token.IsCancellationRequested)
                        return SetupOutcome.Abort("Python setup was cancelled.");
                    if (venvPython != null)
                    {
                        env.Interpreter = venvPython;
                        env.Mode = "private environment " + venvDir;
                    }
                    else
                    {
                        env.Interpreter = candidate.Path;
                        env.UserSite = true;
                        env.Mode = "user site-packages (pip --user)";
                        Debug.LogWarning(
                            $"Could not create the private Python environment at {venvDir}: {venvError} " +
                            "Falling back to installing the dependencies into the user site-packages (pip install --user)." +
                            (ctx.Os == HostOs.Linux
                                ? $" On Debian/Ubuntu, install the venv module (sudo apt install python{candidate.Version.Major}.{candidate.Version.Minor}-venv) so the backend gets an isolated environment."
                                : string.Empty));
                    }
                }

                string pipArguments = BuildPipInstallArguments(ctx, env.UserSite);
                string stampPath = Path.Combine(ctx.StateDir, "requirements-" + ShortHash(NormalizePathKey(env.Interpreter, ctx.Os)) + ".stamp");
                string stamp =
                    "interpreter=" + env.Interpreter + "\n" +
                    "version=" + env.Version + "\n" +
                    "mode=" + env.Mode + "\n" +
                    "pip=" + pipArguments + "\n" +
                    "requirements=\n" + ctx.RequirementsText;

                SetStatus($"Checking Python packages ({env.Interpreter})…");
                PackageCheck check = CheckInstalledPackages(ctx, env);
                if (check.Aborted)
                    return SetupOutcome.Abort("Python setup was cancelled.");
                if (check.Ok && IsStampCurrent(stampPath, stamp))
                {
                    env.PackageReport = check.Report;
                    SetStatus("Python dependencies verified.");
                    return SetupOutcome.Ok(env);
                }

                Debug.Log(check.Ok
                    ? $"requirements.txt or the install options changed since the last install into {env.Interpreter}; running pip."
                    : $"Installing Python dependencies into {env.Mode} ({env.Interpreter}): {check.Summary}");

                OpenPipLog(ctx);
                if (!EnsurePip(ctx, env, out string pipError))
                    return ctx.Token.IsCancellationRequested ? SetupOutcome.Abort(pipError) : SetupOutcome.Fail(pipError);

                string installHeadline = $"Installing Python dependencies into {env.Mode} (first run only; this can take several minutes)…";
                SetStatus(installHeadline);
                var installOptions = new ChildOptions
                {
                    UserSite = env.UserSite,
                    LogToPipLog = true,
                    TrackInLockFile = true,
                    OnLine = line => ReportPipProgress(installHeadline, line),
                };
                ChildResult install = null;
                if (!string.IsNullOrEmpty(ctx.WheelsDir))
                {
                    // With --find-links alone pip still queries the package index for every requirement, which
                    // offline means minutes of retries even when every wheel is there: try the wheels folder alone first.
                    install = RunPython(ctx, env.Interpreter, BuildPipInstallArguments(ctx, env.UserSite, wheelsOnly: true),
                        PipInstallTimeout, installOptions);
                    if (!install.Succeeded && !install.Cancelled && !install.TimedOut)
                    {
                        Debug.Log("Installation/wheels does not hold every required package; retrying with the package index.");
                        install = null;
                    }
                }
                if (install == null)
                    install = RunPython(ctx, env.Interpreter, pipArguments, PipInstallTimeout, installOptions);
                if (install.Cancelled)
                    return SetupOutcome.Abort("Python setup was cancelled.");
                if (install.TimedOut)
                {
                    string message =
                        $"Installing the Python dependencies did not finish in time and was stopped. Check the network " +
                        "connection (the first install downloads several hundred MB) and press Play again. " +
                        "Full pip output: " + ctx.PipLogPath;
                    LogChildFailure(ctx, message, install);
                    return SetupOutcome.Abort(message);
                }
                if (!install.Succeeded)
                {
                    string message = DescribePipFailure(ctx, env, install);
                    LogChildFailure(ctx, message, install);
                    // Another interpreter cannot fix an unreachable package index.
                    return IsNetworkFailure(install) ? SetupOutcome.Abort(message) : SetupOutcome.Fail(message);
                }

                check = CheckInstalledPackages(ctx, env);
                if (check.Aborted)
                    return SetupOutcome.Abort("Python setup was cancelled.");
                if (!check.Ok)
                {
                    return SetupOutcome.Fail(
                        "pip finished, but the installed packages do not match requirements.txt: " + check.Summary +
                        (env.UserSite ? " (a package installed elsewhere may shadow the user site-packages)." : "."));
                }

                WriteStamp(stampPath, stamp);
                env.PackageReport = check.Report;
                SetStatus("Python dependencies installed.");
                return SetupOutcome.Ok(env);
            }
        }

        private static string BuildPipInstallArguments(SetupContext ctx, bool userSite, bool wheelsOnly = false)
        {
            var sb = new StringBuilder("-m pip install");
            if (userSite)
                sb.Append(" --user");
            sb.Append(" -r ").Append(QuoteArgument(ctx.RequirementsPath));
            sb.Append(" --disable-pip-version-check --no-input --progress-bar off");
            if (wheelsOnly)
            {
                sb.Append(" --no-index");
            }
            else if (ctx.Os == HostOs.Linux)
            {
                sb.Append(" --extra-index-url ").Append(LinuxCpuTorchIndexUrl);
            }
            // Offline labs: wheels placed in Installation/wheels are used before (or instead of) the package index.
            if (!string.IsNullOrEmpty(ctx.WheelsDir))
                sb.Append(" --find-links ").Append(QuoteArgument(ctx.WheelsDir));
            return sb.ToString();
        }

        private static void ReportPipProgress(string headline, string line)
        {
            string trimmed = line.Trim();
            if (trimmed.StartsWith("Collecting ", StringComparison.Ordinal) ||
                trimmed.StartsWith("Downloading ", StringComparison.Ordinal) ||
                trimmed.StartsWith("Using cached ", StringComparison.Ordinal) ||
                trimmed.StartsWith("Installing collected packages", StringComparison.Ordinal) ||
                trimmed.StartsWith("Successfully installed", StringComparison.Ordinal) ||
                trimmed.StartsWith("Attempting uninstall", StringComparison.Ordinal) ||
                trimmed.StartsWith("Building ", StringComparison.Ordinal))
            {
                SetStatus(headline + "\n" + Shorten(trimmed, 110));
            }
        }

        private static bool IsNetworkFailure(ChildResult result)
        {
            string all = string.Join("\n", result.Tail);
            return all.IndexOf("NewConnectionError", StringComparison.Ordinal) >= 0 ||
                   all.IndexOf("Max retries exceeded", StringComparison.Ordinal) >= 0 ||
                   all.IndexOf("Temporary failure in name resolution", StringComparison.Ordinal) >= 0 ||
                   all.IndexOf("Name or service not known", StringComparison.Ordinal) >= 0 ||
                   all.IndexOf("getaddrinfo failed", StringComparison.Ordinal) >= 0 ||
                   all.IndexOf("ConnectTimeoutError", StringComparison.Ordinal) >= 0 ||
                   all.IndexOf("CERTIFICATE_VERIFY_FAILED", StringComparison.Ordinal) >= 0 ||
                   all.IndexOf("ProxyError", StringComparison.Ordinal) >= 0;
        }

        private static string DescribePipFailure(SetupContext ctx, PythonEnvironment env, ChildResult install)
        {
            string message = $"pip could not install requirements.txt into {env.Mode} ({DescribeChildFailure(install, "pip install")}).";
            string all = string.Join("\n", install.Tail);
            if (IsNetworkFailure(install))
            {
                message += " pip could not reach the package index: check the network connection or proxy settings, or place " +
                           "the wheels in StreamingAssets/BOData/Installation/wheels for an offline install.";
            }
            else if (all.IndexOf("externally-managed-environment", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                message += " This Python is externally managed (PEP 668) and refuses user installs. Install the venv module " +
                           "(e.g. sudo apt install python3.13-venv) so BOforUnity can create its private environment, or point " +
                           "'Manually Installed Python' at a virtual environment.";
            }
            else if (ctx.Os == HostOs.Windows && all.IndexOf("Long Path", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                message += " Windows refused a long file path: enable long path support (see the pip log) or shorten the project's company/product name.";
            }
            else if (env.Version != null && env.Version.Minor > MaxTestedPythonMinor &&
                     all.IndexOf("No matching distribution", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                message += $" Python {env.Version} is probably too new for the pinned packages; install Python {SupportedPythonMajor}.{MaxTestedPythonMinor}.";
            }
            return message + " Full pip output: " + ctx.PipLogPath;
        }

        private static bool EnsurePip(SetupContext ctx, PythonEnvironment env, out string error)
        {
            error = null;
            SetStatus("Checking pip…");
            ChildResult probe = RunPython(ctx, env.Interpreter, "-m pip --version", ProbeTimeout,
                new ChildOptions { UserSite = env.UserSite, LogToPipLog = true });
            if (probe.Succeeded)
                return true;
            if (probe.Cancelled)
            {
                error = "Python setup was cancelled.";
                return false;
            }

            SetStatus("Installing pip…");
            ChildResult ensure = RunPython(ctx, env.Interpreter, "-m ensurepip --upgrade" + (env.UserSite ? " --user" : string.Empty),
                EnsurePipTimeout, new ChildOptions { UserSite = env.UserSite, LogToPipLog = true, TrackInLockFile = true });
            if (ensure.Succeeded)
                return true;

            error = $"pip is not available for {env.Interpreter} and ensurepip failed ({DescribeChildFailure(ensure, "ensurepip")}). " +
                    "Full output: " + ctx.PipLogPath;
            if (!ensure.Cancelled)
                LogChildFailure(ctx, error, ensure);
            return false;
        }

        // The private environment for this base interpreter: the main one when it is unused or was built from this base,
        // otherwise a sibling keyed by the base path. Switching between interpreters (e.g. a manual one and the
        // discovered bundled one) then reuses each environment instead of wiping the working one with --clear, which
        // offline destroyed the only usable install before the new one failed.
        private static string GetPrivateVenvDirFor(SetupContext ctx, PythonCandidate baseCandidate)
        {
            string mainBase = ReadVenvBase(ctx);
            if (mainBase == null || !File.Exists(GetVenvPython(ctx.Os, ctx.VenvDir)) ||
                string.Equals(mainBase, baseCandidate.Path, StringComparison.Ordinal))
                return ctx.VenvDir;
            return ctx.VenvDir + "-" + ShortHash(NormalizePathKey(baseCandidate.Path, ctx.Os));
        }

        // Returns the private venv's interpreter, or null (with the reason) when no venv can be created.
        private static string EnsurePrivateVenv(SetupContext ctx, PythonCandidate baseCandidate, string venvDir, out string error)
        {
            error = null;
            string venvPython = GetVenvPython(ctx.Os, venvDir);
            string markerPath = Path.Combine(venvDir, "bo4unity-base.txt");
            string marker = "base=" + baseCandidate.Path + "\nversion=" + baseCandidate.Version.Major + "." + baseCandidate.Version.Minor + "\n";

            if (File.Exists(venvPython) && string.Equals(ReadTextOrNull(markerPath), marker, StringComparison.Ordinal))
            {
                PythonCandidate existing = ProbeInterpreter(ctx, venvPython, "private environment");
                if (existing != null && existing.Version.Major == baseCandidate.Version.Major &&
                    existing.Version.Minor == baseCandidate.Version.Minor)
                    return venvPython;
                if (ctx.Token.IsCancellationRequested)
                {
                    error = "cancelled";
                    return null;
                }
                Debug.LogWarning("The private Python environment at " + venvDir + " does not start; recreating it.");
            }
            else if (Directory.Exists(venvDir))
            {
                Debug.Log($"Recreating the private Python environment at {venvDir} for {baseCandidate.Path} (Python {baseCandidate.Version}).");
            }

            OpenPipLog(ctx);
            SetStatus($"Creating a private Python environment ({venvDir})…");
            ChildResult created = RunPython(ctx, baseCandidate.Path, "-m venv --clear " + QuoteArgument(venvDir), VenvTimeout,
                new ChildOptions { LogToPipLog = true, TrackInLockFile = true });
            if (!created.Succeeded || !File.Exists(venvPython))
            {
                error = created.Succeeded
                    ? "python -m venv finished, but " + venvPython + " is missing."
                    : DescribeChildFailure(created, "python -m venv") + ".";
                return null;
            }

            try
            {
                File.WriteAllText(markerPath, marker);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Could not record the private environment's base interpreter: " + ex.Message);
            }
            Debug.Log($"Created the private Python environment {venvDir} from {baseCandidate.Path} (Python {baseCandidate.Version}).");
            return venvPython;
        }

        private static string GetVenvPython(HostOs os, string venvDir)
        {
            return os == HostOs.Windows
                ? Path.Combine(venvDir, "Scripts", "python.exe")
                : Path.Combine(venvDir, "bin", "python3");
        }

        private static PackageCheck CheckInstalledPackages(SetupContext ctx, PythonEnvironment env)
        {
            ChildResult probe = RunPython(ctx, env.Interpreter, "-c " + QuoteArgument(DistributionsProbeScript), ProbeTimeout,
                new ChildOptions { UserSite = env.UserSite });
            if (probe.Cancelled)
                return new PackageCheck { Aborted = true };
            if (!probe.Succeeded)
                return new PackageCheck { Summary = "could not list the installed packages (" + DescribeChildFailure(probe, "the package check") + ")" };

            Dictionary<string, string> installed = null;
            foreach (string line in probe.StdOut.Split('\n'))
            {
                string trimmed = line.Trim();
                if (!trimmed.StartsWith("BO4U-DISTS ", StringComparison.Ordinal))
                    continue;
                try
                {
                    using (var reader = new JsonTextReader(new StringReader(trimmed.Substring("BO4U-DISTS ".Length))))
                        installed = PackageListJson.Deserialize<Dictionary<string, string>>(reader);
                }
                catch (Exception ex)
                {
                    return new PackageCheck { Summary = "could not read the installed package list (" + ex.Message + ")" };
                }
            }
            if (installed == null)
                return new PackageCheck { Summary = "could not read the installed package list" };

            var problems = new List<string>();
            var report = new List<string>();
            foreach (Requirement requirement in ctx.Requirements)
            {
                if (!installed.TryGetValue(requirement.NormalizedName, out string version) || string.IsNullOrEmpty(version))
                {
                    problems.Add(requirement.Name + " is missing");
                    continue;
                }
                report.Add(requirement.Name + " " + version);
                if (requirement.PinnedVersion != null && !PinnedVersionMatches(requirement.PinnedVersion, version))
                    problems.Add($"{requirement.Name} {version} is installed, requirements.txt pins {requirement.PinnedVersion}");
            }

            return new PackageCheck
            {
                Ok = problems.Count == 0,
                Summary = problems.Count == 0 ? "all pinned versions installed" : string.Join("; ", problems),
                Report = string.Join(", ", report),
            };
        }

        private static List<Requirement> ParseRequirements(string text)
        {
            var requirements = new List<Requirement>();
            var pattern = new Regex(@"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$");
            foreach (string rawLine in text.Split('\n'))
            {
                string line = rawLine;
                int comment = line.IndexOf('#');
                if (comment >= 0)
                    line = line.Substring(0, comment);
                line = line.Trim();
                if (line.Length == 0 || line.StartsWith("-", StringComparison.Ordinal))
                    continue;

                Match match = pattern.Match(line);
                if (!match.Success)
                    continue;

                string spec = match.Groups[3].Value.Trim();
                string pinned = null;
                // Exactly "==X" (no markers, no further specifiers) is verified; anything else is checked for presence.
                Match pin = Regex.Match(spec, @"^==\s*([^\s,;]+)$");
                if (pin.Success && !pin.Groups[1].Value.Contains("*"))
                    pinned = pin.Groups[1].Value;
                else if (spec.Contains(";"))
                    continue; // environment markers: may legitimately be absent on this platform

                requirements.Add(new Requirement
                {
                    Name = match.Groups[1].Value,
                    NormalizedName = NormalizeDistributionName(match.Groups[1].Value),
                    PinnedVersion = pinned,
                });
            }
            return requirements;
        }

        private static string NormalizeDistributionName(string name)
        {
            return Regex.Replace(name ?? string.Empty, "[-_.]+", "-").ToLowerInvariant();
        }

        // PEP 440: "==2.14.1" matches "2.14.1+cpu" (local label) and "2.14.1.0" (zero padding).
        private static bool PinnedVersionMatches(string pinned, string installed)
        {
            string a = StripLocalVersion(pinned);
            string b = StripLocalVersion(installed);
            if (string.Equals(a, b, StringComparison.OrdinalIgnoreCase))
                return true;

            string[] pa = a.Split('.');
            string[] pb = b.Split('.');
            if (!pa.Concat(pb).All(part => part.Length > 0 && part.All(char.IsDigit)))
                return false;
            int length = Math.Max(pa.Length, pb.Length);
            for (int i = 0; i < length; i++)
            {
                long va = i < pa.Length ? long.Parse(pa[i], CultureInfo.InvariantCulture) : 0;
                long vb = i < pb.Length ? long.Parse(pb[i], CultureInfo.InvariantCulture) : 0;
                if (va != vb)
                    return false;
            }
            return true;
        }

        private static string StripLocalVersion(string version)
        {
            string v = (version ?? string.Empty).Trim();
            int plus = v.IndexOf('+');
            return plus >= 0 ? v.Substring(0, plus) : v;
        }

        private static bool IsStampCurrent(string stampPath, string expectedStamp)
        {
            try
            {
                return File.Exists(stampPath) && string.Equals(File.ReadAllText(stampPath), expectedStamp, StringComparison.Ordinal);
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static void WriteStamp(string stampPath, string stamp)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(stampPath));
                File.WriteAllText(stampPath, stamp);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Could not write the requirements stamp: " + ex.Message);
            }
        }

        // ── Interpreter discovery ──────────────────────────────────────────────────────────────

        private static List<PythonCandidate> DiscoverInterpreters(SetupContext ctx)
        {
            var paths = new List<KeyValuePair<string, string>>();
            var seen = new HashSet<string>(ctx.Os == HostOs.Windows ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
            // The main private environment and its per-base siblings (python-venv-<hash>) are never base candidates.
            string ownVenvPrefix = ctx.VenvDir;

            void Add(string path, string source)
            {
                if (string.IsNullOrWhiteSpace(path))
                    return;
                path = path.Trim();
                if (path.StartsWith(ownVenvPrefix, StringComparison.OrdinalIgnoreCase) || !File.Exists(path) || !seen.Add(path))
                    return;
                paths.Add(new KeyValuePair<string, string>(path, source));
            }

            Add(GetBundledPythonPath(ctx.Os), "bundled install location");

            var searchDirs = new List<KeyValuePair<string, string>>();
            foreach (string dir in (Environment.GetEnvironmentVariable("PATH") ?? string.Empty).Split(Path.PathSeparator))
            {
                if (!string.IsNullOrWhiteSpace(dir))
                    searchDirs.Add(new KeyValuePair<string, string>(dir.Trim().Trim('"'), "PATH"));
            }

            string[] names;
            switch (ctx.Os)
            {
                case HostOs.Windows:
                    names = new[] { "python.exe" };
                    try
                    {
                        string localPrograms = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python");
                        if (Directory.Exists(localPrograms))
                            foreach (string dir in Directory.GetDirectories(localPrograms, "Python3*"))
                                searchDirs.Add(new KeyValuePair<string, string>(dir, "user install"));
                        string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
                        if (Directory.Exists(programFiles))
                            foreach (string dir in Directory.GetDirectories(programFiles, "Python3*"))
                                searchDirs.Add(new KeyValuePair<string, string>(dir, "Program Files"));
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning("Error searching the Python install folders: " + ex.Message);
                    }
                    break;
                case HostOs.MacOS:
                    names = new[] { "python3.13", "python3" };
                    // Unity started from the Hub or Finder does not inherit the shell's PATH.
                    searchDirs.Add(new KeyValuePair<string, string>("/opt/homebrew/bin", "Homebrew"));
                    searchDirs.Add(new KeyValuePair<string, string>("/usr/local/bin", "/usr/local/bin"));
                    break;
                case HostOs.Linux:
                    names = new[] { "python3.13", "python3" };
                    searchDirs.Add(new KeyValuePair<string, string>("/usr/local/bin", "/usr/local/bin"));
                    searchDirs.Add(new KeyValuePair<string, string>("/usr/bin", "/usr/bin"));
                    break;
                default:
                    names = new[] { "python3", "python" };
                    break;
            }

            foreach (var dir in searchDirs)
            {
                if (ctx.Os == HostOs.Windows && dir.Key.IndexOf("WindowsApps", StringComparison.OrdinalIgnoreCase) >= 0)
                    continue; // Microsoft Store alias stubs open the Store instead of running Python
                foreach (string name in names)
                {
                    string candidate;
                    try { candidate = Path.Combine(dir.Key, name); }
                    catch (ArgumentException) { continue; }
                    Add(candidate, dir.Value);
                }
            }

            string bundledPath = GetBundledPythonPath(ctx.Os);
            string previousBase = ReadVenvBase(ctx);
            var valid = new List<PythonCandidate>();
            var seenInterpreters = new Dictionary<string, string>(
                ctx.Os == HostOs.Windows ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
            int order = 0;
            foreach (var entry in paths)
            {
                if (ctx.Token.IsCancellationRequested)
                    break;
                if (ctx.Os == HostOs.MacOS && entry.Key == "/usr/bin/python3")
                {
                    // Apple's stub: Python 3.9 from the Command Line Tools, and an install dialog when they are missing.
                    Debug.Log("Skipping /usr/bin/python3 (Apple's Command Line Tools Python).");
                    continue;
                }

                SetStatus("Looking for Python… (" + entry.Key + ")");
                PythonCandidate candidate = ProbeInterpreter(ctx, entry.Key, entry.Value);
                if (candidate == null)
                    continue;
                candidate.Order = order++;
                candidate.IsBundled = !string.IsNullOrEmpty(bundledPath) && entry.Key == bundledPath;

                if (!IsSupportedPythonVersion(candidate.Version))
                {
                    Debug.Log($"Skipping Python {candidate.Version} at {candidate.Path}: Python {SupportedPythonMajor}.{MinSupportedPythonMinor} or newer is required.");
                    continue;
                }
                if (ctx.Os == HostOs.MacOS && !IsArm64(candidate.Machine))
                {
                    Debug.Log("Skipping " + candidate.Path + ": " + MacArchitectureMessage(candidate));
                    continue;
                }
                if (candidate.Version.Minor > MaxTestedPythonMinor)
                {
                    Debug.Log(
                        $"Python {candidate.Version} at {candidate.Path} is newer than the newest tested version " +
                        $"({SupportedPythonMajor}.{MaxTestedPythonMinor}); it is only used if no tested interpreter works.");
                }
                // The same interpreter reached through another link would only repeat the same install attempt.
                string identity = candidate.RealPath + "|" + candidate.Version;
                if (seenInterpreters.TryGetValue(identity, out string firstPath))
                {
                    Debug.Log($"Skipping {candidate.Path}: same interpreter as {firstPath}.");
                    continue;
                }
                seenInterpreters[identity] = candidate.Path;
                Debug.Log($"Python candidate: {candidate.Path} (Python {candidate.Version}, {candidate.Machine}, from {candidate.Source})");
                valid.Add(candidate);
            }

            return valid
                .OrderBy(c => c.Version.Minor > MaxTestedPythonMinor ? 1 : 0)
                .ThenByDescending(c => c.IsBundled)
                // Keep using the interpreter the private environment was built from: switching costs a full reinstall.
                .ThenByDescending(c => previousBase != null && string.Equals(c.Path, previousBase, StringComparison.Ordinal))
                .ThenBy(c => c.Version.Minor > MaxTestedPythonMinor ? c.Version.Minor : 0)
                .ThenByDescending(c => c.Version)
                .ThenBy(c => c.Order)
                .ToList();
        }

        private static string ReadVenvBase(SetupContext ctx)
        {
            string marker = ReadTextOrNull(Path.Combine(ctx.VenvDir, "bo4unity-base.txt"));
            if (marker == null)
                return null;
            foreach (string line in marker.Split('\n'))
            {
                if (line.StartsWith("base=", StringComparison.Ordinal))
                    return line.Substring("base=".Length).Trim();
            }
            return null;
        }

        private static PythonCandidate ProbeInterpreter(SetupContext ctx, string path, string source)
        {
            ChildResult result = RunPython(ctx, path, "-c " + QuoteArgument(InterpreterProbeScript), ProbeTimeout, new ChildOptions());
            if (result.Cancelled)
                return null;
            if (!result.Succeeded)
            {
                Debug.LogWarning($"Skipping Python candidate {path} ({source}): {DescribeChildFailure(result, "the version check")}");
                return null;
            }

            foreach (string line in result.StdOut.Split('\n'))
            {
                string trimmed = line.Trim();
                if (!trimmed.StartsWith("BO4U|", StringComparison.Ordinal))
                    continue;
                string[] parts = trimmed.Split('|');
                if (parts.Length >= 6 &&
                    int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int major) &&
                    int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out int minor) &&
                    int.TryParse(parts[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out int micro))
                {
                    return new PythonCandidate
                    {
                        Path = path,
                        Source = source,
                        Version = new Version(major, minor, micro),
                        Machine = parts[4],
                        IsEnvironment = parts[5] == "1",
                        RealPath = parts.Length > 6 ? string.Join("|", parts, 6, parts.Length - 6) : path,
                    };
                }
            }

            Debug.LogWarning($"Skipping Python candidate {path} ({source}): unexpected output: {Shorten(string.Join(" | ", result.Tail), 300)}");
            return null;
        }

        private static bool IsSupportedPythonVersion(Version version)
        {
            return version != null && version.Major == SupportedPythonMajor && version.Minor >= MinSupportedPythonMinor;
        }

        private static bool IsArm64(string machine)
        {
            return string.Equals(machine, "arm64", StringComparison.OrdinalIgnoreCase);
        }

        private static string MacArchitectureMessage(PythonCandidate candidate)
        {
            return $"this Python runs as '{candidate.Machine}', but the pinned PyTorch ships Python {SupportedPythonMajor}.{MaxTestedPythonMinor} " +
                   "macOS wheels for arm64 (Apple Silicon) only. Use an arm64 or universal2 Python (e.g. the bundled python.org installer).";
        }

        // Unity's own architecture decides the slice a universal2 Python starts with. sysctl inherits it, so
        // "sysctl.proc_translated = 1" means Unity runs under Rosetta on Apple Silicon.
        private static bool IsRunningUnderRosetta(SetupContext ctx)
        {
            if (!File.Exists("/usr/sbin/sysctl") || !File.Exists("/usr/bin/arch"))
                return false;
            ChildResult result = RunChild(ctx, "/usr/sbin/sysctl", "-n sysctl.proc_translated", ProbeTimeout, new ChildOptions(), null);
            return result.Succeeded && result.StdOut.Trim() == "1";
        }

        private static string GetBundledPythonPath(HostOs os)
        {
            switch (os)
            {
                case HostOs.Windows:
                    return Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                        $"Python{SupportedPythonMajor}{BundledPythonMinor}",
                        "python.exe");
                case HostOs.MacOS:
                    return $"/Library/Frameworks/Python.framework/Versions/{SupportedPythonMajor}.{BundledPythonMinor}/bin/python3";
                case HostOs.Linux:
                    return $"/usr/bin/python{SupportedPythonMajor}.{BundledPythonMinor}";
                default:
                    return null;
            }
        }

        private static string NormalizeConfiguredPythonPath(string configured, HostOs os)
        {
            string path = (configured ?? string.Empty).Trim();
            // "Copy as path" (Windows) and shell habits add quotes around the path.
            while (path.Length >= 2 &&
                   ((path[0] == '"' && path[path.Length - 1] == '"') || (path[0] == '\'' && path[path.Length - 1] == '\'')))
            {
                path = path.Substring(1, path.Length - 2).Trim();
            }
            if (os != HostOs.Windows && path.StartsWith("~/", StringComparison.Ordinal))
                path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), path.Substring(2));
            if (path.Length > 0 && path.IndexOf(Path.DirectorySeparatorChar) < 0 && path.IndexOf(Path.AltDirectorySeparatorChar) < 0)
            {
                // A bare command name such as "python3": resolve it on PATH.
                foreach (string dir in (Environment.GetEnvironmentVariable("PATH") ?? string.Empty).Split(Path.PathSeparator))
                {
                    if (string.IsNullOrWhiteSpace(dir))
                        continue;
                    try
                    {
                        string resolved = Path.Combine(dir.Trim(), path);
                        if (File.Exists(resolved))
                            return resolved;
                        if (os == HostOs.Windows && File.Exists(resolved + ".exe"))
                            return resolved + ".exe";
                    }
                    catch (ArgumentException)
                    {
                    }
                }
            }
            return path;
        }

        // ── Bundled Python installation (Windows/macOS) ────────────────────────────────────────

        private static bool IsBundledInstallSupported(HostOs os)
        {
            return os == HostOs.Windows || os == HostOs.MacOS;
        }

        private static bool TryInstallBundledPython(SetupContext ctx, string reason, out string error)
        {
            // Two Unity instances (or a restart during the install) must not run the system installer twice.
            using (InstallLock installLock = AcquireInstallLock(ctx))
            {
                if (installLock == null)
                {
                    error = ctx.Token.IsCancellationRequested
                        ? "Python setup was cancelled."
                        : "Timed out waiting for another Unity instance to finish installing Python.";
                    return false;
                }

                // Another instance may have installed it while this one waited for the lock.
                string bundledPath = GetBundledPythonPath(ctx.Os);
                if (!string.IsNullOrEmpty(bundledPath) && File.Exists(bundledPath))
                {
                    error = null;
                    return true;
                }

                return TryInstallBundledPythonUnlocked(ctx, reason, out error);
            }
        }

        private static bool TryInstallBundledPythonUnlocked(SetupContext ctx, string reason, out string error)
        {
            error = null;
            SetStatus($"{reason}. Installing bundled Python {BundledPythonVersionLabel} (may require admin confirmation)…");
            Debug.Log($"{reason}; installing the bundled Python {BundledPythonVersionLabel}.");

            switch (ctx.Os)
            {
                case HostOs.MacOS:
                {
                    string pkgPath = Path.Combine(ctx.BundledInstallerRoot, "MacOs", "Data", "Installation_Objects", MacPythonPkgFile);
                    if (!VerifyInstaller(pkgPath, MacPythonPkgSha256, out error))
                        break;
                    SetStatus($"Installing bundled Python {BundledPythonVersionLabel} (a macOS administrator prompt may appear)…");
                    ChildResult result = RunMacPkgInstallerWithAdminPrompt(ctx, pkgPath);
                    if (!result.Succeeded)
                    {
                        error = $"The bundled Python installation did not complete ({DescribeChildFailure(result, "the installer")}).";
                        break;
                    }
                    break;
                }
                case HostOs.Windows:
                {
                    string windowsDir = Path.Combine(ctx.BundledInstallerRoot, "Windows");
                    string installBat = Path.Combine(windowsDir, "installation_python.bat");
                    if (!File.Exists(installBat))
                    {
                        error = "The bundled Python installer script was not found: " + installBat;
                        break;
                    }
                    if (!VerifyInstaller(Path.Combine(windowsDir, "Installation_Objects", WindowsPythonInstallerFile), WindowsPythonInstallerSha256, out error) ||
                        !VerifyInstaller(Path.Combine(windowsDir, "Installation_Objects", WindowsVcRedistFile), WindowsVcRedistSha256, out error))
                        break;

                    SetStatus($"Installing bundled Python {BundledPythonVersionLabel} (a Windows UAC prompt may appear)…");
                    // cmd strips the outer quotes: the script path stays quoted even if it contains spaces.
                    ChildResult result = RunChild(ctx, "cmd.exe", "/c \"" + QuoteArgument(installBat) + " --unattended\"",
                        BundledInstallTimeout, new ChildOptions(), null);
                    if (!result.Succeeded)
                    {
                        error = $"The bundled Python installation did not complete ({DescribeChildFailure(result, "installation_python.bat")}).";
                        LogChildFailure(ctx, error, result);
                    }
                    break;
                }
                default:
                    error = "Python 3.13 cannot be installed automatically on this platform. Run " +
                            "Assets/StreamingAssets/BOData/Installation/Linux/install_python.sh in a terminal (or install " +
                            "Python 3.13 and its venv module with your package manager), then press Play again.";
                    break;
            }

            if (error == null && !File.Exists(GetBundledPythonPath(ctx.Os)))
                error = "The bundled Python installer finished, but " + GetBundledPythonPath(ctx.Os) + " does not exist.";

            if (error != null)
            {
                Debug.LogError("Bundled Python installation failed: " + error);
                return false;
            }

            Debug.Log($"Bundled Python {BundledPythonVersionLabel} installed at {GetBundledPythonPath(ctx.Os)}.");
            return true;
        }

        private static bool VerifyInstaller(string path, string expectedSha256, out string error)
        {
            error = null;
            if (!File.Exists(path))
            {
                error = "The bundled installer is missing: " + path;
                return false;
            }

            string actual;
            try
            {
                using (var stream = File.OpenRead(path))
                using (var sha = SHA256.Create())
                    actual = ToHex(sha.ComputeHash(stream));
            }
            catch (Exception ex)
            {
                error = "Could not read the bundled installer " + path + ": " + ex.Message;
                return false;
            }

            if (string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
                return true;

            long size = new FileInfo(path).Length;
            error = $"The bundled installer {Path.GetFileName(path)} is damaged or incomplete ({size} bytes, SHA-256 {actual}, " +
                    $"expected {expectedSha256}); it was not run. Re-download the repository (if it was cloned with Git LFS, " +
                    "run 'git lfs pull') or install Python 3.13 yourself.";
            return false;
        }

        private static ChildResult RunMacPkgInstallerWithAdminPrompt(SetupContext ctx, string pkgPath)
        {
            string tempPkgPath = Path.Combine(Path.GetTempPath(), $"bo4unity_python_{Guid.NewGuid():N}.pkg");
            string tempScriptPath = Path.Combine(Path.GetTempPath(), $"bo4unity_install_{Guid.NewGuid():N}.applescript");
            try
            {
                // Copy the payload to a neutral temp location to avoid Desktop/Documents access restrictions,
                // and verify the copy that actually runs.
                File.Copy(pkgPath, tempPkgPath, true);
                if (!VerifyInstaller(tempPkgPath, MacPythonPkgSha256, out string copyError))
                    return new ChildResult { FailedToStart = true, StartError = copyError };

                string escapedTempPkgPath = tempPkgPath.Replace("\\", "\\\\").Replace("\"", "\\\"");
                string applescript =
                    $"do shell script \"cd / && /usr/sbin/installer -pkg \\\"{escapedTempPkgPath}\\\" -target /\" with administrator privileges";
                File.WriteAllText(tempScriptPath, applescript);
                return RunChild(ctx, "/usr/bin/osascript", QuoteArgument(tempScriptPath), BundledInstallTimeout, new ChildOptions(), "/");
            }
            catch (Exception ex)
            {
                return new ChildResult { FailedToStart = true, StartError = ex.Message };
            }
            finally
            {
                TryDelete(tempScriptPath);
                TryDelete(tempPkgPath);
            }
        }

        // ── Child processes ────────────────────────────────────────────────────────────────────

        private static ChildResult RunPython(SetupContext ctx, string interpreter, string arguments, TimeSpan timeout, ChildOptions options)
        {
            ProcessStartInfo template = CreatePythonStartInfo(interpreter, arguments, ctx.UseArm64Wrapper);
            return RunChild(ctx, template.FileName, template.Arguments, timeout, options ?? new ChildOptions(), null, pythonChild: true);
        }

        /// <summary>
        /// Runs a child process with both pipes drained concurrently, a timeout, cancellation (the process is killed),
        /// optional streaming into pip-install.log and a bounded tail of its output for error messages.
        /// </summary>
        private static ChildResult RunChild(SetupContext ctx, string fileName, string arguments, TimeSpan timeout,
            ChildOptions options, string workingDirectory, bool pythonChild = false)
        {
            var result = new ChildResult();
            TimeSpan remaining = ctx.DeadlineUtc == default ? timeout : ctx.DeadlineUtc - DateTime.UtcNow;
            TimeSpan limit = remaining < timeout ? remaining : timeout;
            if (limit <= TimeSpan.Zero)
            {
                result.TimedOut = true;
                return result;
            }
            if (ctx.Token.IsCancellationRequested)
            {
                result.Cancelled = true;
                return result;
            }

            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                WorkingDirectory = string.IsNullOrEmpty(workingDirectory) ? ctx.WorkingDirectory : workingDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            if (pythonChild)
            {
                psi.Environment["PYTHONIOENCODING"] = "utf-8";
                psi.Environment["PYTHONUTF8"] = "1";
                psi.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
                psi.Environment["PIP_NO_INPUT"] = "1";
                psi.Environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1";
                ApplySitePolicy(psi, options.UserSite);
            }

            var sync = new object();
            var stdout = new StringBuilder();
            var stdoutClosed = new ManualResetEvent(false);
            var stderrClosed = new ManualResetEvent(false);

            void OnData(string line, bool isError)
            {
                if (line == null)
                {
                    (isError ? stderrClosed : stdoutClosed).Set();
                    return;
                }
                lock (sync)
                {
                    if (!isError && stdout.Length < MaxCapturedStdout)
                        stdout.Append(line).Append('\n');
                    AddToTail(result.Tail, line);
                    if (isError)
                        AddToTail(result.StderrTail, line);
                }
                if (options.LogToPipLog)
                    WritePipLog(ctx, line);
                try { options.OnLine?.Invoke(line); }
                catch (Exception) { /* progress display only */ }
            }

            if (options.LogToPipLog)
                WritePipLog(ctx, $"=== {DateTime.Now:yyyy-MM-dd HH:mm:ss}  {fileName} {arguments}");

            var process = new Process { StartInfo = psi };
            process.OutputDataReceived += (s, e) => OnData(e.Data, false);
            process.ErrorDataReceived += (s, e) => OnData(e.Data, true);
            try
            {
                process.Start();
            }
            catch (Exception ex)
            {
                result.FailedToStart = true;
                result.StartError = ex.Message;
                if (options.LogToPipLog)
                    WritePipLog(ctx, "Could not start: " + ex.Message);
                process.Dispose();
                return result;
            }

            RegisterChild(process);
            if (options.TrackInLockFile)
                RecordLockChild(ctx, process);
            try
            {
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                var clock = Stopwatch.StartNew();
                bool exited = false;
                while (true)
                {
                    if (process.WaitForExit(200))
                    {
                        exited = true;
                        break;
                    }
                    if (ctx.Token.IsCancellationRequested)
                    {
                        result.Cancelled = true;
                        break;
                    }
                    if (clock.Elapsed > limit)
                    {
                        result.TimedOut = true;
                        break;
                    }
                }

                // Shutdown cancels the token and then kills the tracked children: a child that died that way
                // was cancelled, not failed.
                if (exited && ctx.Token.IsCancellationRequested)
                    result.Cancelled = true;

                if (!exited)
                {
                    TryKill(process);
                    try { process.WaitForExit(5000); } catch (Exception) { }
                    if (options.LogToPipLog)
                        WritePipLog(ctx, result.TimedOut ? $"Stopped after {limit.TotalMinutes:0.#} minutes (time limit)." : "Stopped (setup cancelled).");
                }

                // Give the asynchronous readers a moment to deliver the last lines.
                stdoutClosed.WaitOne(2000);
                stderrClosed.WaitOne(2000);
                if (exited)
                {
                    try { result.ExitCode = process.ExitCode; }
                    catch (Exception) { result.ExitCode = -1; }
                    if (options.LogToPipLog)
                        WritePipLog(ctx, "Exit code: " + result.ExitCode.ToString(CultureInfo.InvariantCulture));
                }
            }
            finally
            {
                UnregisterChild(process);
                if (options.TrackInLockFile)
                    RecordLockChild(ctx, null);
                lock (sync)
                    result.StdOut = stdout.ToString();
                try { process.Dispose(); } catch (Exception) { }
            }
            return result;
        }

        private static void AddToTail(List<string> tail, string line)
        {
            tail.Add(line);
            if (tail.Count > TailLines)
                tail.RemoveAt(0);
        }

        private static void RegisterChild(Process process)
        {
            lock (s_childGate)
                s_children.Add(process);
        }

        private static void UnregisterChild(Process process)
        {
            lock (s_childGate)
                s_children.Remove(process);
        }

        private static int KillTrackedChildren()
        {
            Process[] children;
            lock (s_childGate)
                children = s_children.ToArray();
            int killed = 0;
            foreach (Process child in children)
            {
                if (TryKill(child))
                    killed++;
            }
            return killed;
        }

        private static bool TryKill(Process process)
        {
            try
            {
                if (process.HasExited)
                    return false;
                process.Kill();
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static string DescribeChildFailure(ChildResult result, string what)
        {
            if (result.FailedToStart)
                return $"{what} could not be started: {result.StartError}";
            if (result.TimedOut)
                return $"{what} did not finish within the time limit and was stopped";
            if (result.Cancelled)
                return $"{what} was cancelled";

            string decisive = LastMatchingLine(result.StderrTail) ?? LastMatchingLine(result.Tail);
            string last = decisive ?? LastNonEmptyLine(result.StderrTail) ?? LastNonEmptyLine(result.Tail);
            return $"{what} exited with code {result.ExitCode}" + (last != null ? ": " + Shorten(last, 300) : string.Empty);
        }

        private static string LastMatchingLine(List<string> lines)
        {
            for (int i = lines.Count - 1; i >= 0; i--)
            {
                string line = lines[i].Trim();
                if (line.StartsWith("ERROR:", StringComparison.Ordinal) || line.StartsWith("error:", StringComparison.Ordinal) ||
                    line.IndexOf("Error:", StringComparison.Ordinal) >= 0)
                    return line;
            }
            return null;
        }

        private static string LastNonEmptyLine(List<string> lines)
        {
            for (int i = lines.Count - 1; i >= 0; i--)
            {
                if (!string.IsNullOrWhiteSpace(lines[i]))
                    return lines[i].Trim();
            }
            return null;
        }

        // The deciding pip error is at the END of its output: log the tail, not the head.
        private static void LogChildFailure(SetupContext ctx, string headline, ChildResult result)
        {
            Debug.LogError(
                headline + "\n--- last lines of output ---\n" + string.Join("\n", result.Tail) +
                (string.IsNullOrEmpty(ctx.PipLogPath) ? string.Empty : "\n--- full output: " + ctx.PipLogPath));
        }

        // ── pip-install.log ────────────────────────────────────────────────────────────────────

        private static void OpenPipLog(SetupContext ctx)
        {
            lock (ctx.PipLogLock)
            {
                if (ctx.PipLog != null)
                    return;
                try
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(ctx.PipLogPath));
                    // One file per setup run (all interpreters tried in it), replacing the previous run's log.
                    ctx.PipLog = new StreamWriter(ctx.PipLogPath, false, new UTF8Encoding(false)) { AutoFlush = true };
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("Could not open " + ctx.PipLogPath + ": " + ex.Message);
                }
            }
        }

        private static void WritePipLog(SetupContext ctx, string line)
        {
            lock (ctx.PipLogLock)
            {
                if (ctx.PipLog == null)
                    return;
                try { ctx.PipLog.WriteLine(line); }
                catch (Exception) { /* the Console keeps the tail */ }
            }
        }

        private static void ClosePipLog(SetupContext ctx)
        {
            lock (ctx.PipLogLock)
            {
                try { ctx.PipLog?.Dispose(); } catch (Exception) { }
                ctx.PipLog = null;
            }
        }

        // ── Install lock (persistentDataPath/BOData/Installation/install.lock) ─────────────────

        private sealed class InstallLock : IDisposable
        {
            private readonly SetupContext _ctx;
            private readonly bool _owner;

            public InstallLock(SetupContext ctx, bool owner = true)
            {
                _ctx = ctx;
                _owner = owner;
            }

            public void Dispose()
            {
                // A nested acquisition (the bundled install holds the lock while the environment is prepared)
                // must not release its caller's lock.
                if (!_owner)
                    return;
                TryDelete(_ctx.LockPath);
                _ctx.LockPath = null;
            }
        }

        private static InstallLock AcquireInstallLock(SetupContext ctx)
        {
            // Already held by this setup: re-acquiring would find our own lock, treat it as stale and delete it.
            if (!string.IsNullOrEmpty(ctx.LockPath))
                return new InstallLock(ctx, owner: false);

            string lockPath = Path.Combine(ctx.StateDir, "install.lock");
            GetSelfIdentity(out int selfId, out string selfName);
            bool announced = false;
            int unexplainedFailures = 0;

            while (!ctx.Token.IsCancellationRequested && DateTime.UtcNow < ctx.DeadlineUtc)
            {
                try
                {
                    Directory.CreateDirectory(ctx.StateDir);
                    using (var stream = new FileStream(lockPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
                    using (var writer = new StreamWriter(stream))
                        writer.Write(FormatLock(selfId, selfName, null));
                    ctx.LockPath = lockPath;
                    return new InstallLock(ctx);
                }
                catch (IOException ex)
                {
                    if (!File.Exists(lockPath))
                    {
                        // Deleted in between, or an I/O problem unrelated to the lock.
                        if (++unexplainedFailures > 20)
                        {
                            Debug.LogWarning("Could not create the install lock " + lockPath + ": " + ex.Message);
                            return new InstallLock(ctx);
                        }
                        ctx.Token.WaitHandle.WaitOne(100);
                        continue;
                    }

                    Dictionary<string, string> info = ReadLock(lockPath);
                    int owner = ParseInt(info, "owner");
                    if (owner <= 0 && IsRecentlyWritten(lockPath))
                    {
                        // Empty or partial: its owner is rewriting it (RecordLockChild). Not stale; read it again.
                        ctx.Token.WaitHandle.WaitOne(200);
                        continue;
                    }
                    if (owner > 0 && owner != selfId && IsProcessAlive(owner, Get(info, "ownerName")))
                    {
                        if (!announced)
                        {
                            announced = true;
                            SetStatus($"Waiting for another Unity instance (process {owner}) to finish installing the Python dependencies…");
                            Debug.Log($"Another process ({owner}) is installing the Python dependencies for this project; waiting for it.");
                        }
                        ctx.Token.WaitHandle.WaitOne(1000);
                        continue;
                    }

                    // Stale: its owner is gone, or it is this process before a script reload. Stop the install it
                    // may have left running, so two pip processes never write the same environment.
                    int child = ParseInt(info, "child");
                    if (child > 0 && IsProcessAlive(child, Get(info, "childName")))
                    {
                        try
                        {
                            using (Process orphan = Process.GetProcessById(child))
                                orphan.Kill();
                            Debug.LogWarning($"Stopped a Python dependency install (process {child}) left running by an earlier session.");
                        }
                        catch (Exception killError)
                        {
                            Debug.LogWarning($"Could not stop the earlier dependency install (process {child}): {killError.Message}");
                        }
                    }
                    if (!TryDelete(lockPath))
                        ctx.Token.WaitHandle.WaitOne(500);
                }
                catch (Exception ex)
                {
                    // Unwritable state folder: continue without cross-process protection rather than fail the setup.
                    Debug.LogWarning("Could not create the install lock " + lockPath + ": " + ex.Message);
                    return new InstallLock(ctx);
                }
            }
            return null;
        }

        private static bool IsRecentlyWritten(string path)
        {
            try
            {
                return DateTime.UtcNow - File.GetLastWriteTimeUtc(path) < TimeSpan.FromSeconds(10);
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static void RecordLockChild(SetupContext ctx, Process child)
        {
            string lockPath = ctx.LockPath;
            if (string.IsNullOrEmpty(lockPath))
                return;
            try
            {
                GetSelfIdentity(out int selfId, out string selfName);
                File.WriteAllText(lockPath, FormatLock(selfId, selfName, child));
            }
            catch (Exception)
            {
                // Best effort: the lock still guards; only orphan clean-up loses its target.
            }
        }

        private static string FormatLock(int ownerId, string ownerName, Process child)
        {
            var sb = new StringBuilder();
            sb.Append("owner=").Append(ownerId.ToString(CultureInfo.InvariantCulture)).Append('\n');
            sb.Append("ownerName=").Append(ownerName).Append('\n');
            if (child != null)
            {
                try
                {
                    sb.Append("child=").Append(child.Id.ToString(CultureInfo.InvariantCulture)).Append('\n');
                    sb.Append("childName=").Append(SafeProcessName(child)).Append('\n');
                }
                catch (Exception)
                {
                    // Child already gone.
                }
            }
            return sb.ToString();
        }

        private static Dictionary<string, string> ReadLock(string lockPath)
        {
            var info = new Dictionary<string, string>(StringComparer.Ordinal);
            string text = ReadTextOrNull(lockPath);
            if (text == null)
                return info;
            foreach (string line in text.Split('\n'))
            {
                int eq = line.IndexOf('=');
                if (eq > 0)
                    info[line.Substring(0, eq).Trim()] = line.Substring(eq + 1).Trim();
            }
            return info;
        }

        private static string Get(Dictionary<string, string> info, string key)
        {
            return info.TryGetValue(key, out string value) ? value : null;
        }

        private static int ParseInt(Dictionary<string, string> info, string key)
        {
            return int.TryParse(Get(info, key), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) ? value : -1;
        }

        private static bool IsProcessAlive(int pid, string expectedName)
        {
            try
            {
                using (Process process = Process.GetProcessById(pid))
                {
                    // A recycled process id belongs to some other program: compare the name recorded with the lock.
                    return string.IsNullOrEmpty(expectedName) ||
                           string.Equals(SafeProcessName(process), expectedName, StringComparison.OrdinalIgnoreCase);
                }
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static void GetSelfIdentity(out int id, out string name)
        {
            using (Process self = Process.GetCurrentProcess())
            {
                id = self.Id;
                name = SafeProcessName(self);
            }
        }

        private static string SafeProcessName(Process process)
        {
            try { return process.ProcessName ?? string.Empty; }
            catch (Exception) { return string.Empty; }
        }

        // ── Small helpers ──────────────────────────────────────────────────────────────────────

        private static void SetStatus(string status)
        {
            s_status = status;
        }

        private static HostOs DetectHostOs()
        {
            // Decided at runtime: in the Editor the UNITY_STANDALONE_* defines follow the build target, not the OS
            // the Editor runs on (a macOS Editor targeting Windows would otherwise take the Windows code path).
            switch (SystemInfo.operatingSystemFamily)
            {
                case OperatingSystemFamily.Windows: return HostOs.Windows;
                case OperatingSystemFamily.MacOSX: return HostOs.MacOS;
                case OperatingSystemFamily.Linux: return HostOs.Linux;
            }
            switch (Application.platform)
            {
                case RuntimePlatform.WindowsEditor:
                case RuntimePlatform.WindowsPlayer:
                    return HostOs.Windows;
                case RuntimePlatform.OSXEditor:
                case RuntimePlatform.OSXPlayer:
                    return HostOs.MacOS;
                case RuntimePlatform.LinuxEditor:
                case RuntimePlatform.LinuxPlayer:
                    return HostOs.Linux;
                default:
                    return HostOs.Unknown;
            }
        }

        // Quotes one argument for ProcessStartInfo.Arguments (CommandLineToArgvW rules, which .NET/Mono also apply
        // when splitting the string on macOS/Linux): backslashes are doubled only before a quote.
        private static string QuoteArgument(string value)
        {
            if (string.IsNullOrEmpty(value))
                return "\"\"";
            var sb = new StringBuilder("\"");
            int backslashes = 0;
            foreach (char c in value)
            {
                if (c == '\\')
                {
                    backslashes++;
                    continue;
                }
                if (c == '"')
                {
                    sb.Append('\\', backslashes * 2 + 1).Append('"');
                    backslashes = 0;
                    continue;
                }
                sb.Append('\\', backslashes).Append(c);
                backslashes = 0;
            }
            sb.Append('\\', backslashes * 2).Append('"');
            return sb.ToString();
        }

        private static string NormalizePathKey(string path, HostOs os)
        {
            string full;
            try { full = Path.GetFullPath(path); }
            catch (Exception) { full = path ?? string.Empty; }
            return os == HostOs.Windows ? full.ToLowerInvariant() : full;
        }

        private static string ShortHash(string text)
        {
            using (var sha = SHA256.Create())
                return ToHex(sha.ComputeHash(Encoding.UTF8.GetBytes(text ?? string.Empty))).Substring(0, 12);
        }

        private static string ToHex(byte[] bytes)
        {
            var sb = new StringBuilder(bytes.Length * 2);
            foreach (byte b in bytes)
                sb.Append(b.ToString("x2", CultureInfo.InvariantCulture));
            return sb.ToString();
        }

        private static string ReadTextOrNull(string path)
        {
            try
            {
                return File.Exists(path) ? File.ReadAllText(path) : null;
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static bool TryDelete(string path)
        {
            if (string.IsNullOrEmpty(path))
                return true;
            try
            {
                if (File.Exists(path))
                    File.Delete(path);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static string Shorten(string text, int max)
        {
            if (string.IsNullOrEmpty(text))
                return string.Empty;
            text = text.Trim();
            return text.Length <= max ? text : text.Substring(0, max) + " …";
        }

        private string GetSafeWorkingDirectory()
        {
            try
            {
                string appDataPath = Application.dataPath;
                if (!string.IsNullOrWhiteSpace(appDataPath) && Directory.Exists(appDataPath))
                    return appDataPath;
            }
            catch (Exception)
            {
                // Fallback below.
            }

            string userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (!string.IsNullOrWhiteSpace(userHome) && Directory.Exists(userHome))
                return userHome;

            string tempPath = Path.GetTempPath();
            if (!string.IsNullOrWhiteSpace(tempPath) && Directory.Exists(tempPath))
                return tempPath;

            return s_hostOs == HostOs.Windows ? Environment.SystemDirectory : "/";
        }

        // ── Configuration validation (unchanged rules) ─────────────────────────────────────────

        private static int CountDistinctValidParameterKeys(IList<BOforUnity.ParameterEntry> parameters)
        {
            return GetDistinctValidParameterKeys(parameters).Count;
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
            return GetDistinctValidObjectiveKeys(objectives).Count;
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

        /// <summary>
        /// Fail-fast validation of the contextual-optimization configuration so
        /// obvious mistakes surface before the Python process is launched. The
        /// full validation (embedding lengths etc.) runs in SocketNetwork when
        /// the init message is built.
        /// </summary>
        private static bool TryValidateContextualConfiguration(
            BOforUnity.BoForUnityManager manager,
            out string error)
        {
            error = null;
            if (manager == null || !manager.contextualOptimization)
                return true;

            if (manager.optimizerBackend != BOforUnity.BoForUnityManager.OptimizerBackend.BoTorch)
            {
                // Applies to every non-BoTorch backend (CABOP, MetaTAF, future additions):
                // the LCE-M context pipeline lives in the BoTorch backends only.
                error = "Contextual optimization is only supported with the BoTorch backend.";
                return false;
            }

            if (manager.contexts == null || manager.contexts.Count == 0)
            {
                error = "Contextual optimization is enabled, but no contexts are configured.";
                return false;
            }

            string currentKey = (manager.currentContextKey ?? string.Empty).Trim();
            if (string.IsNullOrEmpty(currentKey))
            {
                error = "Contextual optimization is enabled, but no Current Context Key is set.";
                return false;
            }

            bool currentKeyFound = false;
            foreach (var context in manager.contexts)
            {
                if (context == null || string.IsNullOrWhiteSpace(context.key))
                {
                    error = "Every context entry needs a non-empty key.";
                    return false;
                }
                if (string.Equals(context.key.Trim(), currentKey, StringComparison.OrdinalIgnoreCase))
                    currentKeyFound = true;
            }

            if (!currentKeyFound)
            {
                error = $"Current Context Key '{currentKey}' does not match any configured context.";
                return false;
            }

            return true;
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

            if (manager.optimizerBackend == BOforUnity.BoForUnityManager.OptimizerBackend.MetaTAF)
            {
                if (effectiveObjectiveCount < 2)
                {
                    error =
                        "The MetaTAF backend is multi-objective: configure at least two objectives " +
                        "(use the BoTorch backend for single-objective studies).";
                    return false;
                }

                if (manager.warmStart)
                {
                    error =
                        "The MetaTAF backend does not support Warm Start: population models are its " +
                        "transfer mechanism. Disable Warm Start or use the BoTorch backend.";
                    return false;
                }

                scriptName = "meta_mobo.py";
                return true;
            }

            if (manager.optimizerBackend == BOforUnity.BoForUnityManager.OptimizerBackend.DBO)
            {
                if (effectiveObjectiveCount != 1)
                {
                    error =
                        "The DBO backend is single-objective: configure exactly one objective " +
                        "(the temporal-decay GP models one drifting cost).";
                    return false;
                }

                if (!manager.dboStationaryBaseline &&
                    !(manager.dboInitialAlpha > 0f && manager.dboInitialAlpha < 1f))
                {
                    error =
                        $"DBO Initial Alpha must lie strictly between 0 and 1 (got {manager.dboInitialAlpha}): " +
                        "fitting can never move alpha away from exactly 1, so the run would be stationary " +
                        "BO labelled as DBO. Use 0.99, or enable DBO Stationary Baseline for a stationary control.";
                    return false;
                }

                scriptName = "dbo.py";
                return true;
            }

            error = "Unsupported optimizer backend/objective mode combination.";
            return false;
        }
    }
}
