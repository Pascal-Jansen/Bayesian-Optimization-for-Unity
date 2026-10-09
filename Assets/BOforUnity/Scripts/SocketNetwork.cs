// SocketNetwork.cs
// Unity <-> Python NDJSON protocol using Newtonsoft.Json.
// Place Newtonsoft.Json source under Assets/<YourAsset>/ThirdParty/Newtonsoft.Json with an .asmdef.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

namespace BOforUnity.Scripts
{
    // -------------------- JSON DTOs --------------------
    [Serializable] class MsgBase { public string type; }

    [Serializable] class InitMsg : MsgBase
    {
        public InitConfig config;
        public List<ParamInfo> parameters;
        public List<ObjInfo> objectives;
        public List<CabopGroupCostInfo> cabopGroupCosts;
        public ContextConfigInfo context;
        public UserInfo user;
    }

    [Serializable] class InitConfig
    {
        public int batchSize, numRestarts, rawSamples, numOptimizationIterations, mcSamples, numSamplingIterations, seed;
        public int nParameters, nObjectives;
        public bool warmStart;
        public string initialParametersDataPath, initialObjectivesDataPath, warmStartObjectiveFormat;
        public string optimizerBackend, cabopObjectiveMode, cabopUpdateRule;
        public bool cabopUseCostAwareAcquisition, cabopEnableCostBudget;
        public float cabopMaxCumulativeCost;
        // Meta-TAF backend settings (ignored by the other backends).
        public string metaSourceDir, metaWeightMode;
        public bool metaRequireSources;
        public float metaRho, metaTargetWeight, metaDecayRate;
        public int metaWarmupIters, metaDecayStartIter;
        // DBO backend settings (ignored by the other backends).
        public string dboSpatialKernel, dboAlphaParameterization;
        public float dboInitialAlpha, dboAcquisitionTimeOffset, dboValidationConfidence;
        public int dboValidationEvery;
        public bool dboValidationVisitedOnly, dboStationaryBaseline;
        public float explorationRatio;
    }

    // Bounds stay float, like the inspector fields and the objective values sent later:
    // widening 0.7f to double serializes it as 0.699999988079071 while the value clamped to
    // that bound is sent as 0.7, which the backends then reject as out of bounds.
    [Serializable] class ParamInit { public float low; public float high; }
    [Serializable] class ObjInit   { public float low; public float high; public int minimize; }

    [Serializable] class CabopCostTripletInfo
    {
        public float unchanged;
        public float swapped;
        public float acquired;
    }

    [Serializable] class CabopGroupCostInfo
    {
        public string group;
        public CabopCostTripletInfo cost;
        public CabopCostTripletInfo actualCost;
    }

    [Serializable] class ParamInfo
    {
        public string key;
        public ParamInit init;
        public int optSeqOrder;
        public string group;
        public float tolerance;
        public List<float> prefabValues;
    }

    [Serializable] class ObjInfo
    {
        public string key;
        public ObjInit init;
        public int optSeqOrder;
        public float weight;
    }

    [Serializable] class ContextInfo
    {
        public string key;
        public List<float> embedding;
        public string imagePath;
    }

    [Serializable] class ContextConfigInfo
    {
        public bool enabled;
        public string currentContext;
        public string embeddingSource;
        public bool normalizeEmbeddings;
        public string imageEmbeddingModel;
        public string imageEmbeddingPretrained;
        public List<ContextInfo> contexts;
    }

    [Serializable] class UserInfo
    {
        public string userId, conditionId, groupId;
    }

    [Serializable] class ParametersMsg : MsgBase
    {
        public Dictionary<string, float> values;
        // Global log iteration (the Iteration column of ObservationsPerEvaluation.csv) the suggested design
        // will be logged under; absent from older backends.
        public int? iteration;
    }

    [Serializable] class ObjectivesMsg : MsgBase
    {
        public Dictionary<string, float> values;
    }

    [Serializable] class CoverageMsg : MsgBase
    {
        public float value;
    }

    // Unity -> Python: end the session cleanly (the backend logs the reason and exits).
    [Serializable] class StopMsg : MsgBase
    {
        public string reason;
    }

    // -------------------- SocketNetwork --------------------
    public class SocketNetwork : MonoBehaviour
    {
        private Socket _serverSocket;
        private IPAddress _ip;
        private IPEndPoint _ipEnd;
        private Thread _connectThread;
        private volatile bool _stopRequested;
        private volatile bool _stopMessageSent;
        private volatile bool _connectionClosedByPeer;
        private volatile bool _optimizationFinished;

        public float coverage = 0f;
        public float tempCoverage = 0f;

        private BoForUnityManager _bomanager;

        // TCP buffer for NDJSON framing
        private readonly byte[] _recvBuf = new byte[4096];
        private readonly char[] _charBuf = new char[Encoding.UTF8.GetMaxCharCount(4096)];
        private readonly StringBuilder _lineBuf = new StringBuilder(4096);
        // One decoder per connection: it keeps the bytes of a multi-byte character split across two reads,
        // which Encoding.UTF8.GetString per read turned into U+FFFD replacement characters (a key such as
        // "Gr\u00f6\u00dfe" then no longer matched).
        private Decoder _utf8Decoder;

        // Python's receive limit for one NDJSON line (BO_MAX_RECV_BUF_BYTES, default 64 MiB). A larger init message
        // would only fail in Python as a "possible framing error".
        private const long DefaultMaxMessageBytes = 64L * 1024 * 1024;
        // How long SocketQuit lets Python exit by itself after a stop message or optimization_finished before it is
        // terminated.
        private const int StopGracePeriodMs = 1000;
        private const string ConnectionFailedText =
            "Optimizer connection failed.\nCheck parameter/objective configuration and Python logs, then restart.";

        // JSON settings. Used through one serializer created from them (JsonSerializer.Create ignores
        // JsonConvert.DefaultSettings), so a host project's global defaults - e.g. a camelCase resolver that
        // lowercases the keys Python expects, or MissingMemberHandling.Error, which made every incoming message
        // throw - cannot change the protocol.
        private static readonly JsonSerializerSettings JsonSettings = new JsonSerializerSettings
        {
            ContractResolver = new DefaultContractResolver(),
            MissingMemberHandling = MissingMemberHandling.Ignore,
            NullValueHandling = NullValueHandling.Include,
            DefaultValueHandling = DefaultValueHandling.Include,
            TypeNameHandling = TypeNameHandling.None,
            PreserveReferencesHandling = PreserveReferencesHandling.None,
            MetadataPropertyHandling = MetadataPropertyHandling.Ignore,
            DateParseHandling = DateParseHandling.None,
            FloatParseHandling = FloatParseHandling.Double,
            // Floats are always written invariantly with "R" (round-trip): 0.7f is sent as 0.7.
            Culture = CultureInfo.InvariantCulture,
            Formatting = Formatting.None
        };

        private static readonly JsonSerializer Json = JsonSerializer.Create(JsonSettings);

        // -------------------- Lifecycle --------------------
        /// <summary>
        /// Marks the connection as intentionally closing (play mode exit, application quit, early stop) so the
        /// receive thread treats Python going away as expected instead of reporting a failure. Call before the
        /// Python process is terminated.
        /// </summary>
        public void RequestStop()
        {
            _stopRequested = true;
        }

        public void InitSocket()
        {
            _bomanager = gameObject.GetComponent<BoForUnityManager>();
            _ip = IPAddress.Parse("127.0.0.1");
            _ipEnd = new IPEndPoint(_ip, 56001);

            _stopRequested = false;
            _stopMessageSent = false;
            _connectionClosedByPeer = false;
            _optimizationFinished = false;
            _lineBuf.Length = 0;
            _utf8Decoder = new UTF8Encoding(false).GetDecoder();
            _connectThread = new Thread(SocketReceive) { IsBackground = true };
            _connectThread.Start();
        }

        private void OnEnable()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
#endif
        }

        private void OnDisable()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
#endif
        }

#if UNITY_EDITOR
        private void OnPlayModeStateChanged(UnityEditor.PlayModeStateChange state)
        {
            // PythonStarter kills Python here, before OnDestroy runs; without the flag the receive thread would
            // report the closed connection as a crash.
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
                RequestStop();
        }
#endif

        private void OnApplicationQuit()
        {
            RequestStop();
        }

        private void OnDestroy()
        {
            try { SocketQuit(); } catch { }
        }

        // -------------------- Socket loop --------------------
        private void SocketReceive()
        {
            try
            {
                SocketConnect();
                try
                {
                    SendInitInfo();
                }
                catch (InvalidOperationException ex)
                {
                    // Invalid configuration or an oversized message: nothing was sent yet, so let Python exit
                    // cleanly instead of failing on a dropped connection, and show the reason on screen.
                    Debug.LogError("Optimizer start refused: " + ex.Message);
                    TrySendStop("invalid_configuration");
                    _stopRequested = true;
                    string message = ex.Message;
                    MainThreadDispatcher.Execute(() => OnSocketConnectionFailed(message));
                    CloseSocket();
                    return;
                }

                while (!_stopRequested)
                {
                    int recvLen = _serverSocket.Receive(_recvBuf);
                    if (recvLen == 0)
                    {
                        _connectionClosedByPeer = true;

                        if (_stopRequested)
                        {
                            Debug.Log("Socket connection closed.");
                        }
                        else if (_optimizationFinished)
                        {
                            Debug.Log("Python optimization process closed the connection. Optimization iterations have finished successfully.");
                        }
                        else
                        {
                            Debug.LogError("Socket closed by Python unexpectedly before optimization completed.");
                            MainThreadDispatcher.Execute(() => OnSocketConnectionFailed());
                        }

                        _stopRequested = true;
                        CloseSocket();
                        break;
                    }

                    int charCount = _utf8Decoder.GetChars(_recvBuf, 0, recvLen, _charBuf, 0, false);
                    _lineBuf.Append(_charBuf, 0, charCount);

                    int newlineIndex;
                    bool protocolError = false;
                    while (!_stopRequested && (newlineIndex = IndexOfChar(_lineBuf, '\n')) >= 0)
                    {
                        string line = _lineBuf.ToString(0, newlineIndex).TrimEnd('\r');
                        _lineBuf.Remove(0, newlineIndex + 1);
                        if (string.IsNullOrWhiteSpace(line)) continue;

                        try
                        {
                            ParseJsonMessage(line);
                        }
                        catch (Exception ex)
                        {
                            Debug.LogError($"Error in ParseJsonMessage: {ex.Message}\n{ex.StackTrace}\nPayload: {line}");
                            protocolError = true;
                            _stopRequested = true;
                            MainThreadDispatcher.Execute(() => OnSocketConnectionFailed());
                            CloseSocket();
                            break;
                        }
                    }

                    if (protocolError)
                    {
                        break;
                    }
                }
            }
            catch (SocketException ex)
            {
                if (_stopRequested || _connectionClosedByPeer || _optimizationFinished)
                {
                    Debug.Log("Socket connection closed.");
                }
                else
                {
                    Debug.LogError($"SocketReceive SocketException: {ex.SocketErrorCode} {ex.Message}");
                    MainThreadDispatcher.Execute(() => OnSocketConnectionFailed());
                }

                _stopRequested = true;
                CloseSocket();
            }
            catch (Exception ex)
            {
                if (_stopRequested || _connectionClosedByPeer)
                {
                    Debug.Log("Socket connection closed.");
                }
                else if (ex is ThreadAbortException)
                {
                    Debug.Log("Socket receive thread was aborted.");
                }
                else
                {
                    Debug.LogError($"Error in SocketReceive: {ex.Message}\n{ex.StackTrace}");
                    MainThreadDispatcher.Execute(() => OnSocketConnectionFailed());
                }

                _stopRequested = true;
                CloseSocket();
            }
        }

        private void CloseSocket()
        {
            try { _serverSocket?.Shutdown(SocketShutdown.Both); } catch { }
            try { _serverSocket?.Close(); } catch { }
        }

        private static int IndexOfChar(StringBuilder buffer, char value)
        {
            // Scan without materializing the whole buffer as a string each pass.
            for (int i = 0; i < buffer.Length; i++)
            {
                if (buffer[i] == value)
                    return i;
            }
            return -1;
        }

        private void SocketConnect()
        {
            _serverSocket?.Close();
            _serverSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            Debug.Log("Unity is ready to connect...");
            _serverSocket.Connect(_ipEnd);
        }

        // Main thread. Ends the study with the message on screen (status panel shown, loop terminated,
        // automatic advance cancelled, Python stopped) instead of leaving the participant on a blank screen.
        private void OnSocketConnectionFailed(string detail = null)
        {
            _bomanager = _bomanager != null ? _bomanager : gameObject.GetComponent<BoForUnityManager>();
            if (_bomanager == null)
            {
                Debug.LogError("Optimizer connection failed, but BoForUnityManager is missing.");
                SocketQuit();
                return;
            }

            _bomanager.TerminateWithError(
                string.IsNullOrWhiteSpace(detail)
                    ? ConnectionFailedText
                    : "The optimizer could not be started.\n" + detail
            );
            // TerminateWithError does nothing once the loop has ended; make sure the session is closed anyway.
            SocketQuit();
        }

        private static string SerializeJson(object value)
        {
            var builder = new StringBuilder(256);
            using (var stringWriter = new StringWriter(builder, CultureInfo.InvariantCulture))
            using (var jsonWriter = new JsonTextWriter(stringWriter) { Formatting = Formatting.None })
            {
                Json.Serialize(jsonWriter, value);
            }
            return builder.ToString();
        }

        private static T DeserializeJson<T>(string json)
        {
            using (var stringReader = new StringReader(json))
            using (var jsonReader = new JsonTextReader(stringReader))
            {
                return Json.Deserialize<T>(jsonReader);
            }
        }

        // -------------------- Protocol: incoming --------------------
        private void ParseJsonMessage(string json)
        {
            var peek = DeserializeJson<MsgBase>(json);
            if (peek == null || string.IsNullOrEmpty(peek.type))
            {
                throw new InvalidOperationException("Received protocol message without a valid 'type' field.");
            }

            switch (peek.type)
            {
                case "parameters":
                {
                    var msg = DeserializeJson<ParametersMsg>(json);
                    if (msg?.values == null)
                    {
                        throw new InvalidOperationException("Received 'parameters' message without a valid 'values' payload.");
                    }

                    MainThreadDispatcher.Execute(() =>
                    {
                        if (_stopRequested)
                            return; // the session was closed while this message was queued

                        _bomanager = gameObject.GetComponent<BoForUnityManager>();
                        if (_bomanager == null || _bomanager.parameters == null)
                        {
                            Debug.LogError(
                                "Cannot apply parameters because BoForUnityManager parameters are not configured. " +
                                "Terminating optimizer session to avoid backend deadlock."
                            );
                            OnSocketConnectionFailed();
                            return;
                        }

                        var incomingValues = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
                        var nonFiniteKeys = new List<string>();
                        foreach (var kv in msg.values)
                        {
                            if (string.IsNullOrWhiteSpace(kv.Key))
                                continue;

                            string key = kv.Key.Trim();
                            float value = kv.Value;
                            if (float.IsNaN(value) || float.IsInfinity(value))
                            {
                                nonFiniteKeys.Add(key);
                                continue;
                            }

                            incomingValues[key] = value;
                        }

                        if (nonFiniteKeys.Count > 0)
                        {
                            Debug.LogError(
                                "Received non-finite parameter values from Python for key(s): " +
                                string.Join(", ", nonFiniteKeys)
                            );
                            OnSocketConnectionFailed();
                            return;
                        }

                        var expectedParameters = new List<BOforUnity.ParameterEntry>();
                        var seenParameterKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                        foreach (var pa in _bomanager.parameters)
                        {
                            if (pa == null || pa.value == null || string.IsNullOrWhiteSpace(pa.key))
                                continue;

                            string expectedKey = pa.key.Trim();
                            if (!seenParameterKeys.Add(expectedKey))
                                continue;

                            expectedParameters.Add(pa);
                        }

                        var missingKeys = new List<string>();
                        var expectedKeySet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                        foreach (var pa in expectedParameters)
                        {
                            string expectedKey = pa.key.Trim();
                            expectedKeySet.Add(expectedKey);
                            if (!incomingValues.ContainsKey(expectedKey))
                            {
                                missingKeys.Add(expectedKey);
                            }
                        }

                        if (missingKeys.Count > 0)
                        {
                            Debug.LogError(
                                "Received incomplete parameter payload from Python. Missing key(s): " +
                                string.Join(", ", missingKeys)
                            );
                            OnSocketConnectionFailed();
                            return;
                        }

                        var unexpectedKeys = incomingValues.Keys
                            .Where(k => !expectedKeySet.Contains(k))
                            .OrderBy(k => k, StringComparer.OrdinalIgnoreCase)
                            .ToList();
                        if (unexpectedKeys.Count > 0)
                        {
                            Debug.LogError(
                                "Received parameter payload with unexpected key(s): " +
                                string.Join(", ", unexpectedKeys)
                            );
                            OnSocketConnectionFailed();
                            return;
                        }

                        // Apply values only after payload completeness has been validated.
                        foreach (var pa in expectedParameters)
                        {
                            string expectedKey = pa.key.Trim();
                            pa.value.Value = incomingValues[expectedKey];
                        }
                        _bomanager.LastSuggestionLogIteration = msg.iteration ?? -1;

                        // Notify lifecycle: triggers measurement and later SendObjectives()
                        if (_bomanager.initialized)
                            _bomanager.OptimizationDone();
                        else
                            _bomanager.InitializationDone();
                    });
                    break;
                }

                case "optimization_finished":
                {
                    // Set before dispatching: the handler may end the session (SocketQuit) before this thread
                    // continues, and SocketQuit grants the finished backend its grace period only with the flag set.
                    _optimizationFinished = true;
                    MainThreadDispatcher.Execute(() =>
                    {
                        _bomanager = gameObject.GetComponent<BoForUnityManager>();
                        if (_bomanager == null)
                        {
                            Debug.LogWarning("Received optimization_finished, but BoForUnityManager is missing.");
                            return;
                        }
                        _bomanager.OnOptimizationFinishedFromBackend();
                    });
                    break;
                }

                case "coverage":
                {
                    var msg = DeserializeJson<CoverageMsg>(json);
                    if (msg != null)
                    {
                        coverage = msg.value;
                        Debug.Log("coverage " + coverage.ToString("R", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        throw new InvalidOperationException("Received malformed 'coverage' message.");
                    }
                    break;
                }

                case "tempCoverage":
                {
                    var msg = DeserializeJson<CoverageMsg>(json);
                    if (msg != null)
                    {
                        tempCoverage = msg.value;
                        Debug.Log("tempCoverage " + tempCoverage.ToString("R", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        throw new InvalidOperationException("Received malformed 'tempCoverage' message.");
                    }
                    break;
                }

                case "objectives":
                {
                    throw new InvalidOperationException(
                        "Received unexpected 'objectives' message from backend. " +
                        "This message type is Unity->Python only."
                    );
                }

                default:
                    throw new InvalidOperationException($"Received unknown protocol message type '{peek.type}'.");
            }
        }

        // -------------------- Protocol: outgoing --------------------
        private void SendInitInfo()
        {
            _bomanager = _bomanager ?? gameObject.GetComponent<BoForUnityManager>();
            if (_bomanager == null)
            {
                throw new InvalidOperationException("Cannot send init message because BoForUnityManager is missing.");
            }

            var parameterPayload = new List<ParamInfo>();
            var objectivePayload = new List<ObjInfo>();
            var parameterGroups = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var seenParameterKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var seenObjectiveKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var invalidParameterEntries = new List<string>();
            var duplicateParameterKeys = new List<string>();
            var invalidObjectiveEntries = new List<string>();
            var duplicateObjectiveKeys = new List<string>();

            if (_bomanager.parameters != null)
            {
                for (int i = 0; i < _bomanager.parameters.Count; i++)
                {
                    var parameter = _bomanager.parameters[i];
                    if (parameter == null || parameter.value == null || string.IsNullOrWhiteSpace(parameter.key))
                    {
                        invalidParameterEntries.Add($"index {i}");
                        continue;
                    }

                    string key = parameter.key.Trim();
                    if (!seenParameterKeys.Add(key))
                    {
                        if (!duplicateParameterKeys.Contains(key, StringComparer.OrdinalIgnoreCase))
                            duplicateParameterKeys.Add(key);
                        continue;
                    }

                    parameter.value.optSeqOrder = parameterPayload.Count;
                    string cabopGroup = NormalizeCabopGroup(parameter.value.cabopGroup);
                    parameterGroups.Add(cabopGroup);
                    parameterPayload.Add(new ParamInfo
                    {
                        key = key,
                        init = new ParamInit
                        {
                            low = parameter.value.lowerBound,
                            high = parameter.value.upperBound
                        },
                        optSeqOrder = parameter.value.optSeqOrder,
                        group = cabopGroup,
                        // Sent unchanged: BoConfigValidator and the CABOP backend reject a tolerance outside
                        // [0, 1] instead of it being rewritten to "never reuse".
                        tolerance = parameter.value.cabopTolerance,
                        prefabValues = NormalizeCabopPrefabricatedValues(parameter.value.cabopPrefabricatedValues)
                    });
                }
            }

            if (_bomanager.objectives != null)
            {
                for (int i = 0; i < _bomanager.objectives.Count; i++)
                {
                    var objective = _bomanager.objectives[i];
                    if (objective == null || objective.value == null || string.IsNullOrWhiteSpace(objective.key))
                    {
                        invalidObjectiveEntries.Add($"index {i}");
                        continue;
                    }

                    string key = objective.key.Trim();
                    if (!seenObjectiveKeys.Add(key))
                    {
                        if (!duplicateObjectiveKeys.Contains(key, StringComparer.OrdinalIgnoreCase))
                            duplicateObjectiveKeys.Add(key);
                        continue;
                    }

                    objective.value.optSeqOrder = objectivePayload.Count;
                    objectivePayload.Add(new ObjInfo
                    {
                        key = key,
                        init = new ObjInit
                        {
                            low = objective.value.lowerBound,
                            high = objective.value.upperBound,
                            minimize = objective.value.smallerIsBetter ? 1 : 0
                        },
                        optSeqOrder = objective.value.optSeqOrder,
                        weight = NormalizeCabopObjectiveWeight(objective.value.cabopWeight)
                    });
                }
            }

            if (invalidParameterEntries.Count > 0 || duplicateParameterKeys.Count > 0 ||
                invalidObjectiveEntries.Count > 0 || duplicateObjectiveKeys.Count > 0)
            {
                var details = new List<string>();
                if (invalidParameterEntries.Count > 0)
                    details.Add("invalid parameter entries at " + string.Join(", ", invalidParameterEntries));
                if (duplicateParameterKeys.Count > 0)
                    details.Add("duplicate parameter key(s): " + string.Join(", ", duplicateParameterKeys));
                if (invalidObjectiveEntries.Count > 0)
                    details.Add("invalid objective entries at " + string.Join(", ", invalidObjectiveEntries));
                if (duplicateObjectiveKeys.Count > 0)
                    details.Add("duplicate objective key(s): " + string.Join(", ", duplicateObjectiveKeys));

                throw new InvalidOperationException(
                    "Cannot start optimization with invalid parameter/objective configuration. " +
                    string.Join("; ", details)
                );
            }

            if (parameterPayload.Count == 0 || objectivePayload.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Cannot send init message with empty effective payload. " +
                    $"parameters={parameterPayload.Count}, objectives={objectivePayload.Count}."
                );
            }

            if (_bomanager.optimizerBackend == BOforUnity.BoForUnityManager.OptimizerBackend.CABOP)
            {
                if (_bomanager.cabopObjectiveMode == BOforUnity.BoForUnityManager.CabopObjectiveMode.SingleObjective &&
                    objectivePayload.Count != 1)
                {
                    throw new InvalidOperationException(
                        "CABOP single-objective mode requires exactly one configured objective."
                    );
                }

                if (_bomanager.cabopObjectiveMode == BOforUnity.BoForUnityManager.CabopObjectiveMode.MultiObjectiveScalarized &&
                    objectivePayload.Count < 2)
                {
                    throw new InvalidOperationException(
                        "CABOP multi-objective mode requires at least two configured objectives."
                    );
                }
            }

            if (_bomanager.optimizerBackend == BOforUnity.BoForUnityManager.OptimizerBackend.MetaTAF)
            {
                if (objectivePayload.Count < 2)
                {
                    throw new InvalidOperationException(
                        "The MetaTAF backend is multi-objective: configure at least two objectives."
                    );
                }

                if (_bomanager.warmStart)
                {
                    throw new InvalidOperationException(
                        "The MetaTAF backend does not support Warm Start: population models are its " +
                        "transfer mechanism. Disable Warm Start or use the BoTorch backend."
                    );
                }
            }

            var overlappingKeys = parameterPayload
                .Select(p => p.key)
                .Intersect(objectivePayload.Select(o => o.key), StringComparer.OrdinalIgnoreCase)
                .OrderBy(k => k, StringComparer.OrdinalIgnoreCase)
                .ToList();
            if (overlappingKeys.Count > 0)
            {
                throw new InvalidOperationException(
                    "Parameter and objective keys must be distinct. Overlapping key(s): " +
                    string.Join(", ", overlappingKeys)
                );
            }

            var init = new InitMsg
            {
                type = "init",
                config = new InitConfig
                {
                    batchSize = _bomanager.batchSize,
                    numRestarts = _bomanager.numRestarts,
                    rawSamples = _bomanager.rawSamples,
                    numOptimizationIterations = _bomanager.numOptimizationIterations,
                    mcSamples = _bomanager.mcSamples,
                    numSamplingIterations = _bomanager.GetEffectiveSamplingIterations(),
                    seed = _bomanager.seed,
                    nParameters = parameterPayload.Count,
                    nObjectives = objectivePayload.Count,
                    warmStart = _bomanager.warmStart,
                    initialParametersDataPath = _bomanager.initialParametersDataPath,
                    initialObjectivesDataPath = _bomanager.initialObjectivesDataPath,
                    warmStartObjectiveFormat = NormalizeWarmStartObjectiveFormat(_bomanager.warmStartObjectiveFormat),
                    optimizerBackend = NormalizeOptimizerBackend(_bomanager.optimizerBackend),
                    cabopObjectiveMode = NormalizeCabopObjectiveMode(_bomanager.cabopObjectiveMode),
                    cabopUseCostAwareAcquisition = _bomanager.cabopUseCostAwareAcquisition,
                    cabopUpdateRule = NormalizeCabopUpdateRule(_bomanager.cabopUpdateRule),
                    cabopEnableCostBudget = _bomanager.cabopEnableCostBudget,
                    cabopMaxCumulativeCost = _bomanager.cabopMaxCumulativeCost,
                    metaSourceDir = _bomanager.metaSourceDir,
                    metaRequireSources = _bomanager.metaRequireSources,
                    metaWeightMode = NormalizeMetaWeightMode(_bomanager.metaWeightMode),
                    metaRho = _bomanager.metaRho,
                    metaTargetWeight = _bomanager.metaTargetWeight,
                    metaWarmupIters = _bomanager.metaWarmupIters,
                    metaDecayStartIter = _bomanager.metaDecayStartIter,
                    metaDecayRate = _bomanager.metaDecayRate,
                    dboSpatialKernel = NormalizeDboSpatialKernel(_bomanager.dboSpatialKernel),
                    dboAlphaParameterization = NormalizeDboAlphaParameterization(_bomanager.dboAlphaParameterization),
                    dboInitialAlpha = _bomanager.dboInitialAlpha,
                    dboAcquisitionTimeOffset = _bomanager.dboAcquisitionTimeOffset,
                    dboValidationEvery = _bomanager.dboValidationEvery,
                    dboValidationConfidence = _bomanager.dboValidationConfidence,
                    dboValidationVisitedOnly = _bomanager.dboValidationVisitedOnly,
                    dboStationaryBaseline = _bomanager.dboStationaryBaseline,
                    explorationRatio = _bomanager.dboExplorationRatio
                },
                parameters = parameterPayload,
                objectives = objectivePayload,
                cabopGroupCosts = BuildCabopGroupCosts(parameterGroups),
                context = BuildContextConfig(),
                user = new UserInfo
                {
                    userId = _bomanager.userId,
                    conditionId = _bomanager.conditionId,
                    groupId = _bomanager.groupId
                }
            };

            string json = SerializeJson(init);
            long maxBytes = GetMaxMessageBytes();
            long jsonBytes = Encoding.UTF8.GetByteCount(json) + 1L; // + NDJSON newline
            if (jsonBytes > maxBytes)
            {
                throw new InvalidOperationException(
                    $"The init message is {FormatMiB(jsonBytes)} MiB, more than the backend accepts " +
                    $"({FormatMiB(maxBytes)} MiB, environment variable BO_MAX_RECV_BUF_BYTES). " +
                    "Reduce the context embeddings or prefabricated values, or raise BO_MAX_RECV_BUF_BYTES."
                );
            }
            SocketSendLine(json);
        }

        // Mirrors the backend's limit: Python inherits Unity's environment.
        private static long GetMaxMessageBytes()
        {
            string configured = Environment.GetEnvironmentVariable("BO_MAX_RECV_BUF_BYTES");
            return long.TryParse(configured, NumberStyles.Integer, CultureInfo.InvariantCulture, out long value) && value > 0
                ? value
                : DefaultMaxMessageBytes;
        }

        private static string FormatMiB(long bytes)
        {
            return (bytes / (1024.0 * 1024.0)).ToString("0.#", CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// Builds and validates the contextual-optimization payload of the init
        /// message. Returns null when contextual optimization is disabled and
        /// throws with an actionable message when the configuration is invalid.
        /// </summary>
        private ContextConfigInfo BuildContextConfig()
        {
            if (_bomanager == null || !_bomanager.contextualOptimization)
                return null;

            if (_bomanager.optimizerBackend != BOforUnity.BoForUnityManager.OptimizerBackend.BoTorch)
            {
                // Applies to every non-BoTorch backend (CABOP, MetaTAF, future additions).
                throw new InvalidOperationException(
                    "Contextual optimization (LCE-M GP) is only supported with the BoTorch backend. " +
                    "Disable contextual optimization or switch the optimizer backend."
                );
            }

            var source = _bomanager.contextEmbeddingSource;
            var entries = _bomanager.contexts;
            if (entries == null || entries.Count == 0)
            {
                throw new InvalidOperationException(
                    "Contextual optimization is enabled, but no contexts are configured."
                );
            }

            var payload = new List<ContextInfo>();
            var seenKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            int embeddingDim = -1;
            for (int i = 0; i < entries.Count; i++)
            {
                var entry = entries[i];
                if (entry == null || string.IsNullOrWhiteSpace(entry.key))
                {
                    throw new InvalidOperationException(
                        $"Context entry at index {i} is invalid: every context needs a non-empty key."
                    );
                }

                string key = entry.key.Trim();
                if (!seenKeys.Add(key))
                {
                    throw new InvalidOperationException(
                        $"Duplicate context key '{key}'. Context keys must be unique."
                    );
                }

                var info = new ContextInfo { key = key };

                if (source == BOforUnity.BoForUnityManager.ContextEmbeddingSource.Manual)
                {
                    if (entry.embedding == null || entry.embedding.Count == 0)
                    {
                        throw new InvalidOperationException(
                            $"Context '{key}' has no embedding vector, but the embedding source is Manual."
                        );
                    }
                    if (embeddingDim < 0)
                    {
                        embeddingDim = entry.embedding.Count;
                    }
                    else if (entry.embedding.Count != embeddingDim)
                    {
                        throw new InvalidOperationException(
                            $"Context '{key}' embedding has {entry.embedding.Count} values; " +
                            $"expected {embeddingDim} (all context embeddings must have the same length)."
                        );
                    }
                    foreach (float v in entry.embedding)
                    {
                        if (float.IsNaN(v) || float.IsInfinity(v))
                        {
                            throw new InvalidOperationException(
                                $"Context '{key}' embedding contains a non-finite value."
                            );
                        }
                    }
                    info.embedding = new List<float>(entry.embedding);
                }
                else if (source == BOforUnity.BoForUnityManager.ContextEmbeddingSource.Image)
                {
                    if (string.IsNullOrWhiteSpace(entry.imagePath))
                    {
                        throw new InvalidOperationException(
                            $"Context '{key}' has no image path, but the embedding source is Image."
                        );
                    }
                    info.imagePath = entry.imagePath.Trim();
                }

                payload.Add(info);
            }

            string currentKey = (_bomanager.currentContextKey ?? string.Empty).Trim();
            if (string.IsNullOrEmpty(currentKey))
            {
                throw new InvalidOperationException(
                    "Contextual optimization is enabled, but no Current Context Key is set."
                );
            }
            if (!seenKeys.Contains(currentKey))
            {
                throw new InvalidOperationException(
                    $"Current Context Key '{currentKey}' does not match any configured context. " +
                    "Add a context entry with this key or fix the key."
                );
            }

            return new ContextConfigInfo
            {
                enabled = true,
                currentContext = currentKey,
                embeddingSource = NormalizeContextEmbeddingSource(source),
                normalizeEmbeddings = _bomanager.normalizeContextEmbeddings,
                imageEmbeddingModel = string.IsNullOrWhiteSpace(_bomanager.contextEmbeddingModel)
                    ? null
                    : _bomanager.contextEmbeddingModel.Trim(),
                imageEmbeddingPretrained = string.IsNullOrWhiteSpace(_bomanager.contextEmbeddingPretrained)
                    ? null
                    : _bomanager.contextEmbeddingPretrained.Trim(),
                contexts = payload
            };
        }

        private static string NormalizeContextEmbeddingSource(
            BOforUnity.BoForUnityManager.ContextEmbeddingSource source)
        {
            switch (source)
            {
                case BOforUnity.BoForUnityManager.ContextEmbeddingSource.Manual:
                    return "manual";
                case BOforUnity.BoForUnityManager.ContextEmbeddingSource.Image:
                    return "image";
                case BOforUnity.BoForUnityManager.ContextEmbeddingSource.Learned:
                default:
                    return "learned";
            }
        }

        private static string NormalizeWarmStartObjectiveFormat(string value)
        {
            string normalized = (value ?? "auto").Trim().ToLowerInvariant();
            switch (normalized)
            {
                case "auto":
                case "raw":
                case "normalized_max":
                case "normalized_native":
                    return normalized;
                default:
                    Debug.LogWarning(
                        $"Invalid warmStartObjectiveFormat '{value}'. Falling back to 'auto'. " +
                        "Valid values: auto, raw, normalized_max, normalized_native."
                    );
                    return "auto";
            }
        }

        private static string NormalizeOptimizerBackend(BOforUnity.BoForUnityManager.OptimizerBackend backend)
        {
            switch (backend)
            {
                case BOforUnity.BoForUnityManager.OptimizerBackend.CABOP:
                    return "cabop";
                case BOforUnity.BoForUnityManager.OptimizerBackend.MetaTAF:
                    return "meta-taf";
                case BOforUnity.BoForUnityManager.OptimizerBackend.DBO:
                    return "dbo";
                case BOforUnity.BoForUnityManager.OptimizerBackend.BoTorch:
                default:
                    return "botorch";
            }
        }

        private static string NormalizeDboSpatialKernel(BOforUnity.BoForUnityManager.DboSpatialKernel kernel)
        {
            return kernel == BOforUnity.BoForUnityManager.DboSpatialKernel.Matern52 ? "matern52" : "rbf";
        }

        private static string NormalizeDboAlphaParameterization(BOforUnity.BoForUnityManager.DboAlphaParameterization p)
        {
            return p == BOforUnity.BoForUnityManager.DboAlphaParameterization.Direct ? "direct" : "decay";
        }

        private static string NormalizeMetaWeightMode(BOforUnity.BoForUnityManager.MetaWeightMode mode)
        {
            switch (mode)
            {
                case BOforUnity.BoForUnityManager.MetaWeightMode.TafM:
                    return "taf_m";
                case BOforUnity.BoForUnityManager.MetaWeightMode.TafRPareto:
                    return "taf_r_pareto";
                case BOforUnity.BoForUnityManager.MetaWeightMode.TafR:
                default:
                    return "taf_r";
            }
        }

        private static string NormalizeCabopObjectiveMode(BOforUnity.BoForUnityManager.CabopObjectiveMode mode)
        {
            switch (mode)
            {
                case BOforUnity.BoForUnityManager.CabopObjectiveMode.MultiObjectiveScalarized:
                    return "multi";
                case BOforUnity.BoForUnityManager.CabopObjectiveMode.SingleObjective:
                default:
                    return "single";
            }
        }

        private static string NormalizeCabopUpdateRule(BOforUnity.BoForUnityManager.CabopUpdateRule updateRule)
        {
            switch (updateRule)
            {
                case BOforUnity.BoForUnityManager.CabopUpdateRule.Intended:
                    return "intended";
                case BOforUnity.BoForUnityManager.CabopUpdateRule.Both:
                    return "both";
                case BOforUnity.BoForUnityManager.CabopUpdateRule.Actual:
                default:
                    return "actual";
            }
        }

        private static string NormalizeCabopGroup(string rawGroup)
        {
            string group = (rawGroup ?? string.Empty).Trim();
            return string.IsNullOrEmpty(group) ? "default" : group;
        }

        private static float NormalizeCabopObjectiveWeight(float value)
        {
            if (float.IsNaN(value) || float.IsInfinity(value) || value <= 0f)
                return 1f;
            return value;
        }

        private static List<float> NormalizeCabopPrefabricatedValues(List<float> values)
        {
            if (values == null || values.Count == 0)
                return new List<float>();

            var deduped = new SortedSet<float>();
            for (int i = 0; i < values.Count; i++)
            {
                float value = values[i];
                if (float.IsNaN(value) || float.IsInfinity(value))
                    continue;
                deduped.Add(value);
            }

            return deduped.ToList();
        }

        private List<CabopGroupCostInfo> BuildCabopGroupCosts(HashSet<string> parameterGroups)
        {
            var result = new List<CabopGroupCostInfo>();
            if (_bomanager == null)
                return result;

            var groupMap = new Dictionary<string, CabopGroupCostInfo>(StringComparer.OrdinalIgnoreCase);

            if (_bomanager.cabopGroupCosts != null)
            {
                for (int i = 0; i < _bomanager.cabopGroupCosts.Count; i++)
                {
                    var entry = _bomanager.cabopGroupCosts[i];
                    if (entry == null)
                        continue;

                    string group = NormalizeCabopGroup(entry.group);
                    if (groupMap.ContainsKey(group))
                    {
                        Debug.LogWarning($"Duplicate CABOP group cost entry for group '{group}'. Using first occurrence.");
                        continue;
                    }

                    groupMap[group] = new CabopGroupCostInfo
                    {
                        group = group,
                        cost = NormalizeCabopCostTriplet(entry.cost),
                        actualCost = NormalizeCabopCostTriplet(entry.actualCost)
                    };
                }
            }

            foreach (string group in parameterGroups)
            {
                if (groupMap.ContainsKey(group))
                    continue;

                groupMap[group] = new CabopGroupCostInfo
                {
                    group = group,
                    cost = DefaultCabopCostTriplet(),
                    actualCost = DefaultCabopCostTriplet()
                };
            }

            // Keep deterministic order: explicit inspector order first, then auto-added groups alphabetically.
            if (_bomanager.cabopGroupCosts != null)
            {
                foreach (var entry in _bomanager.cabopGroupCosts)
                {
                    if (entry == null)
                        continue;
                    string group = NormalizeCabopGroup(entry.group);
                    if (groupMap.TryGetValue(group, out var payload))
                    {
                        result.Add(payload);
                        groupMap.Remove(group);
                    }
                }
            }

            foreach (var payload in groupMap.OrderBy(kv => kv.Key, StringComparer.OrdinalIgnoreCase).Select(kv => kv.Value))
                result.Add(payload);

            return result;
        }

        private static CabopCostTripletInfo NormalizeCabopCostTriplet(BOforUnity.CabopCostTriplet source)
        {
            if (source == null)
                return DefaultCabopCostTriplet();

            return new CabopCostTripletInfo
            {
                unchanged = NormalizeCabopCostValue(source.unchanged, 1f),
                swapped = NormalizeCabopCostValue(source.swapped, 10f),
                acquired = NormalizeCabopCostValue(source.acquired, 100f)
            };
        }

        private static CabopCostTripletInfo DefaultCabopCostTriplet()
        {
            return new CabopCostTripletInfo
            {
                unchanged = 1f,
                swapped = 10f,
                acquired = 100f
            };
        }

        private static float NormalizeCabopCostValue(float value, float fallback)
        {
            if (float.IsNaN(value) || float.IsInfinity(value) || value < 0f)
                return fallback;
            return value;
        }

        public void SendObjectives()
        {
            _bomanager = _bomanager ?? gameObject.GetComponent<BoForUnityManager>();
            if (_bomanager == null || _bomanager.objectives == null)
            {
                throw new InvalidOperationException("Cannot send objectives because BoForUnityManager objectives are not configured.");
            }

            var finalObjectives = new Dictionary<string, float>(_bomanager.objectives.Count);
            bool hadAdjustedObjective = false;
            var seenObjectiveKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var invalidObjectiveEntries = new List<string>();
            var duplicateObjectiveKeys = new List<string>();

            for (int i = 0; i < _bomanager.objectives.Count; i++)
            {
                var ob = _bomanager.objectives[i];
                if (ob == null || ob.value == null || string.IsNullOrWhiteSpace(ob.key))
                {
                    invalidObjectiveEntries.Add($"index {i}");
                    continue;
                }

                string key = ob.key.Trim();
                if (!seenObjectiveKeys.Add(key))
                {
                    if (!duplicateObjectiveKeys.Contains(key, StringComparer.OrdinalIgnoreCase))
                        duplicateObjectiveKeys.Add(key);
                    continue;
                }

                var value = ob.value;
                var subMeasures = value.values ?? (value.values = new List<float>());

                int subMeasureWindow = value.numberOfSubMeasures;
                if (subMeasureWindow <= 0)
                {
                    Debug.LogWarning(
                        $"Objective '{ob.key}' has invalid numberOfSubMeasures={subMeasureWindow}. " +
                        "Using 1 as fallback window size."
                    );
                    subMeasureWindow = 1;
                }

                // Mean of the finite sub-measures among the last N: one unanswered item (submitted as NaN)
                // must not discard the answered ones of a multi-item objective. The same reduction feeds the
                // perfect-rating check and the finaldesign row.
                var aggregate = BoObjectiveMath.Aggregate(subMeasures, subMeasureWindow, value.lowerBound, value.upperBound);
                float lo = Mathf.Min(value.lowerBound, value.upperBound);
                float hi = Mathf.Max(value.lowerBound, value.upperBound);

                if (aggregate.DroppedCount > 0)
                {
                    Debug.LogWarning(
                        $"Objective '{ob.key}' received {subMeasures.Count} sub-measures but Number Of Sub Measures is " +
                        $"{subMeasureWindow}: the first {aggregate.DroppedCount} are ignored and only the last " +
                        $"{subMeasureWindow} are averaged. Raise Number Of Sub Measures if every value should count " +
                        "(e.g. more questionnaire items match this objective key than configured)."
                    );
                }

                if (aggregate.UsedMidpointFallback)
                {
                    Debug.LogWarning(
                        (aggregate.WindowCount == 0
                            ? $"Objective '{ob.key}' has no values for this iteration. "
                            : $"Objective '{ob.key}' has only non-numeric values for this iteration. ") +
                        $"Using fallback midpoint {aggregate.Value} in [{lo}, {hi}]."
                    );
                    hadAdjustedObjective = true;
                }
                else if (aggregate.FiniteCount < aggregate.WindowCount)
                {
                    Debug.LogWarning(
                        $"Objective '{ob.key}': ignoring {aggregate.WindowCount - aggregate.FiniteCount} non-numeric " +
                        $"sub-measure(s) and averaging the remaining {aggregate.FiniteCount}."
                    );
                    hadAdjustedObjective = true;
                }

                if (aggregate.Clamped)
                {
                    Debug.LogWarning(
                        $"Objective '{ob.key}' value {aggregate.UnclampedValue} is outside configured bounds [{lo}, {hi}]. " +
                        $"Clamping to {aggregate.Value} before sending to Python."
                    );
                    hadAdjustedObjective = true;
                }

                float val = aggregate.Value;
                finalObjectives[key] = val;
            }

            if (invalidObjectiveEntries.Count > 0 || duplicateObjectiveKeys.Count > 0)
            {
                var details = new List<string>();
                if (invalidObjectiveEntries.Count > 0)
                    details.Add("invalid objective entries at " + string.Join(", ", invalidObjectiveEntries));
                if (duplicateObjectiveKeys.Count > 0)
                    details.Add("duplicate objective key(s): " + string.Join(", ", duplicateObjectiveKeys));

                throw new InvalidOperationException(
                    "Cannot send objectives with invalid objective configuration. " +
                    string.Join("; ", details)
                );
            }

            if (finalObjectives.Count == 0)
            {
                Debug.LogError(
                    "No valid objective values are available to send to Python. " +
                    "Sending an empty objective payload so the backend can fail fast."
                );
            }

            if (hadAdjustedObjective)
            {
                Debug.LogWarning(
                    "One or more objective values were adjusted (fallback/clamped) before sending to Python. " +
                    "Consider checking objective instrumentation and configured bounds in BoForUnityManager."
                );
            }

            var msg = new ObjectivesMsg
            {
                type = "objectives",
                values = finalObjectives
            };

            string json = SerializeJson(msg);
            SocketSendLine(json);
        }

        /// <summary>
        /// Asks Python to end the session cleanly (it logs <paramref name="reason"/> and exits) and marks the
        /// connection as intentionally closing. Best effort: failures are ignored, the process is still stopped
        /// by SocketQuit. Main thread.
        /// </summary>
        public void SendStop(string reason)
        {
            TrySendStop(reason);
            _stopRequested = true;
        }

        private void TrySendStop(string reason)
        {
            if (_stopMessageSent || _optimizationFinished || _serverSocket == null || !_serverSocket.Connected)
                return;

            try
            {
                SocketSendLine(SerializeJson(new StopMsg { type = "stop", reason = reason ?? string.Empty }));
                _stopMessageSent = true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Could not send stop message to Python ({reason}): {ex.Message}");
            }
        }

        // -------------------- Low-level send/quit --------------------
        private void SocketSendLine(string json)
        {
            if (_serverSocket == null || !_serverSocket.Connected)
            {
                throw new InvalidOperationException("Socket is not connected.");
            }

            string line = json + "\n"; // NDJSON framing
            byte[] sendData = Encoding.UTF8.GetBytes(line);
            Debug.Log("Unity sending: " + json);
            int totalSent = 0;
            while (totalSent < sendData.Length)
            {
                int sent = _serverSocket.Send(
                    sendData,
                    totalSent,
                    sendData.Length - totalSent,
                    SocketFlags.None
                );
                if (sent <= 0)
                    throw new SocketException((int)SocketError.ConnectionReset);
                totalSent += sent;
            }
        }

        public void SocketQuit()
        {
            _stopRequested = true;

            // The manager is known only once InitSocket ran; a study ended before that (e.g. an invalid
            // configuration found after Python started) must still stop the process. Not for a duplicate manager
            // discarded in Awake (OnDestroy): it never started anything.
            PythonStarter pythonStarter = _bomanager != null ? _bomanager.pythonStarter : null;
            if (pythonStarter == null)
            {
                var manager = GetComponent<BoForUnityManager>();
                if (manager != null && manager == BoForUnityManager.Instance)
                    pythonStarter = manager.pythonStarter;
            }

            // After a stop message or optimization_finished Python exits by itself: it closes the connection (which
            // ends the receive thread), then saves logs that were locked and exits. Give it a moment before the
            // process is terminated.
            if ((_stopMessageSent || _optimizationFinished) && _connectThread != null &&
                _connectThread != Thread.CurrentThread)
            {
                var grace = System.Diagnostics.Stopwatch.StartNew();
                try { _connectThread.Join(StopGracePeriodMs); } catch { }
                while (pythonStarter != null && Volatile.Read(ref pythonStarter.isPythonProcessRunning) &&
                       grace.ElapsedMilliseconds < StopGracePeriodMs)
                {
                    Thread.Sleep(10);
                }
            }

            try { pythonStarter?.StopPythonProcess(); } catch { }

            try { _serverSocket?.Shutdown(SocketShutdown.Both); } catch { }
            try { _serverSocket?.Close(); } catch { }

            if (_connectThread != null)
            {
                try { _connectThread.Interrupt(); } catch { }
                bool joined = false;
                try { joined = _connectThread.Join(1000); } catch { }
                if (!joined && _connectThread.IsAlive)
                    Debug.LogWarning("Socket receive thread did not stop within the shutdown timeout.");
                _connectThread = null;
            }

        }
    }
}
