// SocketNetwork.cs
// Unity <-> Python NDJSON protocol using Newtonsoft.Json.
// Place Newtonsoft.Json source under Assets/<YourAsset>/ThirdParty/Newtonsoft.Json with an .asmdef.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Newtonsoft.Json;

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
    }

    [Serializable] class ParamInit { public double low; public double high; }
    [Serializable] class ObjInit   { public double low; public double high; public int minimize; }

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

    [Serializable] class UserInfo
    {
        public string userId, conditionId, groupId;
    }

    [Serializable] class ParametersMsg : MsgBase
    {
        public Dictionary<string, float> values;
    }

    [Serializable] class ObjectivesMsg : MsgBase
    {
        public Dictionary<string, float> values;
    }

    [Serializable] class CoverageMsg : MsgBase
    {
        public float value;
    }

    // -------------------- SocketNetwork --------------------
    public class SocketNetwork : MonoBehaviour
    {
        private Socket _serverSocket;
        private IPAddress _ip;
        private IPEndPoint _ipEnd;
        private Thread _connectThread;
        private volatile bool _stopRequested;
        private volatile bool _connectionClosedByPeer;
        private volatile bool _optimizationFinished;

        public float coverage = 0f;
        public float tempCoverage = 0f;

        private BoForUnityManager _bomanager;

        // TCP buffer for NDJSON framing
        private readonly byte[] _recvBuf = new byte[4096];
        private readonly StringBuilder _lineBuf = new StringBuilder(4096);

        // JSON settings
        private static readonly JsonSerializerSettings JsonSettings = new JsonSerializerSettings
        {
            Culture = CultureInfo.InvariantCulture,
            Formatting = Formatting.None
        };

        private bool _shutdownHandled;

        // -------------------- Lifecycle --------------------
        public void InitSocket()
        {
            _bomanager = gameObject.GetComponent<BoForUnityManager>();
            _ip = IPAddress.Parse("127.0.0.1");
            _ipEnd = new IPEndPoint(_ip, 56001);

            _stopRequested = false;
            _connectionClosedByPeer = false;
            _optimizationFinished = false;
            _shutdownHandled = false;
            _lineBuf.Length = 0;
            _connectThread = new Thread(SocketReceive) { IsBackground = true };
            _connectThread.Start();
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
                SendInitInfo();

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
                            MainThreadDispatcher.Execute(OnSocketConnectionFailed);
                        }

                        _stopRequested = true;
                        try { _serverSocket?.Shutdown(SocketShutdown.Both); } catch { }
                        try { _serverSocket?.Close(); } catch { }
                        break;
                    }

                    var chunk = Encoding.UTF8.GetString(_recvBuf, 0, recvLen);
                    _lineBuf.Append(chunk);

                    int newlineIndex;
                    bool protocolError = false;
                    while ((newlineIndex = _lineBuf.ToString().IndexOf('\n')) >= 0)
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
                            MainThreadDispatcher.Execute(OnSocketConnectionFailed);
                            try { _serverSocket?.Shutdown(SocketShutdown.Both); } catch { }
                            try { _serverSocket?.Close(); } catch { }
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
                    MainThreadDispatcher.Execute(OnSocketConnectionFailed);
                }

                _stopRequested = true;
                try { _serverSocket?.Shutdown(SocketShutdown.Both); } catch { }
                try { _serverSocket?.Close(); } catch { }
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
                    MainThreadDispatcher.Execute(OnSocketConnectionFailed);
                }

                _stopRequested = true;
                try { _serverSocket?.Shutdown(SocketShutdown.Both); } catch { }
                try { _serverSocket?.Close(); } catch { }
            }
        }

        private void SocketConnect()
        {
            _serverSocket?.Close();
            _serverSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            Debug.Log("Unity is ready to connect...");
            _serverSocket.Connect(_ipEnd);
        }

        private void OnSocketConnectionFailed()
        {
            _bomanager = _bomanager ?? gameObject.GetComponent<BoForUnityManager>();
            if (_bomanager == null)
                return;

            _bomanager.optimizationRunning = false;
            _bomanager.simulationRunning = false;
            _bomanager.hasNewDesignParameterValues = false;

            if (_bomanager.loadingObj != null)
                _bomanager.loadingObj.SetActive(false);
            if (_bomanager.nextButton != null)
                _bomanager.nextButton.SetActive(false);
            if (_bomanager.outputText != null)
            {
                _bomanager.outputText.text =
                    "Optimizer connection failed.\nCheck parameter/objective configuration and Python logs, then restart.";
            }

            try
            {
                _bomanager.pythonStarter?.StopPythonProcess();
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Failed to stop Python process after socket connection failure: {ex.Message}");
            }
        }

        // -------------------- Protocol: incoming --------------------
        private void ParseJsonMessage(string json)
        {
            var peek = JsonConvert.DeserializeObject<MsgBase>(json);
            if (peek == null || string.IsNullOrEmpty(peek.type))
            {
                throw new InvalidOperationException("Received protocol message without a valid 'type' field.");
            }

            switch (peek.type)
            {
                case "parameters":
                {
                    var msg = JsonConvert.DeserializeObject<ParametersMsg>(json);
                    if (msg?.values == null)
                    {
                        throw new InvalidOperationException("Received 'parameters' message without a valid 'values' payload.");
                    }

                    MainThreadDispatcher.Execute(() =>
                    {
                        _bomanager = gameObject.GetComponent<BoForUnityManager>();
                        if (_bomanager == null || _bomanager.parameters == null)
                        {
                            Debug.LogError(
                                "Cannot apply parameters because BoForUnityManager parameters are not configured. " +
                                "Terminating optimizer session to avoid backend deadlock."
                            );
                            OnSocketConnectionFailed();
                            SocketQuit();
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
                            SocketQuit();
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
                            SocketQuit();
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
                            SocketQuit();
                            return;
                        }

                        // Apply values only after payload completeness has been validated.
                        foreach (var pa in expectedParameters)
                        {
                            string expectedKey = pa.key.Trim();
                            pa.value.Value = incomingValues[expectedKey];
                        }

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
                    _optimizationFinished = true;
                    break;
                }

                case "coverage":
                {
                    var msg = JsonConvert.DeserializeObject<CoverageMsg>(json);
                    if (msg != null)
                    {
                        coverage = msg.value;
                        Debug.Log($"coverage {coverage}");
                    }
                    else
                    {
                        throw new InvalidOperationException("Received malformed 'coverage' message.");
                    }
                    break;
                }

                case "tempCoverage":
                {
                    var msg = JsonConvert.DeserializeObject<CoverageMsg>(json);
                    if (msg != null)
                    {
                        tempCoverage = msg.value;
                        Debug.Log($"tempCoverage {tempCoverage}");
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
                        tolerance = NormalizeCabopTolerance(parameter.value.cabopTolerance),
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
                    cabopMaxCumulativeCost = _bomanager.cabopMaxCumulativeCost
                },
                parameters = parameterPayload,
                objectives = objectivePayload,
                cabopGroupCosts = BuildCabopGroupCosts(parameterGroups),
                user = new UserInfo
                {
                    userId = _bomanager.userId,
                    conditionId = _bomanager.conditionId,
                    groupId = _bomanager.groupId
                }
            };

            string json = JsonConvert.SerializeObject(init, JsonSettings);
            SocketSendLine(json);
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
            return backend == BOforUnity.BoForUnityManager.OptimizerBackend.CABOP ? "cabop" : "botorch";
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

        private static float NormalizeCabopTolerance(float value)
        {
            if (float.IsNaN(value) || float.IsInfinity(value) || value < 0f)
                return 0f;
            return value;
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
                var tmpList = value.values ?? (value.values = new List<float>());

                int subMeasureWindow = value.numberOfSubMeasures;
                if (subMeasureWindow <= 0)
                {
                    Debug.LogWarning(
                        $"Objective '{ob.key}' has invalid numberOfSubMeasures={subMeasureWindow}. " +
                        "Using 1 as fallback window size."
                    );
                    subMeasureWindow = 1;
                }

                // keep the last N submeasures
                int removeCount = Math.Max(0, tmpList.Count - subMeasureWindow);
                if (removeCount > 0)
                {
                    tmpList.RemoveRange(0, removeCount);
                }

                float lo = Mathf.Min(value.lowerBound, value.upperBound);
                float hi = Mathf.Max(value.lowerBound, value.upperBound);
                float val;
                if (tmpList.Count == 0)
                {
                    float fallback = 0.5f * (lo + hi);
                    Debug.LogWarning(
                        $"Objective '{ob.key}' has no values for this iteration. " +
                        $"Using fallback midpoint {fallback} in [{lo}, {hi}]."
                    );
                    val = fallback;
                    hadAdjustedObjective = true;
                }
                else
                {
                    val = (float)tmpList.Average();
                }

                if (float.IsNaN(val) || float.IsInfinity(val))
                {
                    float fallback = 0.5f * (lo + hi);
                    Debug.LogWarning(
                        $"Objective '{ob.key}' produced a non-finite value ({val}). " +
                        $"Using fallback midpoint {fallback} in [{lo}, {hi}]."
                    );
                    val = fallback;
                    hadAdjustedObjective = true;
                }
                else if (val < lo || val > hi)
                {
                    float rawVal = val;
                    val = Mathf.Clamp(val, lo, hi);
                    Debug.LogWarning(
                        $"Objective '{ob.key}' value {rawVal} is outside configured bounds [{lo}, {hi}]. " +
                        $"Clamping to {val} before sending to Python."
                    );
                    hadAdjustedObjective = true;
                }

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

            string json = JsonConvert.SerializeObject(msg, JsonSettings);
            SocketSendLine(json);
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
            _serverSocket.Send(sendData, sendData.Length, SocketFlags.None);
        }

        public void SocketQuit()
        {
            _stopRequested = true;

            try { _bomanager?.pythonStarter?.StopPythonProcess(); } catch { }

            try { _serverSocket?.Shutdown(SocketShutdown.Both); } catch { }
            try { _serverSocket?.Close(); } catch { }

            if (_connectThread != null)
            {
                try { _connectThread.Interrupt(); } catch { }
                try { _connectThread.Join(200); } catch { }
                if (_connectThread.IsAlive)
                {
                    try { _connectThread.Abort(); } catch { }
                }
                _connectThread = null;
            }

        }

        private void HandlePeerInitiatedShutdown(SocketException socketException = null)
        {
            if (_shutdownHandled)
                return;

            _shutdownHandled = true;

            if (_optimizationFinished)
            {
                Debug.Log("Python optimization process closed the connection. Optimization iterations have finished successfully.");
            }
            else
            {
                if (socketException != null)
                {
                    Debug.LogError($"Socket closed by Python unexpectedly before optimization completed. Error: {socketException.SocketErrorCode} {socketException.Message}");
                }
                else
                {
                    Debug.LogError("Socket closed by Python unexpectedly before optimization completed.");
                }
                MainThreadDispatcher.Execute(OnSocketConnectionFailed);
            }
        }
    }
}
