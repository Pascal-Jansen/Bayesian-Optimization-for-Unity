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
        public UserInfo user;
    }

    [Serializable] class InitConfig
    {
        public int batchSize, numRestarts, rawSamples, numOptimizationIterations, mcSamples, numSamplingIterations, seed;
        public int nParameters, nObjectives;
        public bool warmStart;
        public string initialParametersDataPath, initialObjectivesDataPath, warmStartObjectiveFormat;
    }

    [Serializable] class ParamInit { public double low; public double high; }
    [Serializable] class ObjInit   { public double low; public double high; public int minimize; }

    [Serializable] class ParamInfo
    {
        public string key;
        public ParamInit init;
        public int optSeqOrder;
    }

    [Serializable] class ObjInfo
    {
        public string key;
        public ObjInit init;
        public int optSeqOrder;
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

                        if (_optimizationFinished)
                        {
                            Debug.Log("Python optimization process closed the connection. Optimization iterations have finished successfully.");
                        }
                        else
                        {
                            Debug.LogError("Socket closed by Python unexpectedly before optimization completed.");
                            MainThreadDispatcher.Execute(OnSocketConnectionFailed);
                        }

                        SocketQuit();
                        break;
                    }

                    var chunk = Encoding.UTF8.GetString(_recvBuf, 0, recvLen);
                    _lineBuf.Append(chunk);

                    int newlineIndex;
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
                        }
                    }
                }
            }
            catch (SocketException ex)
            {
                if (_stopRequested || _connectionClosedByPeer)
                {
                    Debug.Log("Socket connection closed.");
                }
                else
                {
                    Debug.LogError($"SocketReceive SocketException: {ex.SocketErrorCode} {ex.Message}");
                    MainThreadDispatcher.Execute(OnSocketConnectionFailed);
                }
            }
            catch (Exception ex)
            {
                if (ex.Message != "Thread was being aborted.")
                {
                    Debug.LogError($"Error in SocketReceive: {ex.Message}\n{ex.StackTrace}");
                }
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
            // Optionally reload scene or surface UI feedback
            // UnityEngine.SceneManagement.SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }

        // -------------------- Protocol: incoming --------------------
        private void ParseJsonMessage(string json)
        {
            var peek = JsonConvert.DeserializeObject<MsgBase>(json);
            if (peek == null || string.IsNullOrEmpty(peek.type))
            {
                Debug.LogWarning("Unknown or empty message.");
                return;
            }

            switch (peek.type)
            {
                case "parameters":
                {
                    var msg = JsonConvert.DeserializeObject<ParametersMsg>(json);
                    if (msg?.values == null)
                    {
                        Debug.LogWarning("parameters.values missing");
                        return;
                    }

                    MainThreadDispatcher.Execute(() =>
                    {
                        _bomanager = gameObject.GetComponent<BoForUnityManager>();

                        // Apply values by key
                        foreach (var pa in _bomanager.parameters)
                        {
                            if (msg.values.TryGetValue(pa.key, out var v))
                                pa.value.Value = v;
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
                    break;
                }

                case "objectives":
                {
                    // Python never sends this unsolicited; Unity sends objectives to Python.
                    break;
                }

                default:
                    Debug.LogWarning($"Unknown message type: {peek.type}");
                    break;
            }
        }

        // -------------------- Protocol: outgoing --------------------
        private void SendInitInfo()
        {
            int i = 0;
            foreach (var pa in _bomanager.parameters) pa.value.optSeqOrder = i++;
            i = 0;
            foreach (var ob in _bomanager.objectives) ob.value.optSeqOrder = i++;

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
                    numSamplingIterations = _bomanager.numSamplingIterations,
                    seed = _bomanager.seed,
                    nParameters = _bomanager.parameters.Count,
                    nObjectives = _bomanager.objectives.Count,
                    warmStart = _bomanager.warmStart,
                    initialParametersDataPath = _bomanager.initialParametersDataPath,
                    initialObjectivesDataPath = _bomanager.initialObjectivesDataPath,
                    warmStartObjectiveFormat = NormalizeWarmStartObjectiveFormat(_bomanager.warmStartObjectiveFormat)
                },
                parameters = _bomanager.parameters.Select(p => new ParamInfo
                {
                    key = p.key,
                    init = new ParamInit
                    {
                        // Adjust field names if your Parameter class differs:
                        low = p.value.lowerBound,
                        high = p.value.upperBound
                    },
                    optSeqOrder = p.value.optSeqOrder
                }).ToList(),
                objectives = _bomanager.objectives.Select(o => new ObjInfo
                {
                    key = o.key,
                    init = new ObjInit
                    {
                        // Adjust field names if your Objective class differs:
                        low = o.value.lowerBound,
                        high = o.value.upperBound,
                        minimize = o.value.smallerIsBetter ? 1 : 0
                    },
                    optSeqOrder = o.value.optSeqOrder
                }).ToList(),
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

        public void SendObjectives()
        {
            var finalObjectives = new Dictionary<string, float>(_bomanager.objectives.Count);

            foreach (var ob in _bomanager.objectives)
            {
                var value = ob.value;
                var tmpList = value.values;
                // keep the last N submeasures
                tmpList.RemoveRange(0, Math.Max(0, value.values.Count - value.numberOfSubMeasures));
                float val = tmpList.Count > 0 ? (float)tmpList.Average() : 0f;
                finalObjectives[ob.key] = val;
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
