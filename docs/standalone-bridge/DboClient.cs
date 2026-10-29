// DboClient.cs — Unity client for the standalone dbo-torch bridge (unity_bridge.py).
//
// Start the Python side first, from the folder that holds unity_bridge.py:
//     python unity_bridge.py --host 127.0.0.1 --port 8756 --save-dir runs
//
// The protocol is newline-delimited JSON over TCP. Calls are queued and sent by
// one background thread, strictly in the order they were made, and each result
// returns to the Unity main thread via a callback, so nothing here stalls the
// render loop and a queued call never overtakes an earlier one.
//
// Typical use in a study scene:
//
//     var dbo = GetComponent<DboClient>();
//     dbo.Reset(new[] { new[] { -5f, 9f } }, seedPoints: new[] {
//         new[] { 5f }, new[] { 7f }, new[] { 3f } });
//     dbo.Suggest(x => StartCoroutine(RunTrial(x)));   // runs after Reset
//     // ... after measuring the participant's response:
//     dbo.Observe(x, cost, () => dbo.Suggest(...));

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace DboTorch
{
    [Serializable]
    public class DboSuggestion
    {
        public float[] X;
        public int Iteration;
        public bool IsValidation;
    }

    [Serializable]
    public class DboPrediction
    {
        public float Mean;
        public float StdDev;
    }

    public class DboClient : MonoBehaviour
    {
        [Header("Server")]
        public string Host = "127.0.0.1";
        public int Port = 8756;

        [Tooltip("Seconds to wait for a reply before giving up. GP fitting on " +
                 "large traces can take a moment, so keep this generous.")]
        public float TimeoutSeconds = 30f;

        [Header("Diagnostics")]
        public bool LogTraffic = false;

        private TcpClient _client;
        private NetworkStream _stream;
        private StreamReader _reader;
        private StreamWriter _writer;
        private readonly object _ioLock = new object();

        // Requests are sent by ONE worker thread in FIFO order. (A thread per call
        // contending for a lock gives no ordering guarantee: `Reset(); Suggest();`
        // could reach the server as suggest-then-reset.)
        private readonly object _queueLock = new object();
        private readonly Queue<PendingRequest> _queue = new Queue<PendingRequest>();
        private Thread _worker;
        private bool _shuttingDown;

        private sealed class PendingRequest
        {
            public string Payload;
            public Action<BridgeReply> OnResult;
            public Action<string> OnError;
            public ReplyKind ExpectedReply;
        }

        private enum ReplyKind { Acknowledgment, Suggestion, Prediction }

        private sealed class BridgeReply
        {
            public bool Ok;
            public string Error;
            public DboSuggestion Suggestion;
            public DboPrediction Prediction;
        }

        // Work queued back onto the Unity main thread.
        private readonly ConcurrentQueue<Action> _mainThread = new ConcurrentQueue<Action>();

        public bool IsConnected => _client != null && _client.Connected;

        // ---------------------------------------------------------------
        // Lifecycle
        // ---------------------------------------------------------------

        private void Update()
        {
            while (_mainThread.TryDequeue(out var action))
            {
                action();
            }
        }

        private void OnDestroy() => Shutdown();
        private void OnApplicationQuit() => Shutdown();

        public void Connect()
        {
            lock (_ioLock)
            {
                if (IsConnected) return;

                _client = new TcpClient
                {
                    ReceiveTimeout = Mathf.RoundToInt(TimeoutSeconds * 1000f),
                    SendTimeout = Mathf.RoundToInt(TimeoutSeconds * 1000f),
                    NoDelay = true,
                };
                _client.Connect(Host, Port);
                _stream = _client.GetStream();
                _reader = new StreamReader(_stream, new UTF8Encoding(false));
                _writer = new StreamWriter(_stream, new UTF8Encoding(false)) { AutoFlush = true };
            }
        }

        /// <summary>
        /// Closes the connection; the next call reconnects. Queued calls are kept.
        /// </summary>
        public void Disconnect()
        {
            lock (_ioLock)
            {
                try { _writer?.Dispose(); } catch { /* closing anyway */ }
                try { _reader?.Dispose(); } catch { /* closing anyway */ }
                try { _stream?.Dispose(); } catch { /* closing anyway */ }
                try { _client?.Close(); } catch { /* closing anyway */ }
                _writer = null; _reader = null; _stream = null; _client = null;
            }
        }

        // ---------------------------------------------------------------
        // Public API
        // ---------------------------------------------------------------

        /// <summary>Start a fresh optimisation run.</summary>
        /// <param name="bounds">One {lo, hi} pair per control parameter.</param>
        /// <param name="seedPoints">Optional fixed inputs for the first iterations.</param>
        public void Reset(
            float[][] bounds,
            float[][] seedPoints = null,
            float explorationRatio = 0.1f,
            int? randomSeed = null,
            Action onDone = null,
            Action<string> onError = null)
        {
            var sb = new StringBuilder();
            sb.Append("{\"cmd\":\"reset\",\"bounds\":").Append(Json(bounds));
            sb.Append(",\"exploration_ratio\":").Append(Num(explorationRatio));
            if (seedPoints != null)
                sb.Append(",\"seed_points\":").Append(Json(seedPoints));
            if (randomSeed.HasValue)
                sb.Append(",\"seed\":").Append(randomSeed.Value);
            sb.Append('}');

            Send(sb.ToString(), _ => onDone?.Invoke(), onError);
        }

        /// <summary>Ask for the next input to test.</summary>
        public void Suggest(Action<DboSuggestion> onResult, Action<string> onError = null)
        {
            Send("{\"cmd\":\"suggest\"}",
                 reply => onResult?.Invoke(reply.Suggestion), onError, ReplyKind.Suggestion);
        }

        /// <summary>
        /// Ask for the optimiser's current best estimate, for a validation
        /// iteration. This does not consume an acquisition step.
        /// </summary>
        public void SuggestValidation(Action<DboSuggestion> onResult, Action<string> onError = null)
        {
            Send("{\"cmd\":\"suggest_validation\"}",
                 reply => onResult?.Invoke(reply.Suggestion), onError, ReplyKind.Suggestion);
        }

        /// <summary>Report the measured cost for an input.</summary>
        public void Observe(
            float[] x, float cost, Action onDone = null, Action<string> onError = null)
        {
            var payload = "{\"cmd\":\"observe\",\"x\":" + Json(x) +
                          ",\"y\":" + Num(cost) + "}";
            Send(payload, _ => onDone?.Invoke(), onError);
        }

        /// <summary>Posterior mean and standard deviation at a point.</summary>
        public void Predict(
            float[] x, Action<DboPrediction> onResult, Action<string> onError = null)
        {
            Send("{\"cmd\":\"predict\",\"x\":" + Json(x) + "}",
                 reply => onResult?.Invoke(reply.Prediction), onError, ReplyKind.Prediction);
        }

        /// <summary>
        /// Write the full run history to a JSON file on the server. <paramref name="path"/>
        /// is relative to the server's --save-dir; absolute paths and ".." are refused.
        /// </summary>
        public void Save(string path, Action onDone = null, Action<string> onError = null)
        {
            Send("{\"cmd\":\"save\",\"path\":\"" + Escape(path) + "\"}",
                 _ => onDone?.Invoke(), onError);
        }

        // ---------------------------------------------------------------
        // Transport
        // ---------------------------------------------------------------

        private void Send(string payload, Action<BridgeReply> onResult, Action<string> onError,
            ReplyKind expectedReply = ReplyKind.Acknowledgment)
        {
            lock (_queueLock)
            {
                if (_shuttingDown)
                {
                    _mainThread.Enqueue(() => Fail(onError, "DboClient is shutting down."));
                    return;
                }

                _queue.Enqueue(new PendingRequest
                {
                    Payload = payload, OnResult = onResult, OnError = onError, ExpectedReply = expectedReply
                });
                if (_worker == null || !_worker.IsAlive)
                {
                    _worker = new Thread(ProcessQueue) { IsBackground = true, Name = "DboClient" };
                    _worker.Start();
                }
                Monitor.Pulse(_queueLock);
            }
        }

        private void ProcessQueue()
        {
            while (true)
            {
                PendingRequest request;
                lock (_queueLock)
                {
                    while (_queue.Count == 0 && !_shuttingDown)
                        Monitor.Wait(_queueLock);
                    if (_shuttingDown)
                        return;
                    request = _queue.Dequeue();
                }

                Execute(request);
            }
        }

        private void Execute(PendingRequest request)
        {
            string reply;
            BridgeReply parsed;
            try
            {
                lock (_ioLock)
                {
                    if (!IsConnected) Connect();
                    if (LogTraffic) Debug.Log("[DBO] -> " + request.Payload);
                    _writer.Write(request.Payload);
                    _writer.Write('\n');
                    reply = _reader.ReadLine();
                }

                if (reply == null)
                    throw new IOException("Server closed the connection.");
                if (LogTraffic) Debug.Log("[DBO] <- " + reply);

                // Validate before queueing the main thread success callback.
                parsed = ParseReply(reply, request.ExpectedReply);
                if (!parsed.Ok)
                {
                    var msg = parsed.Error;
                    _mainThread.Enqueue(() => Fail(request.OnError, msg));
                    return;
                }
            }
            catch (Exception e)
            {
                var msg = e.Message;
                Disconnect();   // force a clean reconnect on the next call
                _mainThread.Enqueue(() => Fail(request.OnError, msg));
                return;
            }

            _mainThread.Enqueue(() => request.OnResult?.Invoke(parsed));
        }

        // Stops the worker and drops queued calls. Closing the socket without taking
        // _ioLock unblocks a worker waiting for a reply, so quitting never stalls on
        // the request timeout.
        private void Shutdown()
        {
            lock (_queueLock)
            {
                _shuttingDown = true;
                _queue.Clear();
                Monitor.PulseAll(_queueLock);
            }

            try { _client?.Close(); } catch { /* closing anyway */ }
            Disconnect();
        }

        private static void Fail(Action<string> onError, string message)
        {
            if (onError != null) onError(message);
            else Debug.LogError("[DBO] " + message);
        }

        // ---------------------------------------------------------------
        // Request formatting and strict reply parsing.
        // Requires Unity's com.unity.nuget.newtonsoft-json package.
        // ---------------------------------------------------------------

        private static string Num(float v) =>
            v.ToString("R", CultureInfo.InvariantCulture);

        private static string Escape(string s) =>
            s.Replace("\\", "\\\\").Replace("\"", "\\\"")
             .Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");

        private static string Json(float[] v)
        {
            var sb = new StringBuilder("[");
            for (int i = 0; i < v.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(Num(v[i]));
            }
            return sb.Append(']').ToString();
        }

        private static string Json(float[][] v)
        {
            var sb = new StringBuilder("[");
            for (int i = 0; i < v.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(Json(v[i]));
            }
            return sb.Append(']').ToString();
        }

        private static BridgeReply ParseReply(string json, ReplyKind kind)
        {
            var root = JObject.Parse(json, new JsonLoadSettings
            {
                DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error
            });
            if (root["ok"]?.Type != JTokenType.Boolean)
                throw new InvalidDataException("Reply must contain a boolean 'ok' field.");
            var reply = new BridgeReply { Ok = root["ok"].Value<bool>() };
            if (!reply.Ok)
            {
                reply.Error = root["error"]?.Type == JTokenType.String
                    ? root["error"].Value<string>() : "unknown server error";
                return reply;
            }
            if (kind == ReplyKind.Suggestion)
            {
                if (!(root["x"] is JArray values) || values.Count == 0)
                    throw new InvalidDataException("Suggestion must contain a nonempty numeric 'x' array.");
                var x = new float[values.Count];
                for (int i = 0; i < values.Count; i++)
                    x[i] = ReadFiniteFloat(values[i], "x[" + i + "]");
                if (root["iteration"]?.Type != JTokenType.Integer ||
                    !int.TryParse(root["iteration"].ToString(), NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out int iteration) || iteration < 0)
                    throw new InvalidDataException("Suggestion must contain a nonnegative integer 'iteration'.");
                if (root["is_validation"]?.Type != JTokenType.Boolean)
                    throw new InvalidDataException("Suggestion must contain a boolean 'is_validation'.");
                reply.Suggestion = new DboSuggestion
                {
                    X = x, Iteration = iteration, IsValidation = root["is_validation"].Value<bool>()
                };
            }
            else if (kind == ReplyKind.Prediction)
            {
                float mean = ReadFiniteFloat(root["mean"], "mean");
                float std = ReadFiniteFloat(root["std"], "std");
                if (std < 0f)
                    throw new InvalidDataException("Prediction standard deviation must be nonnegative.");
                reply.Prediction = new DboPrediction { Mean = mean, StdDev = std };
            }
            return reply;
        }

        private static float ReadFiniteFloat(JToken token, string name)
        {
            if (token == null || (token.Type != JTokenType.Integer && token.Type != JTokenType.Float))
                throw new InvalidDataException("Reply field '" + name + "' must be a finite number.");
            double value = token.Value<double>();
            if (double.IsNaN(value) || double.IsInfinity(value) || value < -float.MaxValue || value > float.MaxValue)
                throw new InvalidDataException("Reply field '" + name + "' exceeds the finite float range.");
            return (float)value;
        }
    }
}
