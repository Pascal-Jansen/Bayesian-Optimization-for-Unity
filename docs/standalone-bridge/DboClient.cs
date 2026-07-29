// DboClient.cs — Unity client for the dbo-torch optimiser server.
//
// Start the Python side first:
//     dbo-serve --host 127.0.0.1 --port 8756
//
// The protocol is newline-delimited JSON over TCP. Every call blocks on a
// background thread and returns to the Unity main thread via a callback, so
// nothing here stalls the render loop.
//
// Typical use in a study scene:
//
//     var dbo = GetComponent<DboClient>();
//     dbo.Reset(new[] { new[] { -5f, 9f } }, seedPoints: new[] {
//         new[] { 5f }, new[] { 7f }, new[] { 3f } });
//     dbo.Suggest(x => StartCoroutine(RunTrial(x)));
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

        private void OnDestroy() => Disconnect();
        private void OnApplicationQuit() => Disconnect();

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
                 reply => onResult?.Invoke(ParseSuggestion(reply)), onError);
        }

        /// <summary>
        /// Ask for the optimiser's current best estimate, for a validation
        /// iteration. This does not consume an acquisition step.
        /// </summary>
        public void SuggestValidation(Action<DboSuggestion> onResult, Action<string> onError = null)
        {
            Send("{\"cmd\":\"suggest_validation\"}",
                 reply => onResult?.Invoke(ParseSuggestion(reply)), onError);
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
                 reply => onResult?.Invoke(new DboPrediction
                 {
                     Mean = FieldF(reply, "mean"),
                     StdDev = FieldF(reply, "std"),
                 }),
                 onError);
        }

        /// <summary>Write the full run history to a JSON file on the server.</summary>
        public void Save(string path, Action onDone = null, Action<string> onError = null)
        {
            Send("{\"cmd\":\"save\",\"path\":\"" + Escape(path) + "\"}",
                 _ => onDone?.Invoke(), onError);
        }

        // ---------------------------------------------------------------
        // Transport
        // ---------------------------------------------------------------

        private void Send(string payload, Action<string> onResult, Action<string> onError)
        {
            var worker = new Thread(() =>
            {
                string reply;
                try
                {
                    lock (_ioLock)
                    {
                        if (!IsConnected) Connect();
                        if (LogTraffic) Debug.Log("[DBO] -> " + payload);
                        _writer.Write(payload);
                        _writer.Write('\n');
                        reply = _reader.ReadLine();
                    }

                    if (reply == null)
                        throw new IOException("Server closed the connection.");
                    if (LogTraffic) Debug.Log("[DBO] <- " + reply);

                    if (!FieldB(reply, "ok"))
                    {
                        var msg = FieldS(reply, "error") ?? "unknown server error";
                        _mainThread.Enqueue(() => Fail(onError, msg));
                        return;
                    }
                }
                catch (Exception e)
                {
                    var msg = e.Message;
                    Disconnect();   // force a clean reconnect on the next call
                    _mainThread.Enqueue(() => Fail(onError, msg));
                    return;
                }

                _mainThread.Enqueue(() => onResult?.Invoke(reply));
            })
            { IsBackground = true };

            worker.Start();
        }

        private static void Fail(Action<string> onError, string message)
        {
            if (onError != null) onError(message);
            else Debug.LogError("[DBO] " + message);
        }

        // ---------------------------------------------------------------
        // Minimal JSON helpers
        //
        // The server's replies are flat and machine-generated, so a full JSON
        // parser would be overkill. JsonUtility is not used because it cannot
        // handle top-level arrays of primitives.
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

        private static string FieldS(string json, string key)
        {
            int i = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (i < 0) return null;
            i = json.IndexOf(':', i);
            if (i < 0) return null;
            i++;
            while (i < json.Length && char.IsWhiteSpace(json[i])) i++;
            if (i >= json.Length || json[i] != '"') return null;
            var sb = new StringBuilder();
            for (i++; i < json.Length && json[i] != '"'; i++)
            {
                if (json[i] == '\\' && i + 1 < json.Length) i++;
                sb.Append(json[i]);
            }
            return sb.ToString();
        }

        private static bool FieldB(string json, string key)
        {
            // RawField tolerates whitespace after the colon; Python's
            // json.dumps emits '": true' with a space, so an offset-based
            // check would reject every server reply.
            return RawField(json, key) == "true";
        }

        private static float FieldF(string json, string key)
        {
            var raw = RawField(json, key);
            return raw != null && float.TryParse(
                raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var f) ? f : float.NaN;
        }

        private static int FieldI(string json, string key)
        {
            var raw = RawField(json, key);
            return raw != null && int.TryParse(
                raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) ? n : -1;
        }

        private static string RawField(string json, string key)
        {
            int i = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (i < 0) return null;
            i = json.IndexOf(':', i);
            if (i < 0) return null;
            int start = ++i;
            while (start < json.Length && char.IsWhiteSpace(json[start])) start++;
            int end = start;
            while (end < json.Length && json[end] != ',' && json[end] != '}' && json[end] != ']') end++;
            return json.Substring(start, end - start).Trim();
        }

        private static float[] FieldFArray(string json, string key)
        {
            int i = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (i < 0) return Array.Empty<float>();
            int open = json.IndexOf('[', i);
            int close = json.IndexOf(']', open + 1);
            if (open < 0 || close < 0) return Array.Empty<float>();

            var body = json.Substring(open + 1, close - open - 1).Trim();
            if (body.Length == 0) return Array.Empty<float>();

            var parts = body.Split(',');
            var result = new List<float>(parts.Length);
            foreach (var p in parts)
            {
                if (float.TryParse(p.Trim(), NumberStyles.Float,
                                   CultureInfo.InvariantCulture, out var f))
                    result.Add(f);
            }
            return result.ToArray();
        }

        private static DboSuggestion ParseSuggestion(string reply) => new DboSuggestion
        {
            X = FieldFArray(reply, "x"),
            Iteration = FieldI(reply, "iteration"),
            IsValidation = FieldB(reply, "is_validation"),
        };
    }
}
