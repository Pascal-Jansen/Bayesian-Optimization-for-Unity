using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace BOforUnity.Scripts
{
    public class SocketNetwork : MonoBehaviour
    {
        private Socket _serverSocket;
        private IPAddress _ip;
        private IPEndPoint _ipEnd;
        private Thread _connectThread;
        private BoForUnityManager _boManager;

        public float coverage     = 0;
        public float tempCoverage = 0;

        // ───────────────────────────────────────────────────────────────────
        public void InitSocket()
        {
            _boManager = GetComponent<BoForUnityManager>();

            _ip = IPAddress.Parse("127.0.0.1");
            _ipEnd = new IPEndPoint(_ip, 56001);

            _connectThread = new Thread(SocketReceive) {IsBackground = true};
            _connectThread.Start();
        }

        // ───────────────────────────────────────────────────────────────────
        void SocketReceive()
        {
            try
            {
                SocketConnect();
                SendInitInfo();                               // first packet to Python

                var recvData = new byte[2048];
                while (true)
                {
                    int recvLen = _serverSocket.Receive(recvData);
                    if (recvLen == 0) { SocketQuit(); break; }

                    string recvStr = Encoding.ASCII.GetString(recvData, 0, recvLen);
                    try   { ParseMessage(recvStr.Trim()); }
                    catch (Exception ex) { Debug.LogError($"ParseMessage: {ex}"); }
                }
            }
            catch (Exception ex) { Debug.LogError($"SocketReceive: {ex}"); }
        }

        // ───────────────────────────────────────────────────────────────────
        private void ParseMessage(string msg)
        {
            var parts = msg.Split(',');
            if (parts.Length == 0) { Debug.LogWarning("Empty socket msg"); return; }

            switch (parts[0])
            {
                case "parameters":
                    HandleDesignVector(parts.Skip(1).ToArray());
                    break;

                case "optimization_finished":
                    MainThreadDispatcher.Execute(() => {
                        Debug.Log(">>>>>> Optimisation finished!");
                        _boManager.optimizationFinished = true;
                        _boManager.OptimizationDone();
                    });
                    break;

                case "coverage":
                    coverage = float.Parse(parts[1], CultureInfo.InvariantCulture);
                    Debug.Log($"coverage {coverage}");
                    break;

                case "tempCoverage":
                    tempCoverage = float.Parse(parts[1], CultureInfo.InvariantCulture);
                    Debug.Log($"tempCoverage {tempCoverage}");
                    break;

                default:
                    Debug.LogWarning($"Unknown socket tag '{parts[0]}'");
                    break;
            }
        }

        // ------------------------------------------------------------------
        private void HandleDesignVector(string[] valueStrs)
        {
            if (valueStrs.Length != _boManager.parameters.Count)
            {
                Debug.LogError($"Expected {_boManager.parameters.Count} design values, got {valueStrs.Length}");
                return;
            }
            var values = valueStrs.Select(s => float.Parse(s, CultureInfo.InvariantCulture)).ToList();

            MainThreadDispatcher.Execute(() =>
            {
                foreach (var pa in _boManager.parameters)
                    pa.value.Value = values[pa.value.optSeqOrder];

                if (_boManager.initialized)
                    _boManager.OptimizationDone();
                else
                    _boManager.InitializationDone();
            });
        }

        // ───────────────────────────────────────────────────────────────────
        void SocketConnect()
        {
            _serverSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            Debug.Log("Unity socket: connecting…");
            _serverSocket.Connect(_ipEnd);
        }

        // ───────────────────────────────────────────────────────────────────
        // Initialization payload sent once at start ‒ now includes context-dim
        // Header order:
        //   batchSize,numRestarts,rawSamples,
        //   nIterations,mcSamples,nInitial,seed,
        //   #parameters,#objectives,#contexts,
        //   warmStart,initialParamCSV,initialObjCSV
        // ───────────────────────────────────────────────────────────────────
        void SendInitInfo()
        {
            /* ---------- a: BO hyper-parameters + counts ---------- */
            string a =
                $"{_boManager.batchSize},{_boManager.numRestarts},{_boManager.rawSamples}," +
                $"{_boManager.nIterations},{_boManager.mcSamples},{_boManager.nInitial},{_boManager.seed}," +
                $"{_boManager.parameters.Count},{_boManager.objectives.Count}," +
                $"{_boManager.contexts.Count}_" +
                $"{_boManager.warmStart},{_boManager.initialParametersDataPath},{_boManager.initialObjectivesDataPath}";

            /* ---------- b: parameter / objective range blocks ---------- */
            string b = string.Empty;

            /* ---------- c: names in order (parameters, then objectives, then contexts) ---------- */
            string c = string.Empty;

            int seq = 0;
            foreach (var pa in _boManager.parameters)
            {
                pa.value.optSeqOrder = seq++;
                b += pa.value.GetInitInfoStr();
                c += pa.key + ",";
            }
            b = b.TrimEnd('/');
            c = c.TrimEnd(',') + "_";                 // separator before objective names

            seq = 0;
            foreach (var ob in _boManager.objectives)
            {
                ob.value.optSeqOrder = seq++;
                b += "/" + ob.value.GetInitInfoStr();
                c += ob.key + ",";
            }
            b = b.TrimEnd('/');
            c = c.TrimEnd(',') + "_";                 // separator before context names

            /* ---------- context names (no range block needed) ---------- */
            foreach (var ctx in _boManager.contexts)
                c += ctx.key + ",";
            c = c.TrimEnd(',');                       // final trailing comma removed

            /* ---------- d: study IDs ---------- */
            string d = $"{_boManager.userId},{_boManager.conditionId},{_boManager.groupId}";

            /* ---------- final payload ---------- */
            string payload = $"{a}_{b}_{c}_{d}";
            SocketSend(payload);
        }

        // Normalise and clamp to [0,1] (handles zero range gracefully)
        private static float Normalise01(float val, float lo, float hi)
        {
            if (Mathf.Approximately(hi, lo))
                return 0f;                               // avoid divide-by-zero
            return Mathf.Clamp01((val - lo) / (hi - lo));
        }

        // ───────────────────────────────────────────────────────────────────
        /// <summary>
        /// Build objective+context line and push to Python
        /// </summary>
        public void SendObjectives()
        {
            /*  OBJECTIVE VALUES  (as in your original code)
                ------------------------------------------------------------ */
            var objVals = new List<float>();
            foreach (var ob in _boManager.objectives)
            {
                var entry   = ob.value;
                var tmpList = entry.values;

                // keep only the most-recent sub-measures
                int keep = entry.numberOfSubMeasures;
                if (keep < tmpList.Count)
                    tmpList = tmpList.Skip(tmpList.Count - keep).ToList();

                objVals.Add(tmpList.Average());
            }
            if (objVals.Count != _boManager.objectives.Count)
                Debug.LogWarning($"Expected {_boManager.objectives.Count} objectives, got {objVals.Count}");

            /*  CONTEXT VECTOR  (taken from _boManager.contexts)
                Each .value.Value must already be normalised to [0,1]
            ------------------------------------------------------------ */
            var ctxVals = new List<float>();
            foreach (var ctx in _boManager.contexts)
            {
                float raw   = ctx.value.Value;          // sensor value
                ctxVals.Add(Normalise01(raw, 0, 1));  // always returns [0,1]
            }

            if (ctxVals.Count != _boManager.contexts.Count)
                Debug.LogWarning($"Context length {ctxVals.Count} ≠ declared {_boManager.contexts.Count}");

            /*  CONCATENATE  and send  ------------------------------------------------ */
            var allVals = objVals.Concat(ctxVals)
                .Select(v => v.ToString("F3", CultureInfo.InvariantCulture));

            SocketSend(string.Join(",", allVals));
        }

        // ------------------------------------------------------------------
        private void SocketSend(string msg)
        {
            byte[] data = Encoding.ASCII.GetBytes(msg);
            Debug.Log("Unity → Python: " + msg);
            _serverSocket.Send(data, data.Length, SocketFlags.None);
        }

        // ------------------------------------------------------------------
        public void SocketQuit()
        {
            _connectThread?.Interrupt();
            _connectThread?.Abort();
            _serverSocket?.Close();
            _boManager.pythonStarter.StopPythonProcess();
            Debug.Log("Unity disconnected socket");
        }
    }
}
