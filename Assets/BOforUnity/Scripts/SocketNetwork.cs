using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BOforUnity.Scripts
{
    // The SocketNetwork class implements a socket client to receive data from a server
    public class SocketNetwork : MonoBehaviour
    {
        private Socket _serverSocket;
        private IPAddress _ip;
        private IPEndPoint _ipEnd;
        private Thread _connectThread;
        public float coverage = 0;
        public float tempCoverage = 0;

        private BoForUnityManager _bomanager;
        
        /// <summary>
        /// The InitSocket method sets up the IP address and port number of the server,
        /// and starts a new thread to receive data from the server
        /// </summary>
        public void InitSocket()
        {
            _bomanager = gameObject.GetComponent<BoForUnityManager>();
            
            _ip = IPAddress.Parse("127.0.0.1");
            _ipEnd = new IPEndPoint(_ip, 56001);
            //_ipEnd = new IPEndPoint(_ip, 56003);
            
            _connectThread = new Thread(new ThreadStart(SocketReceive));
            _connectThread.Start();
        }


        /// <summary>
        /// The SocketReceive method is the main method for receiving data from the server.
        /// It first establishes a connection with the server using the SocketConnect method, then sends an
        /// initialization message to the server using the SendInitInfo method.
        /// </summary>
        void SocketReceive()
        {
            try
            {
                // Establish connection and send data for initialization
                SocketConnect();
                SendInitInfo();

                // Keep receiving data
                while (true)
                {
                    var recvData = new byte[1024];
                    var recvLen = _serverSocket.Receive(recvData);
                    if (recvLen == 0)
                    {
                        Debug.Log("Connection closed by server");
                        SocketQuit();
                        break;
                    }
                    var recvStr = Encoding.ASCII.GetString(recvData, 0, recvLen);

                    try
                    {
                        ParseMessage(recvStr);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"Error in ParseMessage: {ex.Message}\n{ex.StackTrace}");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error in SocketReceive: {ex.Message}\n{ex.StackTrace}");
            }
        }

        /// <summary>
        /// This code defines a function ParseMessage that takes in a string recvStr as input and
        /// parses it to retrieve relevant information. The string is split into an array of
        /// substrings using the comma separator.
        /// If the message type is "parameters", the ParseMessage method extracts the parameter
        /// values from the message and sets the values of the corresponding parameters in the
        /// Optimizer object.
        /// If the message type is "coverage" or "tempCoverage", the ParseMessage method extracts
        /// the coverage value from the message and sets it to the coverage or tempCoverage field
        /// of the SocketNetwork class, respectively.
        /// </summary>
        /// <param name="recvStr"></param>
        private void ParseMessage(string recvStr)
        {
            var strArr = recvStr.Split(',');

            if (strArr.Length != 0)
            {
                if (strArr[0] == "parameters")
                {
                    List<string> strList = strArr.ToList();
                    strList.RemoveAt(0);
                    List<float> floatList = strList.Select(float.Parse).ToList();
                    
                    MainThreadDispatcher.Execute((flist) =>
                    {
                        _bomanager = gameObject.GetComponent<BoForUnityManager>();
                        
                        // Update the parameter values
                        foreach (var pa in _bomanager.parameters)
                        {
                            pa.value.Value = flist[pa.value.optSeqOrder];
                        }

                        if (_bomanager.initialized)
                        {
                            // Tell the manager that the current iteration has finished
                            _bomanager.OptimizationDone();
                        }
                        else {
                            // Tell the manager that the system was initialized
                            _bomanager.InitializationDone();
                        }
                    }, floatList);
                }
                else if (strArr[0] == "optimization_finished")
                {
                    MainThreadDispatcher.Execute(() =>
                    {
                        Debug.Log(">>>>>> Optimization finished!");
                        
                        _bomanager = gameObject.GetComponent<BoForUnityManager>();
                        
                        _bomanager.optimizationFinished = true;
                        _bomanager.OptimizationDone();
                    });
                }
                else if (strArr[0] == "coverage")
                {
                    coverage = Convert.ToSingle(strArr[1]);
                    Debug.Log($"coverage {coverage}");
                }
                else if (strArr[0] == "tempCoverage")
                {
                    tempCoverage = Convert.ToSingle(strArr[1]);
                    Debug.Log($"tempCoverage {tempCoverage}");
                }
                else
                {
                    Debug.LogWarning($"Unknown message type: {strArr[0]}");
                }
            }
            else
            {
                Debug.LogWarning("Received an empty message.");
            }
        }
        
        void SocketConnect()
        {
            try
            {
                if (_serverSocket != null)
                    _serverSocket.Close();

                _serverSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                Debug.Log("Unity is ready to connect...");

                _serverSocket.Connect(_ipEnd);
            }
            catch (SocketException ex)
            {
                if (ex.SocketErrorCode == SocketError.ConnectionRefused)
                {
                    MainThreadDispatcher.Execute(OnSocketConnectionFailed);
                    Debug.LogError("SocketException: " + ex.SocketErrorCode);
                }
                else
                {
                    Debug.LogError("SocketException: " + ex.SocketErrorCode);
                }
            }
        }
        
        private void OnSocketConnectionFailed()
        {
            //SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }
        
        /// <summary>
        /// This function sends the initialization information of the optimization problem to the
        /// server using the SocketSend() function. It first creates an empty string a to which it
        /// adds the initialization information of the parameters and objectives.
        ///
        /// For each parameter and objective in the Optimizer object, it sets its optSeqOrder attribute
        /// to i and increments i. The optSeqOrder attribute represents the order in which the parameters
        /// and objectives were added to the Optimizer object, which is important for correctly parsing
        /// the incoming messages from the server.
        ///
        /// Then, for each parameter and objective, it adds the string returned by the GetInitInfoStr()
        /// method to the string a. The GetInitInfoStr() method returns a string that represents the
        /// initialization information of the parameter or objective in a format that can be parsed by
        /// the server.
        ///
        /// Finally, it logs the initialization information to the console and sends it to the server
        /// using the SocketSend() function.
        /// </summary>
        void SendInitInfo()
        {
            string a = "";
            string b = "";
            string c = "";
            string d = "";
            
            a += $"{_bomanager.batchSize},{_bomanager.numRestarts},{_bomanager.rawSamples}," +
                 $"{_bomanager.nIterations},{_bomanager.mcSamples},{_bomanager.nInitial},{_bomanager.seed}," +
                 $"{_bomanager.parameters.Count},{_bomanager.objectives.Count}_" +
                 $"{_bomanager.warmStart},{_bomanager.initialParametersDataPath},{_bomanager.initialObjectivesDataPath}";

            int i = 0;
            foreach (var pa in _bomanager.parameters)
            {
                pa.value.optSeqOrder = i;
                i++;
                b += pa.value.GetInitInfoStr();
                
                c += pa.key + ",";
            }
            
            b = b.Substring(0, b.Length -1);
            b += "_";
            i = 0;
           
            c = c.Substring(0, c.Length - 1);
            c += "_";

            //Debug.Log("Objective: " + Optimizer.objectives.Count);
            foreach (var ob in _bomanager.objectives)
            {
                ob.value.optSeqOrder = i;
                i++;
                b += ob.value.GetInitInfoStr();
                
                c += ob.key + ",";
            }

            // Delete last / from string a and c
            b = b.Substring(0, b.Length - 1);
            c = c.Substring(0, c.Length - 1);

            d += $"{_bomanager.userId},{_bomanager.conditionId}";
            
            // Send string: Hyper-parameter info _ Parameter info _ Objectives info
            Debug.Log("Send Init Info to Python process: " + a + "_" + b + "_" + c + "_" + d);
            SocketSend(a + "_" + b + "_" + c + "_" + d);
        }
        
        /// <summary>
        /// the SendObjectives method takes in a list of floats called finalObjectives, converts it to
        /// a string sendStr with the objectives separated by commas, and then sends the string over the
        /// socket connection using the SocketSend method.
        ///
        /// Before sending the string, it sets the canRead flag of the Optimizer object to false, which
        /// may indicate that the program is waiting for a response from the server before continuing.
        /// </summary>
        public void SendObjectives()
        {
            var finalObjectives = new List<float>();
            foreach (var ob in _bomanager.objectives)
            {
                var value = ob.value;
                var tmpList = value.values;
                tmpList.RemoveRange(0, value.values.Count - value.numberOfSubMeasures);
                finalObjectives.Add(tmpList.Average());
            }
            
            var sendStr = "";
            foreach (var t in finalObjectives)
            {
                sendStr += t + ",";
            }
            if (sendStr != "")
            {
                sendStr = sendStr.Remove(sendStr.Length - 1);
            }

            SocketSend(sendStr);
        }
        
        /// <summary>
        /// The purpose of this method is to send data over a network socket using a TCP/IP protocol.
        ///
        /// The method first converts the sendStr string to a byte array using the ASCII encoding method,
        /// which returns a byte array containing the ASCII code for each character in the string. The byte
        /// array is then assigned to the sendData variable.
        /// 
        /// Finally, the method calls the Send method of the serverSocket object, passing in the sendData
        /// byte array, the length of the data, and the SocketFlags.None parameter. This sends the data over
        /// the socket connection to the remote endpoint. 
        /// </summary>
        /// <param name="sendStr"></param>
        private void SocketSend(string sendStr)
        {
            byte[] sendData = Encoding.ASCII.GetBytes(sendStr);
            Debug.Log("Unity sending: " + sendStr);
            _serverSocket.Send(sendData, sendData.Length, SocketFlags.None);
        }
        
        /// <summary>
        /// The purpose of this method is to gracefully close the network socket connection and stop the thread
        /// that is managing the socket connection.
        /// </summary>
        public void SocketQuit()
        {
            //close thread
            if (_connectThread != null)
            {
                _connectThread.Interrupt();
                _connectThread.Abort();

            }
            //close socket
            if (_serverSocket != null) { 
                _serverSocket.Close();
            }
            
            //stop the mobo.py
            _bomanager.pythonStarter.StopPythonProcess();
            
            Debug.Log("Unity disconnected socket");
        }
    }
}