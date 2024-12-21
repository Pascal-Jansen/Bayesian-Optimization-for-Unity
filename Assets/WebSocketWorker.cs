using WebSocketSharp;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp.Server;

public class HeartrateReciever : MonoBehaviour
{
    WebSocketServer wssv;

    void Start()
    {
        wssv = new WebSocketServer(System.Net.IPAddress.Any, 9090); 
        wssv.AddWebSocketService<HeartRateBehavior>("/HeartRate");
        wssv.Start();

        // this adress should match the adress in the smartwatch app. 
        // localhost is the ip adress and the number after : is the port
        Debug.Log("WebSocket Server started on ws://localhost:3000/HeartRate");
    }

    void OnDestroy()
    {
        if (wssv != null)
        {
        wssv.Stop();
        wssv = null;
        }

    }
}

public class HeartRateBehavior : WebSocketBehavior
{
    protected override void OnOpen()
    {
        Debug.Log("New WebSocket connection opened.");
    }
    protected override void OnMessage(MessageEventArgs e)
    {
        string heartRate = e.Data;
        Debug.Log("Received heart rate: " + heartRate);
        // Heart rate data can be used now. It should be updated every second
    }
    protected override void OnClose(CloseEventArgs e)
    {
        Debug.Log($"WebSocket connection closed: {e.Reason}");
    }
}

