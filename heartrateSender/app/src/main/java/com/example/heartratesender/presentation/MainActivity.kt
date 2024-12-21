package com.example.heartratesender.presentation

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.Text
import com.example.heartratesender.presentation.theme.HeartrateSenderTheme
import org.java_websocket.client.WebSocketClient
import org.java_websocket.handshake.ServerHandshake
import java.net.URI
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var heartRateSensor: Sensor? = null
    private var heartRate by mutableStateOf("No Data")
    private lateinit var webSocketClient: WebSocketClient
    private val BODY_SENSORS_PERMISSION_REQUEST_CODE = 1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)

        checkBodySensorsPermission()

        // Load saved IP address and port from SharedPreferences if connected before
        val sharedPreferences = getSharedPreferences("AppPreferences", Context.MODE_PRIVATE)
        val savedIpAddress = sharedPreferences.getString("ipAddress", "")
        val savedPort = sharedPreferences.getString("port", "")

        setContent {
            WearApp(heartRate, savedIpAddress ?: "", savedPort ?: "") { ipAddress, port ->
                startWebSocketConnection(ipAddress, port)
                saveIpAndPort(ipAddress, port)
            }
        }
    }

    private fun checkBodySensorsPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.BODY_SENSORS),
                BODY_SENSORS_PERMISSION_REQUEST_CODE
            )
        } else {
            registerHeartRateListener()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == BODY_SENSORS_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                registerHeartRateListener()
            } else {
                Toast.makeText(this, "Heart rate sensor permission denied.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // Register heart rate sensor listener
    private fun registerHeartRateListener() {
        heartRateSensor?.also { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    // Unregister listener when app is paused
    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    // Sensor event listener for heart rate
    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_HEART_RATE) {
            heartRate = event.values[0].toInt().toString()
            Log.d("HeartRateApp", "Heart rate: $heartRate")

            // Send heart rate data if WebSocket is open
            if (::webSocketClient.isInitialized && webSocketClient.isOpen) {
                webSocketClient.send(heartRate)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // Start WebSocket connection after IP and Port are entered
    private fun startWebSocketConnection(ipAddress: String, port: String) {
        val uri = URI("ws://$ipAddress:$port/HeartRate")
        webSocketClient = object : WebSocketClient(uri) {
            override fun onOpen(handshakedata: ServerHandshake?) {
                Log.d("WebSocketClient", "WebSocket Connection Opened")
            }

            override fun onMessage(message: String?) {
                Log.d("WebSocketClient", "Received message: $message")
            }

            override fun onClose(code: Int, reason: String?, remote: Boolean) {
                Log.e("WebSocketClient", "WebSocket Closed: $reason")
            }

            override fun onError(ex: Exception?) {
                Log.e("WebSocketClient", "WebSocket Error: ${ex?.message}")
            }
        }
        webSocketClient.connect()
    }

    // Save IP and port in SharedPreferences
    private fun saveIpAndPort(ipAddress: String, port: String) {
        val sharedPreferences = getSharedPreferences("AppPreferences", Context.MODE_PRIVATE)
        val editor = sharedPreferences.edit()
        editor.putString("ipAddress", ipAddress)
        editor.putString("port", port)
        editor.apply()  // Save the IP and port
    }

    // Composable function for the main UI
    @Composable
    fun WearApp(
        heartRate: String,
        savedIpAddress: String,
        savedPort: String,
        onConnect: (String, String) -> Unit
    ) {
        val navController = rememberNavController()

        NavHost(navController, startDestination = "main_screen") {
            composable("main_screen") {
                MainScreen(navController = navController, heartRate = heartRate)
            }
            composable("ip_input_screen") {
                IpInputScreen(
                    navController = navController,
                    savedIpAddress = savedIpAddress,
                    savedPort = savedPort,
                    onConnect = onConnect
                )
            }
        }
    }

    // Main Screen that shows Heart Rate and a button to navigate to IP Input Screen
    @Composable
    fun MainScreen(navController: androidx.navigation.NavHostController, heartRate: String) {
        Column(
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
                .padding(16.dp)
        ) {
            Text(text = "Heart Rate: $heartRate bpm", color = Color.White, fontSize = 20.sp)
            Spacer(modifier = Modifier.height(20.dp))
            Button(
                onClick = { navController.navigate("ip_input_screen") },
                modifier = Modifier.wrapContentWidth().padding(8.dp).size(70.dp)
            ) {
                Text("IP & Port")
            }
        }
    }

    // IP Input Screen where user can enter IP Address and Port, then connect
    @Composable
    fun IpInputScreen(
        navController: androidx.navigation.NavHostController,
        savedIpAddress: String,
        savedPort: String,
        onConnect: (String, String) -> Unit
    ) {
        var ipAddress by remember { mutableStateOf(savedIpAddress) }
        var port by remember { mutableStateOf(savedPort) }

        Column(
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .background(Color.Black)
        ) {
            BasicTextField(
                value = ipAddress,
                onValueChange = { ipAddress = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp)
                    .background(Color.Gray)
                    .padding(8.dp),
                textStyle = TextStyle(color = Color.White, fontSize = 16.sp),
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Text,
                    imeAction = ImeAction.Next
                ),
                decorationBox = { innerTextField ->
                    if (ipAddress.isEmpty()) {
                        Text(
                            text = "Enter IP Address (xxx.xxx...)",  // Placeholder text
                            style = TextStyle(color = Color.LightGray, fontSize = 16.sp)
                        )
                    }
                    innerTextField()  // This is the actual input field
                }
            )
            Spacer(modifier = Modifier.height(8.dp))
            BasicTextField(
                value = port,
                onValueChange = { port = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp)
                    .background(Color.Gray)
                    .padding(8.dp),
                textStyle = TextStyle(color = Color.White, fontSize = 16.sp),
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Number,
                    imeAction = ImeAction.Done
                ),
                decorationBox = { innerTextField ->
                    if (port.isEmpty()) {
                        Text(
                            text = "Enter Port (e.g. 3000)",  // Placeholder text
                            style = TextStyle(color = Color.LightGray, fontSize = 16.sp)
                        )
                    }
                    innerTextField()  // This is the actual input field
                }
            )
            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = {
                if (ipAddress.isNotEmpty() && port.isNotEmpty()) {
                    onConnect(ipAddress, port)
                    navController.popBackStack()
                }
            }, modifier = Modifier
                .wrapContentWidth().padding(8.dp)
                .size(60.dp)
            ) {
                Text("Connect")
            }
        }
    }
}
