# Bayesian Optimization for Unity

This file just contains additional information about this branch. For detailed information of the main system check out the [README](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/blob/main/README.md) of the main branch.

by [Christian Hummler]

## Table of Contents
* [About](#about)
* [Installation](#installation)
* [Smartwatch Application](#smartwatch-application)
* [Connection with Unity](#connection-with-unity)
* [Implicit Data for Optimizer](#implicit-data-for-optimizer)

## About

The project in this branch contains an alternative method to feed the optimizer with data. In the system of the main branch, the optimizer receives its data by filling out a questionnaire at each iteration step. Here, on the other hand, data is collected by sensors and sent to the optimizer at the end of an iteration. 

The branch also contains a smartwtch application that allows users to send data from the smartwatch to unity. The appropriate script, which opens a socket connection that the smartwatch app uses, is also available.

#### Key Features and Differences
- Smart watch application that can send heart rate data. Could be enhanced to send other data as well. 
- Unity script that can receive data send by the smartwatch.
- Implicit data fetching of optimizer. The Questionaire is removed here. 

#### Links

[Smartwatch Application](/heartrateSender/)

[Websocket Script](Assets/WebSocketWorker.cs)

[BoForUnityManager](Assets/BOforUnity/BoForUnityManager.cs)


#### Usage
The smartwatch can be used as sensor to gather data from users of the program. It will send the data to the opened websocket connection. The data can then be used in Unity to feed the optimizer, which will then be able to use the values. 

![alt text](/imagesHRS/image-1.png)

The program is generally created, so also other sensors or data sources can be used to feed the optimizer. How the data is gathered does not matter to the optimizer, as long as it is available in the right format before the optimization step.

**Main branch: data for optimizer from questions**

![alt text](/imagesHRS/image-12.png)

**This branch data for optimizer from other data sources** 

![alt text](/imagesHRS/image-2.png)



## Installation
These instructions are not intended to get the entire project up and running from the start. You must first follow the instructions in the [README](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/blob/main/README.md) of the main branch. This guide is only for getting the additional files/functions of this branch to run. These are: Smartwatch app, unity script to recieve smartwatch data, altered optimizer. See the following chapters for detailed instruction of how to use the features

## Smartwatch Application

What you will need: 

- Smartwatch (or emulator)
- Android Studio




The easiest way to install the application on your galaxy watch 4 is to open the android studio project and to use the device manager.

1) Open [Smartwatch Application](/heartrateSender/) in android studio. Make sure the project is set up correctly on your system and you can choose "app" before running the project. 

![alt text](/imagesHRS/image-3.png)

2) The important working file here is [MainActivity.kt](heartrateSender/app/src/main/java/com/example/heartratesender/presentation/MainActivity.kt)

3) Follow these instruction to set up a debugger for the project [how to debug smartwatch apps](https://developer.android.com/training/wearables/get-started/debugging)

4)  Now you should be able to run the application. You should see something like this:

![alt text](/imagesHRS/image-4.png)

Its a simple application, that can show the current Heart Rate and tries to send it to a websocket with a given ip adress and port. 


## Connection with Unity

What you will need: 

- Real smartwatch, with smartwatchapp installed. (following these instructions will install the appplication on the watch [how to debug smartwatch apps](https://developer.android.com/training/wearables/get-started/debugging))
- [Websocket Script](Assets/WebSocketWorker.cs) should be running in Unity. 


1) Make sure that both devices (pc and watch) are connected to the same network. (open the terminal on your device and use ipconfig (windows)/ifconfig (mac os) to see the ip adress).

2) Run the example scene in Unity. There should be an empty game object with an attached script. This will open a server which waits for send data.
![alt text](/imagesHRS/image-5.png)

3) You should get this message, which indicates that the server is waiting for data. You could change the port and the way it handles data accordingly.
![alt text](/imagesHRS/image-6.png)

4) Run the smartwatch application and choose the ip adress of your pc which runs the unity project. Also choose the port that was shown in the console message before. Then click connect. It should look somewhat like this: 

![alt text](/imagesHRS/image-9.png)

5) After connecting, you should see your Heartrate again. 

![alt text](/imagesHRS/image-10.png)

6) If everything connected correctly, you should see the heart rate in the log messages of your unity project 

![alt text](/imagesHRS/image-11.png)

7) Now you could use these values to feed the optimizer. 


## Implicit Data for Optimizer

The important file we will work with is [BoForUnityManager](Assets/BOforUnity/BoForUnityManager.cs).

To test if the feature works, there is simulated data available through [Value Simulator](Assets/BOforUnity/Scripts/ValueSimulator.cs). If this file is attached to a game object, it will create random data for heart rate and step count. We will use these as examples to show how the feature works and how to add other data to the optimizer. 

1) Create your Objectives in the BoForUnityManager. Important are: 
 - Lower Bound & Upper Bound should be set.
 - The Names are important because they are used to match objectives to data.
 - Number Of Sub Measures can be ignored here, as it is updated depedning on the list size of the data.

![alt text](/imagesHRS/image-14.png)

2) Now you can run the application. After pressing the next button, you should get to this point 

![alt text](/imagesHRS/image-15.png)

![alt text](/imagesHRS/image-16.png)

3) After clicking "end simulation", the average of data of the objectives will be sent to the optimizer, and you will get to the next iteration step. 

![alt text](/imagesHRS/image-17.png)

![alt text](/imagesHRS/image-18.png)

4) You can repeat the process, until the program terminates.


#### How to add sensors / data

In [BoForUnityManager](Assets/BOforUnity/BoForUnityManager.cs) the method UpdateObjectives() will add data to the objectives when the "end simulation" button is pressed.
![alt text](/imagesHRS/image-19.png)

To add data sources: 

1) Before running the program, add your objective in the inspector (like we did before with HeartRate and Stepcount). Then use the example of HeartRate and StepCount to see how the data is handled.

2) Implement the fetchData method, which will need the name of your objective

![alt text](/imagesHRS/image-20.png)

3) Implement the clearData method. This ensures that data is not sent to the optimizer again in the next iteration step.

![alt text](/imagesHRS/image-21.png)
 