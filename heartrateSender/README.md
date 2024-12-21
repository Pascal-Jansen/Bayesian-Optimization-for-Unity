# Heart Rate Sender for Galaxy Watch 4

by Christian Hummler

This folder contains a seperate android studio project with an app written for the Samsung Galaxy Watch 4.

### Installation

The easiest way to install the application on your galaxy watch 4 is to open the android studio project and to use the device manager. 

    1) Make sure that both devices (pc and watch) are connected to the same network.
    
    2) On the watch, locate developer options and activate the option "allow adb debugging". 
    
    3) Open the device manager in android studio and choose "pair device using wifi". Follow the steps to pair the watch. 
    
    4) Run the app and it should get installed on your watch. You can now also use the app without connecting to the computer. 


### Usage

To receive data from the smartwatch in unity, you need to make sure you have the according C# script set up and it is running. In the script you can adjust the port where you want to receive the data. First, make sure your smartwatch and your computer are connected to the same network. After the application is open on the smartwatch, you can switch to another screen where the ip adress of the other device and the port is needed. These need to match the data of your other device. When all is set up correctly, you should see the heart rate data in the console of your running unity project. 
