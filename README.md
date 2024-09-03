# Bayesian Optimization for Unity

by [Pascal Jansen](https://scholar.google.de/citations?user=cR1_0-EAAAAJ&hl=en)

### About

This Unity asset integrates Bayesian Optimization (based on [botorch.org](https://botorch.org/)) into your projects, enabling the optimization of design parameters to maximize or minimize objective values. It utilizes a **Human-in-the-Loop** approach, iteratively querying user feedback on designs to refine parameter values.

Key features include:
- Set optimization hyperparameters directly in Unity.
- Automatic communication between Unity and the Python process running the botorch-based implementation.

### Usage

#### Human-in-the-Loop Process

To utilize the Human-in-the-Loop optimization, this asset requires the [QuestionnaireToolkit](https://assetstore.unity.com/packages/tools/gui/questionnairetoolkit-157330) for collecting explicit subjective feedback from users. This feedback serves as the design objective value in the optimization process. 

Example use case:
- Measure the usability of a design using the System Usability Scale (SUS) and use this data for optimization.

#### Limitations

Currently, there is no *implicit* feedback variant of this asset. It cannot, for example, process physiological data from users as input for design objective values.

### Versions
Currently there are two example versions of the bayesian optimization for unity project. They are separated in branches.

#### Main-Branch
In this version, the example is based on one unity-scene, meaning there is no real change of scenes. The interface gets replaced by the next one in the same unity-scene instead.

#### Multi-Scene-Branch
In this version, the example is based on multiple unity-scenes. This means, everytime you see a new interface, unity switched to another scene.

### Installation
This is a step-by-step explanation how you get this asset running on your system.
1. Clone the repository 
2. Launch the installation_python.bat (or the install_python.sh for MacOS) to install Python and the library requirements.
These files are located in *Assets/StreamingAssets/BOData/Installation*

* **Note:** If you have Python already installed locally you have to set the path to python manually. This is explained in the [Python Settings](#python-settings) chapter.

3. Download Unity Hub
4. Create or Login to your student-licensed Unity-account
5. Install Unity 2022.3.21f1
6. Add the project in the Unity Hub by selecting the repository folder
7. Open the project and set the [Python Settings](#python-settings) accordingly

### Example Usage
In this chapter this Asset is explained by going through the Demo experiment step-by-step once.
1. Press the play button in the top middle of the screen
2. Press the next button in the screen and wait for the system to load. Then press next again.
3. Now, the simulation is shown. In this case you can see two shapes with colors to evaluate.
4. If you are done, you can press the `End Simulation` button. Now a questionnaire appears, where you have to evaluate the simulation.
5. Answer the questions accordingly and press `Finish` if you are done. Now the optimizer will save these inputs and change the simulation parameters.
6. Press next to start a new iteration. Now the process will start from step `3.` again until the set number of iterations is reached. Then the system tells you, that you can close the application now.

* **Note:** The results of the experiment can be seen in *Assets/BOforUnity/BOData/BayesianOptimization/LogData*.

### Configuration
All the configurations, that are neccessary can be made in Unity. To do so, you have to open the folder of the Unity scene which is *Assets/BOforUnity*. Here you click on the *BO-example-scene.unity* and open the scene. After that, select the *BOforUnityManager* Object on the left (blue) and click on *select* in the top of the inspector. Now you can change the settings accordingly. In this Object, all the possible configurations can be made. The different possibilities are explained from top to bottom below.

#### Parameters
The parameters get optimized by the optimizer. In this configuration section you can create, change or remove such parameters.

##### Create parameter
Click on the `+`-symbol at the bottom of the parameter collection. A new prefilled parameter appeares which has to be edited accordingly. How it can be edited is explained [here](#change-parameter).

* **Note:** Make sure, the added parameter is used in your simulation.
* **Note:** It is recommended, to store the previous *ObservationPerEvaluation.csv* in *Assets/BOforUnity/BOData/BayesianOptimization/LogData* and delete it in this folder afterwards, to make sure the header is correct.

##### Change parameter
Settable options are explained from top to bottom.

###### Value
Here the value, that the optimizer gave the parameter after optimization, can be seen.

###### Lower/Upper Bound
Caps the values of the objective to a range.

###### Is Discrete
If this box is checked, a the optimizer generates discrete values for this parameter.

###### Step
Only important, if the parameter is discrete. This sets the step size of the value (for example 1 means, that every number is valid, 2 means just every even number is valid, 3 very third number and so on).

###### Script Reference
Used to automatically find the correct parameter in unity by the `UpdateParameter`-method in the *Optimizer.cs*-file.

###### Variable Name 
Used to automatically find the correct parameter in unity by the `UpdateParameter`-method in the *Optimizer.cs*-file.

###### Game Object Name
If there are more game objects that are optimized, the game object that the parameter refers to can be set here.

###### Script Name
Used to automatically find the correct parameter in unity by the `UpdateParameter`-method in the *Optimizer.cs*-file.

* **Note:** The `UpdateParameter`-method shouldn't be used if possible, because it often happens, that the exact script can't be found even with setting the above parameters. Instead get the needed value through the parameter list in the *BOforUnityManager* by selecting the correct index.

##### Remove parameter
Select the parameter you want to delete, by clicking on the `=`-symbol at the top left corner of the parameter. Make sure it is higlighted blue as shown in the image. Then click on the `-`-symbol at the bottom of the parameter-collection.

![Remove parameter](./images/remove_parameter.png)

#### Objectives
The objectives are the inputs that the optimizer receives. In this configuration section you can create, change or remove such objectives.

##### Create objective
Click on the `+`-symbol at the bottom of the objective collection. A new prefilled objective appeares which has to be edited accordingly. How it can be edited is explained [here](#change-objective).

* **Note:** An objective needs to receive a value before the optimization step to make it work. In this demo this can be achieved, by creating a new question in the questionnaire or by changing one question for a different objective to the new objective. How you do this, is explained below.
* **Note:** It is recommended, to store the previous *ObservationPerEvaluation.csv* in *Assets/BOforUnity/BOData/BayesianOptimization/LogData* and delete it in this folder afterwards, to make sure the header is correct.

###### Create question
In the hierarchy of the *BO-example-scene* go to *QTQuestionnaireManager/QuestionPage-1*. Then on the right you can see *Question Item Creation*. Choose the inputs as needed (the *Header Name* needs tobe the same as the according objective name). After that click on *Create Item* and the question gets added. Now you can edit the question by following the next paragraph.

###### Change existing question
In the hierarchy of the *BO-example-scene* go to *QTQuestionnaireManager/QuestionPage-1/Scroll View/Viewpoint/Content/* and select the question, you want to change. To make the question count for the new objective the *Header Name* of the question needs to be the same as th eobjective name.

##### Change objective
Settable options are explained from top to bottom.

###### Number of Sub
This setting says, how many values exist for this object (in this case how many questions). The number has to be at least 1.

###### Values
Here the values can be seen, after the questionnaire is complete.

###### Lower/Upper Bound
Caps the values of the objective to a range.

###### Smaller is Better
If this box is checked, a small value is better than a high one (default: Higher is better).

##### Remove objective
Select the objective you want to delete, by clicking on the `=`-symbol at the top left corner of the objective. Make sure it is higlighted blue as shown in the image. Then click on the `-`-symbol at the bottom of the objective-collection.

![Remove objective](./images/remove_objective.png)

#### Python Settings
Here you can set the path to Python manually. To do so you have to get the local path of Python 3.11.3 .
* for Windows you can go into a cmd terminal and type `where python`. This will list all installed Python versions. Now you can copy the path of the correct version.
* for Linux you can go into the terminal and type `whereis python3.11` or `which python3.11`. Now you can copy the path.

After that you can tick the box in the *Python Settings* section in the *BOforUnityManager* and paste the copied path in the upcoming text field. You do not need to strip anything of the path.
If you downloaded python via the installation script, you can simply uncheck the box.

#### Study Settings
Here you can fill in the ID of the user as well as the ID of the current condition you are running th eexperiment on. This ensures sortable results.

#### Warm Start Settings
If you check the box for warm start, the initial rounds are skipped. This means, the optimizer will start to optimize from the first iteration on. 
Not checking the box results into the default start. Then the optimizer uses specific parameter-values and collects the objective values without optimizing. after the set amount if inital iterations the optimizer uses all of the collected values to start optimizing. 

#### BO-Hyper-parameters 
In this section you can configure, how many iterations the experiment should have. The total amount of iterations is the sum of the amount of intial rounds and the amount of normal iterations. You can set both of these values.

### Portability to your own Project
If you want to use this optimization tool in your own project, you can simply export it as a unity package and import it in your project. To do so, make sure, you are in the *Asset/BOforUnity* Folder in the Unity project hierarchy. If this is the case, click on *Assets* in the top menu and then click on *Export package*. Now select all files you want to include in your own project. Click on *Export* and save the package. To include it in your project, simply click on *Import package -> Custom package* in the *Assets* menu and select the saved package.

### Known issues
#### Python Warnings
During the experiment, it can happen, that the mobo python script throws warnings about the normalization of the input objectives. Unity treats these warnings as Errors. But these warnings can be ignored and wont affect the result of the optimizer as far as we know.

### License
This project underlies the **MIT License** which can be found in the folder this README is in.
