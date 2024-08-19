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

+ **Note:** The results of the experiment can be seen in *Assets/BOforUnity/BOData/BayesianOptimization/LogData*.

### Configuration
All the configurations, that are neccessary can be made in Unity. To do so, you have to open the folder of the Unity scene which is *Assets/BOforUnity*. Here you click on the *BOforUnityManager* Prefab Asset. In this Asset, all the possible configurations can be made. The different possibilities are explained from top to bottom below.

#### Parameters
The parameters get optimized by the optimizer. In this configuration section you can create, change or remove such parameters.

##### Create parameter
TODO
##### Change parameter
TODO
##### Remove parameter
Select the parameter you want to delete, by clicking on the `=`-symbol at the top left corner of the parameter. Make sure it is higlighted blue as shown in the image. Then click on the `-`-symbol at the bottom of the parameter-collection.

![Remove parameter](./images/remove_parameter.png)

#### Objectives
The objectives are the inputs that the optimizer receives. In this configuration section you can create, change or remove such objectives.

##### Create objective
TODO
##### Change objective
TODO
##### Remove objective
Select the objective you want to delete, by clicking on the `=`-symbol at the top left corner of the objective. Make sure it is higlighted blue as shown in the image. Then click on the `-`-symbol at the bottom of the parameter-collection.

![Remove objective](./images/remove_objective.png)

#### Python Settings
Here you can set the path to Python manually. To do so you have to get the local path of Python 3.11.3 .
* for Windows you can go into a cmd terminal and type *where python*. This will list all installed Python versions. Now you can copy the path of the correct version.
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
TODO

### Known issues
#### Python Warnings
During the experiment, it can happen, that the mobo python script throws warnings about the normalization of the input objectives. Unity treats these warnings as Errors. But these warnings can be ignored and wont affect the result of the optimizer as far as we know.

### License
This project underlies the **MIT License** which can be found in the folder this README is in.