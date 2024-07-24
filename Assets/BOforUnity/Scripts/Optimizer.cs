using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

// The Optimizer class manages optimization settings and parameters for the application.
// It provides methods to start and control the optimization process, add and retrieve parameters,
// and perform various optimization-related tasks. This class serves as the core component
// for managing optimization behavior.
namespace BOforUnity.Scripts
{
    public class Optimizer : MonoBehaviour
    {
        private const bool Continuous = false;
        private const bool Discrete = true;
        private const bool BiggerIsBetter = false;
        private const bool SmallerIsBetter = true;
        private const int Likert7Scale = 7;
        private const int Likert5Scale = 5;

        private List<Dictionary<string, object>> _csvData;

        private BoForUnityManager _bomanager;
        
        public void Start()
        {
            _bomanager = gameObject.GetComponent<BoForUnityManager>();
        }
        
        public void DebugOptimizer()
        {
            Debug.Log("Debug: Add Parameters");

            // Add Parameters with value ranges and if it is discrete/continuous
            AddParameter("Trajectory", 0f, 1f, Continuous);
            AddParameter("TrajectoryAlpha", 0.1f, 1f, Continuous);
            AddParameter("TrajectorySize", 0.1f, 0.6f, Continuous);
            AddParameter("EgoTrajectory", 0f, 1f, Continuous);
            AddParameter("EgoTrajectoryAlpha", 0.1f, 1f, Continuous);
            AddParameter("EgoTrajectorySize", 0.1f, 0.6f, Continuous);
            AddParameter("PedestrianIntention", 0f, 1f, Continuous);
            AddParameter("PedestrianIntentionSize", 0.1f, 0.2f, Continuous);
            AddParameter("SemanticSegmentation", 0f, 1f, Continuous);
            AddParameter("SemanticSegmentationAlpha", 0.1f, 1f, Continuous);
            AddParameter("CarStatus", 0f, 1f, Continuous);
            AddParameter("CarStatusAlpha", 0.1f, 1f, Continuous);
            AddParameter("CoveredArea", 0f, 1f, Continuous);
            AddParameter("CoveredAreaAlpha", 0.1f, 1f, Continuous);
            AddParameter("CoveredAreaSize", 0.2f, 0.8f, Continuous);
            AddParameter("OccludedCars", 0f, 1f, Continuous);

            // Add objectives with value ranges and optimization preferences.
            //addObjective("Trust", 1f, 5f, BIGGER_IS_BETTER);
            //addObjective("Understanding", 1f, 5f, BIGGER_IS_BETTER);
            //addObjective("MentalLoad", 1f, 20f, SMALLER_IS_BETTER);
            //addObjective("PerceivedSafety", -3f, 3f, BIGGER_IS_BETTER);
            //addObjective("Aesthetics", 1f, 7f, BIGGER_IS_BETTER);
            //addObjective("Acceptance", 1f, 7f, BIGGER_IS_BETTER);

            //ParamController = GameObject.FindGameObjectWithTag("ParameterController");
            //trajectoryParameter = ParamController.GetComponent<TrajectoryParameter>();
            //egoTrajectoryParameter = ParamController.GetComponent<EgoTrajectoryParameter>();
            //intentionParameter = ParamController.GetComponent<IntentionParameter>();
            //semSegParameter = ParamController.GetComponent<SemanticSegmentationParameter>();
            //carStatusParameter = ParamController.GetComponent<CarStatusParameter>();
            //arcParameter = ParamController.GetComponent<ARC4ADController>();
        }

        /// <summary>
        /// This method updates the values of the parameters in the optimization process using the data in the
        /// CSV file at the specified currentIndex. It loops through each parameter in the parameters dictionary
        /// and sets its Value property to the corresponding value in the CSV file at the given index. After
        /// updating the parameter values, it starts a coroutine called ShowIndexChange, which likely shows some
        /// kind of visual feedback to the user that the parameter values have changed.
        /// </summary>
        /// <param name="currentIndex"></param>
        public void UpdateParameter(int currentIndex)
        {
            foreach (var pa in _bomanager.parameters)
            {
                pa.value.Value = float.Parse(_csvData[currentIndex][pa.key].ToString());
            }
        }

        /// <summary>
        /// The first method, addParameter(string name, float lowerBound, float upperBound, float step, bool isDiscrete),
        /// adds a new parameter with the given name, lowerBound, upperBound, step (if isDiscrete is true), and stores it
        /// in the parameters dictionary.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        /// <param name="step"></param>
        /// <param name="isDiscrete"></param>
        public void AddParameter(string name, float lowerBound, float upperBound, float step, bool isDiscrete)
        {
            try
            {
                _bomanager.parameters.Add(new ParameterEntry(name, new ParameterArgs(lowerBound, upperBound, (isDiscrete) ? step : 0)));
            }
            catch (ArgumentException)
            {
                //Debug.LogError($"An element with Key = {name} already exists.", Instance);
            }
        }
    
        /// <summary>
        /// The second method, addParameter(string name, float lowerBound, float upperBound, float step, bool isDiscrete),
        /// adds a new parameter with the given name, lowerBound, upperBound, step (if isDiscrete is true), and stores it
        /// in the parameters dictionary. "step" is always 1 if using this method.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        /// <param name="step"></param>
        /// <param name="isDiscrete"></param>
        public void AddParameter(string name, float lowerBound, float upperBound, bool isDiscrete)
        {
            int step = 1;
            try
            {
                _bomanager.parameters.Add(new ParameterEntry(name, new ParameterArgs(lowerBound, upperBound, (isDiscrete) ? step : 0)));
            }
            catch (ArgumentException)
            {
                //Debug.LogError($"An element with Key = {name} already exists.", Instance);
            }
        }
    
        /// <summary>
        /// The third method, addParameter(string name, float lowerBound, float upperBound, bool isDiscrete, ref float reference),
        /// is also similar to the first method, but additionally takes a ref float parameter called reference. This method stores
        /// the ParameterArgs object in the parameters dictionary with a reference to the reference parameter, so that when the
        /// parameter is updated during optimization, the corresponding reference value is also updated. "step" is always 1 if using this method.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        /// <param name="isDiscrete"></param>
        /// <param name="reference"></param>
        public void AddParameter(string name, float lowerBound, float upperBound, bool isDiscrete, ref float reference)
        {
            int step = 1;
            try
            {
                _bomanager.parameters.Add(new ParameterEntry(name, new ParameterArgs(lowerBound, upperBound, (isDiscrete) ? step : 0, ref reference)));
            }
            catch (ArgumentException)
            {
                //Debug.LogError($"An element with Key = {name} already exists.", Instance);
            }
        }

        /// <summary>
        /// This is a public static method that takes a string parameter name and returns a float value. The method first initializes a
        /// float variable value to zero. It then attempts to retrieve a value associated with the name parameter from a dictionary
        /// named parameters using the square bracket syntax. If the key is not found in the dictionary, a KeyNotFoundException is thrown,
        /// and an error message is logged to the console using the Debug.LogError method. Finally, the method returns the value variable.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public float GetParameterValue(string name)
        {
            var value = 0.0f;
            //Debug.Log("Parameters: " + parameters[name].Value);
            try
            {
                foreach (var pa in _bomanager.parameters)
                {
                    if (pa.key == name)
                    {
                        value = pa.value.Value;
                    }
                }
            }
            catch (KeyNotFoundException)
            {
                //Debug.LogError(String.Format("Key = {0} is not found.", name), Instance);
            }
            return value;
        }


        /// <summary>
        /// The method getParameter(string name) takes a string parameter name and returns a ParameterArgs object. It retrieves a value
        /// associated with the name parameter from a dictionary named parameters. If the key is not found in the dictionary, an error message
        /// is logged to the console, and a default ParameterArgs object is returned.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public ParameterArgs GetParameter(string name)
        {
            ParameterArgs value = new ParameterArgs();
            try
            {
                foreach (var pa in _bomanager.parameters)
                {
                    if (pa.key == name)
                    {
                        value = pa.value;
                    }
                }
            }
            catch (KeyNotFoundException)
            {
                //Debug.LogError(String.Format("Key = {0} is not found.", name), Instance);
            }
            return value;
        }


        /// <summary>
        /// The method addObjective(string name, ObjectiveArgs args) takes a string parameter name and an ObjectiveArgs object args. It adds
        /// the args object to the objectives dictionary with a key of name. If the key already exists in the dictionary, an error message
        /// is logged to the console.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="args"></param>
        public void AddObjective(string name, ObjectiveArgs args)
        {
            foreach (var ob in _bomanager.objectives)
            {
                if (ob.key == name)
                {
                    // if found in the list ... update the value
                    ob.value = args;
                    return;
                }
            }
            // if not found in the list ... add as new entry
            _bomanager.objectives.Add(new ObjectiveEntry(name, args));
        }

        public void AddObjectiveValue(string name, float currVal)
        {
            foreach (var ob in _bomanager.objectives)
            {
                if (name.Contains(ob.key))
                {
                    // if name found in the list ... add the current value to the list of values
                    ob.value.values.Add(currVal);
                    return;
                }
            }
        }


        /// <summary>
        /// The method addObjective(string name, float lowerBound, float upperBound, bool smallerIsBetter = false) takes a string parameter name,
        /// a float parameter lowerBound, a float parameter upperBound, and an optional boolean parameter smallerIsBetter. It creates a new
        /// ObjectiveArgs object with the lowerBound, upperBound, and smallerIsBetter values and adds the object to the objectives dictionary with
        /// a key of name. If the key already exists in the dictionary, an error message is logged to the console.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        /// <param name="numberOfSubMeasures"></param>
        /// <param name="smallerIsBetter"></param>
        public void AddObjective(string name, float lowerBound, float upperBound, int numberOfSubMeasures, bool smallerIsBetter = false)
        {
            foreach (var ob in _bomanager.objectives)
            {
                if (ob.key == name)
                {
                    // if found in the list ... update the values
                    ob.value.lowerBound = lowerBound;
                    ob.value.upperBound = upperBound;
                    ob.value.smallerIsBetter = smallerIsBetter;
                    ob.value.numberOfSubMeasures = numberOfSubMeasures;
                    return;
                }
            }
            // if not found in the list ... add as new entry
            _bomanager.objectives.Add(new ObjectiveEntry(name, new ObjectiveArgs(lowerBound, upperBound, smallerIsBetter,numberOfSubMeasures)));
        }


        /// <summary>
        /// The method getObjective(string name) takes a string parameter name and returns the ObjectiveArgs object associated with the name key in the
        /// objective dictionary. If the key is not found in the dictionary, an error message is logged to the console, and a default ObjectiveArgs
        /// object is returned.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public ObjectiveArgs GetObjective(string name)
        {
            ObjectiveArgs value = new ObjectiveArgs();
            foreach (var ob in _bomanager.objectives)
            {
                if (ob.key == name)
                {
                    value = ob.value;
                }
            }
            return value;
        }


        // Update design parameters based on the current optimization values.
        // Retrieve parameter values from the optimizer and set them in related components.
        // Control various design features based on parameter values.
        public void UpdateDesignParameters()
        {
            Debug.Log("Updating Parameters");
            float trajectoryValue = GetParameterValue("Trajectory");
            float trajectoryAlphaValue = GetParameterValue("TrajectoryAlpha");
            float trajectorySizeValue = GetParameterValue("TrajectorySize");

            float egoTrajectoryValue = GetParameterValue("EgoTrajectory");
            float egoTrajectoryAlphaValue = GetParameterValue("EgoTrajectoryAlpha");
            float egoTrajectorySizeValue = GetParameterValue("EgoTrajectorySize");

            float intentionValue = GetParameterValue("PedestrianIntention");
            float intentionSizeValue = GetParameterValue("PedestrianIntentionSize");

            float semSegValue = GetParameterValue("SemanticSegmentation");
            float semSegAlphaValue = GetParameterValue("SemanticSegmentationAlpha");

            float carStatusValue = GetParameterValue("CarStatus");
            float carStatusAlphaValue = GetParameterValue("CarStatusAlpha");

            float coveredAreaValue = GetParameterValue("CoveredArea");
            float coveredAreaAlphaValue = GetParameterValue("CoveredAreaAlpha");
            float coveredAreaSizeValue = GetParameterValue("CoveredAreaSize");

            float occludedCarsValue = GetParameterValue("OccludedCars");
            
            /*
            trajectoryParameter.setTrajectoryAlpha(trajectoryAlphaValue);
            trajectoryParameter.setTrajectorySize(trajectorySizeValue);

            egoTrajectoryParameter.setEgoTrajectoryAlpha(egoTrajectoryAlphaValue);
            egoTrajectoryParameter.setEgoTrajectorySize(egoTrajectorySizeValue);

            intentionParameter.setIntentionSize(intentionSizeValue);

            semSegParameter.setSemSegAlpha(semSegAlphaValue);

            carStatusParameter.setAlphaCarStatus(carStatusAlphaValue);

            arcParameter.setCoveredAreaAlpha(coveredAreaAlphaValue);
            arcParameter.setCoveredAreaSize(coveredAreaSizeValue);


            if (trajectoryValue < 0.5f)
            {
                trajectoryParameter.setTrajectory(false);
            }
            else if (trajectoryValue >= 0.5f)
            {
                trajectoryParameter.setTrajectory(true);
            }

            if (egoTrajectoryValue < 0.5f)
            {
                egoTrajectoryParameter.setEgoTrajectory(false);
            }
            else if (egoTrajectoryValue >= 0.5f)
            {
                egoTrajectoryParameter.setEgoTrajectory(true);
            }


            if (intentionValue < 0.5f)
            {
                intentionParameter.setIntentionUI(false);
            }
            else if (intentionValue >= 0.5f)
            {
                intentionParameter.setIntentionUI(true);
            }


            if(semSegValue < 0.5f)
            {
                semSegParameter.setSemanticSegmentation(false);
            }
            else if (semSegValue >= 0.5f)
            {
                semSegParameter.setSemanticSegmentation(true);
            }


            if (carStatusValue < 0.5f)
            {
                carStatusParameter.setCarStatus(false);
            }
            else if (carStatusValue >= 0.5f)
            {
                carStatusParameter.setCarStatus(true);
            }


            if (coveredAreaValue < 0.5f)
            {
                arcParameter.setCoveredArea(false);
            }
            else if (coveredAreaValue >= 0.5f)
            {
                arcParameter.setCoveredArea(true);
            }
            

            if (occludedCarsValue < 0.5f)
            {
                arcParameter.setOccludedCars(false);
            }
            else if (occludedCarsValue >= 0.5f)
            {
                arcParameter.setOccludedCars(true);
            }
            */
        }
    }
}
