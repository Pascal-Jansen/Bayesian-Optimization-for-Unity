using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
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
            /*
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
            */
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

        public void UpdateDesignParameters(GameObject caller = null) // the gameObjects call this method to update their parameters
        {
            //Debug.Log("Updating Parameters");

            foreach (var pa in _bomanager.parameters)
            {
                if (pa.value.gameObjectName != null && pa.value.scriptName != null && !string.IsNullOrEmpty(pa.value.variableName))
                {
                    MonoBehaviour script;
                    
                    if (caller != null)
                    {
                        // Get all MonoBehaviour components on the GameObject
                        MonoBehaviour[] scripts = caller.GetComponents<MonoBehaviour>();
                        // Find the script by its name
                        script = scripts.FirstOrDefault(behaviour => behaviour.GetType().Name == pa.value.scriptName);
                    }
                    else
                    {
                        // Find the GameObject by its name
                        GameObject obj = GameObject.Find(pa.value.gameObjectName);
                        if (obj == null)
                        {
                            Debug.LogWarning("GameObject not found: " + pa.value.gameObjectName);
                            continue;
                        }
                        // Get all MonoBehaviour components on the GameObject
                        MonoBehaviour[] scripts = obj.GetComponents<MonoBehaviour>();
                        // Find the script by its name
                        script = scripts.FirstOrDefault(behaviour => behaviour.GetType().Name == pa.value.scriptName);
                    }

                    // Apply discrete logic if applicable
                    float value = pa.value.Value;
                    
                    if (pa.value.isDiscrete)
                    {
                        int steps = Mathf.RoundToInt((pa.value.upperBound - pa.value.lowerBound) / pa.value.step);
                        if (steps <= 1)
                        {
                            bool boolValue = value >= (pa.value.lowerBound + pa.value.upperBound) / 2;
                            SetBoolFieldOrProperty(script, pa.value.variableName, boolValue);
                        }
                        else
                        {
                            float stepSize = (pa.value.upperBound - pa.value.lowerBound) / steps;
                            value = Mathf.Round(value / stepSize) * stepSize;
                            SetFieldOrProperty(script, pa.value.variableName, value);
                        }
                    }
                    else
                    {
                        SetFieldOrProperty(script, pa.value.variableName, value);
                    }
                }
            }
        }

        
        private void SetFieldOrProperty(MonoBehaviour script, string variableName, float value)
        {
            if (script == null)
            {
                Debug.LogWarning($"Script is null, cannot set {variableName}");
                return;
            }
            
            FieldInfo field = script.GetType().GetField(variableName, BindingFlags.Instance | BindingFlags.Public);
            PropertyInfo property = script.GetType().GetProperty(variableName, BindingFlags.Instance | BindingFlags.Public);

            if (field != null && field.FieldType == typeof(float))
            {
                field.SetValue(script, value);
            }
            else if (property != null && property.PropertyType == typeof(float))
            {
                property.SetValue(script, value, null);
            }
            else
            {
                Debug.LogWarning($"Variable {variableName} not found or not of type float in {script.name}");
            }
        }

        private void SetBoolFieldOrProperty(MonoBehaviour script, string variableName, bool value)
        {
            if (script == null)
            {
                Debug.LogWarning($"Script is null, cannot set {variableName}");
                return;
            }
            
            FieldInfo field = script.GetType().GetField(variableName, BindingFlags.Instance | BindingFlags.Public);
            PropertyInfo property = script.GetType().GetProperty(variableName, BindingFlags.Instance | BindingFlags.Public);

            if (field != null && field.FieldType == typeof(bool))
            {
                field.SetValue(script, value);
            }
            else if (property != null && property.PropertyType == typeof(bool))
            {
                property.SetValue(script, value, null);
            }
            else
            {
                Debug.LogWarning($"Variable {variableName} not found or not of type bool in {script.name}");
            }
        }
    }
}
