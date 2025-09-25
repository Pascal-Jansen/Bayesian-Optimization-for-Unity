using System;
using System.Collections.Generic;
using System.Globalization;
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
        private List<Dictionary<string, object>> _csvData;

        private BoForUnityManager _bomanager;
        
        public void Start()
        {
            _bomanager = gameObject.GetComponent<BoForUnityManager>();
        }

        /// <summary>
        /// This method updates the values of the parameters in the optimization process using the data in the
        /// CSV file at the specified currentIndex. It loops through each parameter in the parameters dictionary
        /// and sets its Value property to the corresponding value in the CSV file at the given index using
        /// invariant culture parsing so number formats remain consistent across locales.
        /// </summary>
        /// <param name="currentIndex"></param>
        public void UpdateParameter(int currentIndex)
        {
            foreach (var pa in _bomanager.parameters)
            {
                pa.value.Value = float.Parse(_csvData[currentIndex][pa.key].ToString(), CultureInfo.InvariantCulture);
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
        public void AddParameter(string name, float lowerBound, float upperBound)
        {
            try
            {
                _bomanager.parameters.Add(new ParameterEntry(name, new ParameterArgs(lowerBound, upperBound)));
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
    }
}
