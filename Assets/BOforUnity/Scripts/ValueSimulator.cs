using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace BOforUnity
{
    public class ValueSimulator : MonoBehaviour
    {
        //contains all data that was created here
        private Dictionary<string, List<float>> simulatedData = new Dictionary<string, List<float>>();
        private Coroutine simulationCoroutine;

        private void Awake()
        {
            DontDestroyOnLoad(gameObject); // Ensure this persists across scenes
            StartSimulations(); // Start generating example data
        }

        // Start generating data for multiple simulated variables
        public void StartSimulations()
        {
            if (simulationCoroutine == null)
            {
                simulationCoroutine = StartCoroutine(SimulateValues());
                Debug.Log("Simulations started.");
            }
            else
            {
                Debug.Log("Simulations are already running.");
            }
        }

        // Simulate values for multiple variables (e.g., HeartRate, StepCount)
        private IEnumerator SimulateValues()
        {
            System.Random random = new System.Random();

            while (true)
            {
                // Simulate example variables
                AddSimulatedData("HeartRate", random.Next(60, 101)); // Simulate heart rate
                AddSimulatedData("StepCount", random.Next(0, 500));  // Simulate step count

                yield return new WaitForSeconds(1f); // Wait 1 second before generating more data
            }
        }

        // Add a simulated value for a specific variable
        private void AddSimulatedData(string key, float value)
        {
            if (!simulatedData.ContainsKey(key))
            {
                simulatedData[key] = new List<float>();
            }

            simulatedData[key].Add(value);
        }

        // Fetch all simulated data for a given variable
        public List<float> GetSimulatedData(string key)
        {
            if (simulatedData.TryGetValue(key, out var data))
            {
                return new List<float>(data); // Return a copy of the data
            }
            else
            {
                Debug.LogWarning($"No data found for key: {key}");
                return new List<float>();
            }
        }

public void ClearSimulatedDataForKey(string key)
{
    if (simulatedData.ContainsKey(key))
    {
        simulatedData[key].Clear(); // Clear the list for the specified key
        Debug.Log($"Cleared simulated data for key: {key}");
    }
    else
    {
        Debug.LogWarning($"Key '{key}' not found in simulatedData. Cannot clear.");
    }
}

        // Stop all simulations
        public void StopSimulations()
        {
            if (simulationCoroutine != null)
            {
                StopCoroutine(simulationCoroutine);
                simulationCoroutine = null;
                Debug.Log("Simulations stopped.");
            }
        }
    }
}
