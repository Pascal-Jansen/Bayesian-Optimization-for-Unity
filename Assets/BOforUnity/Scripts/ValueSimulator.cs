using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace BOforUnity{

    public class ValueSimulator : MonoBehaviour
    {
        private Coroutine heartRateSimulationCoroutine;
        private BoForUnityManager _boForUnityManager;

        private void Awake()
        {
            _boForUnityManager = FindObjectOfType<BoForUnityManager>();
            if (_boForUnityManager == null)
            {
                Debug.LogError("BoForUnityManager not found in the scene.");
            }
        }

        public void StartHeartRateSimulation()
        {
            if (heartRateSimulationCoroutine == null)
            {
                heartRateSimulationCoroutine = StartCoroutine(SimulateHeartRate());
            }
        }

        public void StopHeartRateSimulation()
        {
            if (heartRateSimulationCoroutine != null)
            {
                StopCoroutine(heartRateSimulationCoroutine);
                heartRateSimulationCoroutine = null;
            }
        }

        private IEnumerator SimulateHeartRate()
        {
            if (_boForUnityManager == null)
            {
                yield break;
            }
            
            System.Random random = new System.Random();

            while (true)
            {
                float simulatedHeartRate = random.Next(60, 101);

                // Send the value to the BoForUnityManager
                _boForUnityManager.UpdateObjective("HeartRate", simulatedHeartRate);

                Debug.Log($"Simulated Heart Rate: {simulatedHeartRate}");

                // Wait for 1 second before generating the next value
                yield return new WaitForSeconds(1f);
            }
        }
    }

}

