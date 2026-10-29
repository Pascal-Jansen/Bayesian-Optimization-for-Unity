using System.Collections.Generic;
using BayesianOptimization.Scripts;
using UnityEngine;

namespace BayesianOptimization.Examples
{
    /// <summary>Evaluates a bounded quadratic objective without questionnaires or scene reloads.</summary>
    [DefaultExecutionOrder(-100)]
    public sealed class AnalyticOptimizationExample : MonoBehaviour
    {
        [Range(1, 65535)] public int port = 56001;
        [Min(1)] public int samplingIterations = 5;
        [Min(0)] public int optimizationIterations = 10;
        public int completedEvaluations;
        public float bestCost = float.PositiveInfinity;
        private BoForUnityManager _manager;
        private int _lastEvaluatedIteration = -1;

        private void Awake()
        {
            _manager = GetComponent<BoForUnityManager>();
            if (_manager == null)
            {
                Debug.LogError("Analytic example requires the manager components created by the demo setup command.");
                enabled = false;
                return;
            }
            _manager.port = port;
            _manager.parameters = new List<ParameterEntry>
            {
                new ParameterEntry("x", new ParameterArgs(0f, 1f)),
                new ParameterEntry("y", new ParameterArgs(0f, 1f))
            };
            _manager.objectives = new List<ObjectiveEntry>
            {
                new ObjectiveEntry("cost", new ObjectiveArgs(0f, 2f, true, 1))
            };
            _manager.optimizerBackend = BoForUnityManager.OptimizerBackend.BoTorch;
            _manager.iterationAdvanceMode = BoForUnityManager.IterationAdvanceMode.Automatic;
            _manager.reloadSceneOnIterationAdvance = false;
            _manager.automaticAdvanceDelaySec = 0f;
            _manager.enableFinalDesignRound = false;
            _manager.perfectRatingActive = false;
            _manager.SetIterationBudget(samplingIterations, optimizationIterations);
            _manager.rawSamples = 64;
            _manager.numRestarts = 4;
            _manager.mcSamples = 64;
            _manager.userId = "analytic-demo";
            _manager.conditionId = "quadratic";
        }

        private void Update()
        {
            if (_manager == null || !_manager.initialized || !_manager.simulationRunning ||
                _manager.hasNewDesignParameterValues || _manager.optimizationRunning || _manager.optimizationFinished ||
                _manager.LastSuggestionLogIteration < 0 || _lastEvaluatedIteration == _manager.LastSuggestionLogIteration)
                return;

            _lastEvaluatedIteration = _manager.LastSuggestionLogIteration;
            float cost = Evaluate(_manager.parameters[0].value.Value, _manager.parameters[1].value.Value);
            _manager.objectives[0].value.values.Add(cost);
            completedEvaluations++;
            bestCost = Mathf.Min(bestCost, cost);
            _manager.OptimizationStart();
        }

        public static float Evaluate(float x, float y)
        {
            float dx = x - 0.25f;
            float dy = y - 0.75f;
            return dx * dx + dy * dy;
        }

        private void OnGUI()
        {
            GUILayout.BeginArea(new Rect(24, 24, 620, 180), GUI.skin.box);
            GUILayout.Label("Analytic Bayesian optimization");
            GUILayout.Label("cost = (x - 0.25)^2 + (y - 0.75)^2");
            GUILayout.Label($"Completed evaluations: {completedEvaluations}    Best cost: {bestCost:G6}");
            GUILayout.Label(_manager != null && _manager.optimizationFinished ? "Optimization completed." : "Python setup and optimization progress are recorded in the Console and CSV logs.");
            GUILayout.EndArea();
        }
    }
}
