using System;
using System.Collections.Generic;
using UnityEngine;

// The Optimizer class manages optimization settings and parameters for the application.
// It provides methods to start and control the optimization process, add and retrieve parameters,
// and perform various optimization-related tasks. This class serves as the core component
// for managing optimization behavior.
namespace BOforUnity.Scripts
{
    public class Optimizer : MonoBehaviour
    {
        private BoForUnityManager _bomanager;
        // Unknown keys are reported once each: a typo would otherwise give a constant design all study long.
        private readonly HashSet<string> _warnedUnknownKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private bool _warnedMissingManager;

        public void Start()
        {
            _bomanager = ResolveManager();
        }

        // Looked up on first use, so calls made before Start (e.g. from another script's Awake or Start) work.
        private BoForUnityManager Manager
        {
            get
            {
                if (_bomanager == null)
                    _bomanager = ResolveManager();
                if (_bomanager == null && !_warnedMissingManager)
                {
                    _warnedMissingManager = true;
                    Debug.LogWarning("Optimizer: no BoForUnityManager found; parameter and objective calls are ignored.");
                }
                return _bomanager;
            }
        }

        private BoForUnityManager ResolveManager()
        {
            // The running (persistent) manager wins over this GameObject's copy: in a reloaded scene that copy is a
            // duplicate that is being discarded.
            BoForUnityManager running = BoForUnityManager.Instance;
            return running != null ? running : GetComponent<BoForUnityManager>();
        }

        private void WarnUnknownKey(string kind, string name)
        {
            string key = (name ?? string.Empty).Trim();
            if (!_warnedUnknownKeys.Add(kind + "\n" + key))
                return;

            Debug.LogWarning(
                kind == "objective value"
                    ? $"Optimizer: no objective key matches '{key}'; the value is ignored. This warning is shown once per name."
                    : $"Optimizer: no {kind} with key '{key}' is configured in BoForUnityManager. Check the spelling " +
                      "(keys are matched case-insensitively). This warning is shown once per key."
            );
        }

        private ParameterEntry FindParameter(string name)
        {
            var manager = Manager;
            if (manager == null || manager.parameters == null)
                return null;

            string targetName = (name ?? string.Empty).Trim();
            foreach (var pa in manager.parameters)
            {
                if (pa == null || pa.value == null)
                    continue;

                if (string.Equals((pa.key ?? string.Empty).Trim(), targetName, StringComparison.OrdinalIgnoreCase))
                    return pa;
            }

            WarnUnknownKey("parameter", name);
            return null;
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
            var manager = Manager;
            if (manager == null || manager.parameters == null)
            {
                return;
            }

            manager.parameters.Add(new ParameterEntry(name, new ParameterArgs(lowerBound, upperBound)));
        }

        /// <summary>
        /// Returns the current value of the parameter with the given key (case-insensitive). An unknown key logs a
        /// warning once and returns 0.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public float GetParameterValue(string name)
        {
            var entry = FindParameter(name);
            return entry != null ? entry.value.Value : 0.0f;
        }


        /// <summary>
        /// Returns the ParameterArgs of the parameter with the given key (case-insensitive). An unknown key logs a
        /// warning once and returns a new ParameterArgs that is not part of the configuration.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public ParameterArgs GetParameter(string name)
        {
            var entry = FindParameter(name);
            return entry != null ? entry.value : new ParameterArgs();
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
            var manager = Manager;
            if (manager == null || manager.objectives == null)
            {
                return;
            }

            string targetName = (name ?? string.Empty).Trim();
            if (string.IsNullOrWhiteSpace(targetName))
            {
                return;
            }
            if (args == null)
            {
                args = new ObjectiveArgs();
            }

            foreach (var ob in manager.objectives)
            {
                if (ob == null)
                    continue;

                if (string.Equals((ob.key ?? string.Empty).Trim(), targetName, StringComparison.OrdinalIgnoreCase))
                {
                    // if found in the list ... update the value
                    ob.value = args;
                    return;
                }
            }
            // if not found in the list ... add as new entry
            manager.objectives.Add(new ObjectiveEntry(targetName, args));
        }

        public void AddObjectiveValue(string name, float currVal)
        {
            var manager = Manager;
            if (string.IsNullOrWhiteSpace(name) || manager == null || manager.objectives == null)
            {
                return;
            }

            string targetName = name.Trim();

            ObjectiveEntry bestMatch = null;
            var bestMatchLength = -1;
            foreach (var ob in manager.objectives)
            {
                if (ob == null || ob.value == null || string.IsNullOrWhiteSpace(ob.key))
                {
                    continue;
                }

                string objectiveKey = ob.key.Trim();
                if (ContainsObjectiveKeyMatch(targetName, objectiveKey) && objectiveKey.Length > bestMatchLength)
                {
                    bestMatch = ob;
                    bestMatchLength = objectiveKey.Length;
                }
            }

            if (bestMatch == null)
            {
                WarnUnknownKey("objective value", name);
                return;
            }

            // If multiple objective keys are substrings of the same header, use the most specific (longest) key.
            if (bestMatch.value.values == null)
            {
                bestMatch.value.values = new List<float>();
            }
            bestMatch.value.values.Add(currVal);
        }

        public bool HasObjectiveMatch(string name)
        {
            var manager = Manager;
            if (string.IsNullOrWhiteSpace(name) || manager == null || manager.objectives == null)
            {
                return false;
            }

            string targetName = name.Trim();
            foreach (var ob in manager.objectives)
            {
                if (ob == null || ob.value == null || string.IsNullOrWhiteSpace(ob.key))
                {
                    continue;
                }

                if (ContainsObjectiveKeyMatch(targetName, ob.key.Trim()))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool ContainsObjectiveKeyMatch(string source, string objectiveKey)
        {
            if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(objectiveKey))
                return false;

            int start = 0;
            while (start < source.Length)
            {
                int idx = source.IndexOf(objectiveKey, start, StringComparison.OrdinalIgnoreCase);
                if (idx < 0)
                    return false;

                int end = idx + objectiveKey.Length;
                if (IsObjectiveBoundary(source, idx) && IsObjectiveBoundary(source, end))
                    return true;

                start = idx + 1;
            }

            return false;
        }

        private static bool IsObjectiveBoundary(string text, int boundaryIndex)
        {
            if (boundaryIndex <= 0 || boundaryIndex >= text.Length)
                return true;

            char left = text[boundaryIndex - 1];
            char right = text[boundaryIndex];
            if (!char.IsLetterOrDigit(left) || !char.IsLetterOrDigit(right))
                return true;

            if (char.IsLetter(left) && char.IsLetter(right) && char.IsLower(left) && char.IsUpper(right))
                return true;

            if (char.IsLetter(left) && char.IsDigit(right))
                return true;
            if (char.IsDigit(left) && char.IsLetter(right))
                return true;

            return false;
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
            var manager = Manager;
            if (manager == null || manager.objectives == null)
            {
                return;
            }

            string targetName = (name ?? string.Empty).Trim();
            if (string.IsNullOrWhiteSpace(targetName))
            {
                return;
            }

            foreach (var ob in manager.objectives)
            {
                if (ob == null || ob.value == null)
                    continue;

                if (string.Equals((ob.key ?? string.Empty).Trim(), targetName, StringComparison.OrdinalIgnoreCase))
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
            manager.objectives.Add(new ObjectiveEntry(targetName, new ObjectiveArgs(lowerBound, upperBound, smallerIsBetter,numberOfSubMeasures)));
        }


        /// <summary>
        /// Returns the ObjectiveArgs of the objective with the given key (case-insensitive). An unknown key logs a
        /// warning once and returns a new ObjectiveArgs that is not part of the configuration.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public ObjectiveArgs GetObjective(string name)
        {
            var manager = Manager;
            if (manager == null || manager.objectives == null)
                return new ObjectiveArgs();

            string targetName = (name ?? string.Empty).Trim();

            foreach (var ob in manager.objectives)
            {
                if (ob == null || ob.value == null)
                    continue;

                if (string.Equals((ob.key ?? string.Empty).Trim(), targetName, StringComparison.OrdinalIgnoreCase))
                    return ob.value;
            }

            WarnUnknownKey("objective", name);
            return new ObjectiveArgs();
        }
    }
}
