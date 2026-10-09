using System;
using System.Collections.Generic;
using System.Globalization;

namespace BOforUnity.Scripts
{
    /// <summary>
    /// Configuration checks for settings that would otherwise crash or silently corrupt a run only after it has
    /// started (often after the participant finished the sampling phase). Shared by the inspector (HelpBox errors),
    /// the launcher and the manager's check before the init message, so all three report the same problems.
    /// </summary>
    public static class BoConfigValidator
    {
        /// <summary>
        /// Fixed columns of <c>ObservationsPerEvaluation.csv</c> in every backend (<c>IsBest</c> for single-objective,
        /// <c>IsPareto</c> for multi-objective runs; both are reserved because FinalDesignSelector reads either).
        /// Unity matches CSV columns case-insensitively, so keys are compared case-insensitively too.
        /// </summary>
        public static readonly IReadOnlyList<string> FixedObservationLogColumns = new[]
        {
            "UserID", "ConditionID", "GroupID", "Timestamp", "Iteration", "Phase", "IsBest", "IsPareto"
        };

        /// <summary>Observation-log column that contextual runs add (reserved only when contextual optimization is on).</summary>
        public const string ContextLogColumn = "Context";

        /// <summary>Returns one message per configuration error (empty when the configuration can be launched).</summary>
        public static List<string> GetErrors(BoForUnityManager manager)
        {
            var errors = new List<string>();
            if (manager == null)
            {
                errors.Add("BoForUnityManager is missing.");
                return errors;
            }

            var reserved = new HashSet<string>(FixedObservationLogColumns, StringComparer.OrdinalIgnoreCase);
            if (manager.contextualOptimization)
                reserved.Add(ContextLogColumn);

            var parameterKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var objectiveKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var duplicateParameterKeys = new List<string>();
            var duplicateObjectiveKeys = new List<string>();
            var reservedKeys = new List<string>();

            if (manager.parameters != null)
            {
                for (int i = 0; i < manager.parameters.Count; i++)
                {
                    var entry = manager.parameters[i];
                    if (entry == null || entry.value == null || string.IsNullOrWhiteSpace(entry.key))
                    {
                        errors.Add($"Parameter {(i + 1).ToString(CultureInfo.InvariantCulture)} has an empty key.");
                        continue;
                    }

                    string key = entry.key.Trim();
                    if (!parameterKeys.Add(key))
                    {
                        AddOnce(duplicateParameterKeys, key);
                        continue;
                    }

                    if (reserved.Contains(key))
                        AddOnce(reservedKeys, key);

                    float lo = entry.value.lowerBound;
                    float hi = entry.value.upperBound;
                    bool validBounds = BoObjectiveMath.IsFinite(lo) && BoObjectiveMath.IsFinite(hi) && lo < hi;
                    if (!validBounds)
                    {
                        errors.Add(
                            $"Parameter '{key}': Lower Bound ({Format(lo)}) must be below Upper Bound ({Format(hi)})."
                        );
                    }

                    if (manager.optimizerBackend == BoForUnityManager.OptimizerBackend.CABOP)
                        AddCabopParameterErrors(errors, key, entry.value, validBounds);
                }
            }

            if (manager.objectives != null)
            {
                for (int i = 0; i < manager.objectives.Count; i++)
                {
                    var entry = manager.objectives[i];
                    if (entry == null || entry.value == null || string.IsNullOrWhiteSpace(entry.key))
                    {
                        errors.Add($"Objective {(i + 1).ToString(CultureInfo.InvariantCulture)} has an empty key.");
                        continue;
                    }

                    string key = entry.key.Trim();
                    if (!objectiveKeys.Add(key))
                    {
                        AddOnce(duplicateObjectiveKeys, key);
                        continue;
                    }

                    if (reserved.Contains(key))
                        AddOnce(reservedKeys, key);

                    // Equal bounds are accepted: every backend maps a degenerate objective interval to 0. (Number Of
                    // Sub Measures is not checked here: it is read only when objectives are sent, may be set by code
                    // during the study, and SendObjectives falls back to 1 with a warning.)
                    float lo = entry.value.lowerBound;
                    float hi = entry.value.upperBound;
                    if (!BoObjectiveMath.IsFinite(lo) || !BoObjectiveMath.IsFinite(hi) || lo > hi)
                    {
                        errors.Add(
                            $"Objective '{key}': Lower Bound ({Format(lo)}) must not be above Upper Bound ({Format(hi)}). " +
                            "Use Smaller Is Better to flip the direction instead of swapping the bounds."
                        );
                    }
                }
            }

            if (duplicateParameterKeys.Count > 0)
                errors.Add("Duplicate parameter key(s): " + string.Join(", ", duplicateParameterKeys) + ".");
            if (duplicateObjectiveKeys.Count > 0)
                errors.Add("Duplicate objective key(s): " + string.Join(", ", duplicateObjectiveKeys) + ".");

            var overlap = new List<string>();
            foreach (string key in parameterKeys)
            {
                if (objectiveKeys.Contains(key))
                    overlap.Add(key);
            }
            if (overlap.Count > 0)
            {
                overlap.Sort(StringComparer.OrdinalIgnoreCase);
                errors.Add("Parameter and objective keys must be distinct: " + string.Join(", ", overlap) + ".");
            }

            if (reservedKeys.Count > 0)
            {
                errors.Add(
                    "Key(s) " + string.Join(", ", reservedKeys) + " collide with fixed columns of " +
                    "ObservationsPerEvaluation.csv (" + string.Join(", ", FixedObservationLogColumns) +
                    (manager.contextualOptimization ? ", " + ContextLogColumn : string.Empty) + "). Rename them."
                );
            }

            if (manager.optimizerBackend != BoForUnityManager.OptimizerBackend.CABOP)
            {
                if (manager.numRestarts < 1)
                {
                    errors.Add(
                        $"Optimizer Restarts must be at least 1 (got {manager.numRestarts.ToString(CultureInfo.InvariantCulture)})."
                    );
                }
                else if (manager.rawSamples < manager.numRestarts)
                {
                    errors.Add(
                        $"Raw Samples ({manager.rawSamples.ToString(CultureInfo.InvariantCulture)}) must be at least " +
                        $"Optimizer Restarts ({manager.numRestarts.ToString(CultureInfo.InvariantCulture)}): the restarts " +
                        "are chosen from the raw samples."
                    );
                }
            }

            if (manager.optimizerBackend == BoForUnityManager.OptimizerBackend.MetaTAF && manager.seed < 0)
            {
                errors.Add(
                    $"The MetaTAF backend needs a Seed of 0 or more (got {manager.seed.ToString(CultureInfo.InvariantCulture)}): " +
                    "openbo rejects negative seeds."
                );
            }

            return errors;
        }

        /// <summary>
        /// True when the configuration can be launched; otherwise <paramref name="error"/> lists every problem, one
        /// per line, ready to show on screen.
        /// </summary>
        public static bool TryValidate(BoForUnityManager manager, out string error)
        {
            var errors = GetErrors(manager);
            error = errors.Count == 0 ? null : string.Join("\n", errors);
            return errors.Count == 0;
        }

        // The CABOP backend refuses these at startup (cabop_runtime.py).
        private static void AddCabopParameterErrors(List<string> errors, string key, ParameterArgs value, bool validBounds)
        {
            float lo = value.lowerBound;
            float hi = value.upperBound;
            float tolerance = value.cabopTolerance;
            if (!BoObjectiveMath.IsFinite(tolerance) || tolerance < 0f || tolerance > 1f)
            {
                errors.Add(
                    $"Parameter '{key}': CABOP Tolerance ({Format(tolerance)}) must lie in [0, 1]: it is a fraction " +
                    "of the parameter's range (0.05 = 5% of Upper - Lower). A tolerance in parameter units is that " +
                    (validBounds
                        ? $"value divided by the range ({Format(hi - lo)})."
                        : "value divided by (Upper - Lower).")
                );
            }

            if (!validBounds || value.cabopPrefabricatedValues == null)
                return;

            var outside = new List<string>();
            foreach (float prefab in value.cabopPrefabricatedValues)
            {
                // Non-finite entries are dropped before sending; the backend allows 1e-9 of slack.
                if (BoObjectiveMath.IsFinite(prefab) && (prefab < lo - 1e-9 || prefab > hi + 1e-9))
                    outside.Add(Format(prefab));
            }
            if (outside.Count > 0)
            {
                errors.Add(
                    $"Parameter '{key}': CABOP Prefabricated Value(s) {string.Join(", ", outside)} lie outside its " +
                    $"bounds [{Format(lo)}, {Format(hi)}]; designs snapped to them would leave the configured range."
                );
            }
        }

        private static void AddOnce(List<string> list, string key)
        {
            foreach (string existing in list)
            {
                if (string.Equals(existing, key, StringComparison.OrdinalIgnoreCase))
                    return;
            }
            list.Add(key);
        }

        private static string Format(float value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }
    }
}
