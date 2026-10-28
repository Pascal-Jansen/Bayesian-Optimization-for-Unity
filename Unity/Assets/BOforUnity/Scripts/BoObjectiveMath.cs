using System;
using System.Collections.Generic;
using System.Globalization;

namespace BOforUnity.Scripts
{
    /// <summary>Result of reducing an objective's sub-measures to the one value sent to Python and logged.</summary>
    public struct ObjectiveAggregate
    {
        /// <summary>The value to send/log: mean of the finite sub-measures, clamped to the bounds (midpoint if none is finite).</summary>
        public float Value;
        /// <summary>The mean before clamping (NaN when no sub-measure was finite).</summary>
        public float UnclampedValue;
        /// <summary>Sub-measures inside the window.</summary>
        public int WindowCount;
        /// <summary>Finite sub-measures inside the window.</summary>
        public int FiniteCount;
        /// <summary>Older sub-measures outside the window (not part of the value).</summary>
        public int DroppedCount;
        /// <summary>True when no sub-measure was finite and the midpoint of the bounds was used.</summary>
        public bool UsedMidpointFallback;
        /// <summary>True when the mean was outside the bounds and was clamped.</summary>
        public bool Clamped;
    }

    /// <summary>
    /// The one place that turns objective sub-measures into a value and formats values for CSV logs, so the
    /// value Python receives, the final-design row and the example logs cannot drift apart.
    /// </summary>
    public static class BoObjectiveMath
    {
        public static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        /// <summary>
        /// Mean of the finite values among the last <paramref name="window"/> sub-measures, clamped to
        /// [min(bound), max(bound)]. Non-finite sub-measures (e.g. an unanswered optional item) are skipped;
        /// the midpoint of the bounds is used only when no sub-measure is finite.
        /// </summary>
        public static ObjectiveAggregate Aggregate(IReadOnlyList<float> values, int window, float lowerBound, float upperBound)
        {
            float lo = Math.Min(lowerBound, upperBound);
            float hi = Math.Max(lowerBound, upperBound);
            int total = values?.Count ?? 0;
            int effectiveWindow = Math.Max(1, window);
            int start = Math.Max(0, total - effectiveWindow);

            var result = new ObjectiveAggregate
            {
                WindowCount = total - start,
                DroppedCount = start,
                UnclampedValue = float.NaN
            };

            double sum = 0.0;
            for (int i = start; i < total; i++)
            {
                float v = values[i];
                if (!IsFinite(v))
                    continue;
                sum += v;
                result.FiniteCount++;
            }

            if (result.FiniteCount == 0)
            {
                result.Value = 0.5f * (lo + hi);
                result.UsedMidpointFallback = true;
                return result;
            }

            float mean = (float)(sum / result.FiniteCount);
            result.UnclampedValue = mean;
            if (mean < lo || mean > hi)
            {
                result.Value = Math.Min(Math.Max(mean, lo), hi);
                result.Clamped = true;
            }
            else
            {
                result.Value = mean;
            }

            return result;
        }

        /// <summary>
        /// Culture-invariant, round-trip formatting for CSV logs: the shortest text that parses back to the
        /// same float32 (e.g. <c>0.7</c>, <c>0.000274</c>), matching what Unity sends to Python. Empty for NaN/Inf.
        /// </summary>
        public static string FormatCsvFloat(float value)
        {
            return IsFinite(value) ? value.ToString("R", CultureInfo.InvariantCulture) : string.Empty;
        }
    }
}
