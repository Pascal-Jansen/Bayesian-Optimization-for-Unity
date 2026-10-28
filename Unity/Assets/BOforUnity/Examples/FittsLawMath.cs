using System;
using System.Collections.Generic;
using UnityEngine;

namespace BOforUnity.Examples
{
    /// <summary>One completed Fitts trial in play-area coordinates (pixels).</summary>
    public struct FittsTrialSample
    {
        /// <summary>Center of the previously selected target, i.e. where the movement started.</summary>
        public Vector2 From;
        /// <summary>Center of the target of this trial.</summary>
        public Vector2 To;
        /// <summary>Where the selection was registered.</summary>
        public Vector2 Endpoint;
        /// <summary>Time from the previous selection (target onset) to this selection.</summary>
        public float MovementTimeMs;
        /// <summary>
        /// False for the first trial of a block: it starts from an arbitrary cursor position, so it has no
        /// defined amplitude and is excluded from the ISO 9241-9 metrics.
        /// </summary>
        public bool HasFrom;
    }

    /// <summary>ISO 9241-9 (Soukoreff &amp; MacKenzie, 2004) summary of one block of trials. NaN when undefined.</summary>
    public struct FittsBlockMetrics
    {
        /// <summary>Trials the metrics are computed from (the first trial of the block is excluded).</summary>
        public int TrialCount;
        /// <summary>W: nominal target width (diameter).</summary>
        public float WidthPixels;
        /// <summary>D: mean center-to-center movement amplitude of the included trials.</summary>
        public float AmplitudePixels;
        /// <summary>ID = log2(D / W + 1), in bits.</summary>
        public float IndexOfDifficulty;
        /// <summary>De: mean effective amplitude (amplitude plus the endpoint deviation along the task axis).</summary>
        public float EffectiveAmplitudePixels;
        /// <summary>We = 4.133 · SD of the endpoint deviations projected on the task axis.</summary>
        public float EffectiveWidthPixels;
        /// <summary>IDe = log2(De / We + 1), in bits.</summary>
        public float EffectiveIndexOfDifficulty;
        /// <summary>MT: mean movement time of the included trials.</summary>
        public float MovementTimeMs;
        /// <summary>TP = IDe / MT, in bits per second.</summary>
        public float ThroughputBitsPerSecond;
    }

    /// <summary>
    /// Pure Fitts-law helpers used by <see cref="FittsLawTask"/>: the target order and the ISO 9241-9 metrics
    /// written to the app logs. These are analysis outputs only; they are not sent to the optimizer.
    /// </summary>
    public static class FittsLawMath
    {
        /// <summary>We = 4.133 · SD(dx) (the effective width for a 96% hit rate).</summary>
        public const float EffectiveWidthFactor = 4.133f;

        /// <summary>
        /// Target index of trial <paramref name="trialIndex"/> in the across-the-circle order. For an odd
        /// target count the sequence steps by (n + 1) / 2 (the ISO 9241-9 pattern), so every movement has the
        /// same amplitude, D · cos(90° / n), and all targets are visited. For an even count it alternates
        /// between diametrically opposite pairs (0, n/2, 1, n/2 + 1, ...), as in earlier versions: the movements
        /// alternate between D and D · cos(180° / n) (e.g. 0.966 · D for n = 12).
        /// </summary>
        public static int AcrossCircleTargetIndex(int targetCount, int startIndex, int trialIndex)
        {
            if (targetCount <= 0)
                return 0;

            int start = PositiveModulo(startIndex, targetCount);
            if (targetCount % 2 == 1)
            {
                long step = (targetCount + 1) / 2;
                return PositiveModulo((int)((start + step * trialIndex) % targetCount), targetCount);
            }

            int half = Math.Max(1, targetCount / 2);
            int pairIndex = (trialIndex / 2) % half;
            int sideOffset = trialIndex % 2 == 0 ? 0 : half;
            return PositiveModulo(start + pairIndex + sideOffset, targetCount);
        }

        public static void BuildAcrossCircleSequence(int targetCount, int startIndex, int trialCount, List<int> output)
        {
            if (output == null)
                throw new ArgumentNullException(nameof(output));

            output.Clear();
            for (int i = 0; i < trialCount; i++)
                output.Add(AcrossCircleTargetIndex(targetCount, startIndex, i));
        }

        /// <summary>Center of target <paramref name="index"/> on a ring of the given diameter (target 0 at angle 0).</summary>
        public static Vector2 RingPosition(int index, int targetCount, float ringDiameter, float startAngleDegrees = 0f)
        {
            float angle = (startAngleDegrees + 360f * index / Math.Max(1, targetCount)) * Mathf.Deg2Rad;
            return new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * (ringDiameter * 0.5f);
        }

        /// <summary>ID = log2(amplitude / width + 1); NaN for a non-positive width or a negative amplitude.</summary>
        public static float IndexOfDifficulty(float amplitude, float width)
        {
            if (!(width > 0f) || !(amplitude >= 0f) || float.IsInfinity(amplitude) || float.IsInfinity(width))
                return float.NaN;

            return (float)Math.Log(amplitude / (double)width + 1.0, 2.0);
        }

        /// <summary>
        /// Signed deviation of the endpoint from the target center along the task axis (from → to);
        /// positive = overshoot. NaN when from and to coincide.
        /// </summary>
        public static float EndpointDeviationOnTaskAxis(Vector2 from, Vector2 to, Vector2 endpoint)
        {
            Vector2 axis = to - from;
            float length = axis.magnitude;
            if (!(length > 1e-6f))
                return float.NaN;

            return Vector2.Dot(endpoint - to, axis / length);
        }

        public static FittsBlockMetrics ComputeBlockMetrics(IReadOnlyList<FittsTrialSample> trials, float widthPixels)
        {
            var result = new FittsBlockMetrics
            {
                WidthPixels = widthPixels,
                AmplitudePixels = float.NaN,
                IndexOfDifficulty = float.NaN,
                EffectiveAmplitudePixels = float.NaN,
                EffectiveWidthPixels = float.NaN,
                EffectiveIndexOfDifficulty = float.NaN,
                MovementTimeMs = float.NaN,
                ThroughputBitsPerSecond = float.NaN
            };
            if (trials == null)
                return result;

            var deviations = new List<double>(trials.Count);
            double amplitudeSum = 0.0;
            double effectiveAmplitudeSum = 0.0;
            double movementTimeSum = 0.0;
            for (int i = 0; i < trials.Count; i++)
            {
                FittsTrialSample trial = trials[i];
                if (!trial.HasFrom)
                    continue;

                float dx = EndpointDeviationOnTaskAxis(trial.From, trial.To, trial.Endpoint);
                if (float.IsNaN(dx) || float.IsNaN(trial.MovementTimeMs) || float.IsInfinity(trial.MovementTimeMs))
                    continue;

                float amplitude = Vector2.Distance(trial.From, trial.To);
                amplitudeSum += amplitude;
                effectiveAmplitudeSum += amplitude + dx;
                movementTimeSum += trial.MovementTimeMs;
                deviations.Add(dx);
            }

            int n = deviations.Count;
            result.TrialCount = n;
            if (n == 0)
                return result;

            result.AmplitudePixels = (float)(amplitudeSum / n);
            result.IndexOfDifficulty = IndexOfDifficulty(result.AmplitudePixels, widthPixels);
            result.EffectiveAmplitudePixels = (float)(effectiveAmplitudeSum / n);
            result.MovementTimeMs = (float)(movementTimeSum / n);

            if (n < 2)
                return result;

            double mean = 0.0;
            for (int i = 0; i < n; i++)
                mean += deviations[i];
            mean /= n;

            double squares = 0.0;
            for (int i = 0; i < n; i++)
                squares += (deviations[i] - mean) * (deviations[i] - mean);

            double sd = Math.Sqrt(squares / (n - 1));
            double effectiveWidth = EffectiveWidthFactor * sd;
            result.EffectiveWidthPixels = (float)effectiveWidth;
            if (!(effectiveWidth > 0.0))
                return result;

            result.EffectiveIndexOfDifficulty = IndexOfDifficulty(result.EffectiveAmplitudePixels, (float)effectiveWidth);
            if (result.MovementTimeMs > 0f && !float.IsNaN(result.EffectiveIndexOfDifficulty))
                result.ThroughputBitsPerSecond = result.EffectiveIndexOfDifficulty / (result.MovementTimeMs / 1000f);

            return result;
        }

        private static int PositiveModulo(int value, int modulo)
        {
            if (modulo <= 0)
                return 0;

            int result = value % modulo;
            return result < 0 ? result + modulo : result;
        }
    }
}
