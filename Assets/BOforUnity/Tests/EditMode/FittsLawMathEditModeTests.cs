using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using BOforUnity.Examples;

namespace BOforUnity.Tests.EditMode
{
    public class FittsLawMathEditModeTests
    {
        [Test]
        public void IndexOfDifficultyIsShannonFormulation()
        {
            Assert.That(FittsLawMath.IndexOfDifficulty(7f, 1f), Is.EqualTo(3f).Within(1e-6f));
            Assert.That(FittsLawMath.IndexOfDifficulty(100f, 20f), Is.EqualTo((float)Math.Log(6.0, 2.0)).Within(1e-6f));
            Assert.That(float.IsNaN(FittsLawMath.IndexOfDifficulty(100f, 0f)), Is.True);
        }

        [Test]
        public void OddTargetCountsHaveAConstantAmplitudeAndVisitEveryTarget()
        {
            const int n = 13;
            const float diameter = 600f;
            var sequence = new List<int>();
            FittsLawMath.BuildAcrossCircleSequence(n, 0, 2 * n, sequence);

            float expected = diameter * Mathf.Cos(Mathf.PI / (2f * n)); // D · cos(90° / n) = 0.993 · D
            for (int i = 1; i < sequence.Count; i++)
            {
                float amplitude = Vector2.Distance(
                    FittsLawMath.RingPosition(sequence[i - 1], n, diameter),
                    FittsLawMath.RingPosition(sequence[i], n, diameter));
                Assert.That(amplitude, Is.EqualTo(expected).Within(1e-2f), $"Movement {i} has a different amplitude.");
            }

            var visited = new HashSet<int>(sequence.GetRange(0, n));
            Assert.That(visited.Count, Is.EqualTo(n), "The first n trials must visit every target once.");
            Assert.That(sequence[0], Is.EqualTo(0));
            Assert.That(sequence[1], Is.EqualTo(7), "Odd counts step by (n + 1) / 2.");
        }

        [Test]
        public void EvenTargetCountsKeepTheDiametricPairOrder()
        {
            var sequence = new List<int>();
            FittsLawMath.BuildAcrossCircleSequence(12, 0, 6, sequence);

            Assert.That(sequence, Is.EqualTo(new[] { 0, 6, 1, 7, 2, 8 }));
        }

        [Test]
        public void EndpointDeviationIsSignedAlongTheTaskAxis()
        {
            Vector2 from = new Vector2(0f, 0f);
            Vector2 to = new Vector2(100f, 0f);

            Assert.That(FittsLawMath.EndpointDeviationOnTaskAxis(from, to, new Vector2(105f, 3f)), Is.EqualTo(5f).Within(1e-5f));
            Assert.That(FittsLawMath.EndpointDeviationOnTaskAxis(from, to, new Vector2(95f, -2f)), Is.EqualTo(-5f).Within(1e-5f));
            Assert.That(FittsLawMath.EndpointDeviationOnTaskAxis(to, from, new Vector2(-5f, 0f)), Is.EqualTo(5f).Within(1e-5f),
                "Overshoot is positive in the movement direction.");
            Assert.That(float.IsNaN(FittsLawMath.EndpointDeviationOnTaskAxis(to, to, to)), Is.True);
        }

        [Test]
        public void BlockMetricsFollowIso9241Part9AndSkipTheFirstTrial()
        {
            Vector2 a = new Vector2(0f, 0f);
            Vector2 b = new Vector2(100f, 0f);
            var trials = new List<FittsTrialSample>
            {
                // First trial of the block: arbitrary start, absurd values that must not count.
                new FittsTrialSample { To = a, Endpoint = new Vector2(50f, 50f), MovementTimeMs = 5000f, HasFrom = false },
                new FittsTrialSample { From = a, To = b, Endpoint = new Vector2(102f, 1f), MovementTimeMs = 400f, HasFrom = true },
                new FittsTrialSample { From = b, To = a, Endpoint = new Vector2(2f, -1f), MovementTimeMs = 500f, HasFrom = true },   // dx = -2
                new FittsTrialSample { From = a, To = b, Endpoint = new Vector2(104f, 0f), MovementTimeMs = 600f, HasFrom = true },  // dx = +4
                new FittsTrialSample { From = b, To = a, Endpoint = new Vector2(4f, 0f), MovementTimeMs = 500f, HasFrom = true },    // dx = -4
            };

            FittsBlockMetrics metrics = FittsLawMath.ComputeBlockMetrics(trials, 20f);

            double sd = Math.Sqrt((4.0 + 4.0 + 16.0 + 16.0) / 3.0); // deviations +2, -2, +4, -4; mean 0
            double we = 4.133 * sd;
            double de = 100.0;
            double ide = Math.Log(de / we + 1.0, 2.0);

            Assert.That(metrics.TrialCount, Is.EqualTo(4));
            Assert.That(metrics.AmplitudePixels, Is.EqualTo(100f).Within(1e-4f));
            Assert.That(metrics.IndexOfDifficulty, Is.EqualTo((float)Math.Log(6.0, 2.0)).Within(1e-5f));
            Assert.That(metrics.EffectiveAmplitudePixels, Is.EqualTo((float)de).Within(1e-4f));
            Assert.That(metrics.EffectiveWidthPixels, Is.EqualTo((float)we).Within(1e-3f));
            Assert.That(metrics.EffectiveIndexOfDifficulty, Is.EqualTo((float)ide).Within(1e-4f));
            Assert.That(metrics.MovementTimeMs, Is.EqualTo(500f).Within(1e-3f));
            Assert.That(metrics.ThroughputBitsPerSecond, Is.EqualTo((float)(ide / 0.5)).Within(1e-3f));
        }

        [Test]
        public void EffectiveWidthNeedsAtLeastTwoTrials()
        {
            var trials = new List<FittsTrialSample>
            {
                new FittsTrialSample { HasFrom = false },
                new FittsTrialSample { From = Vector2.zero, To = new Vector2(80f, 0f), Endpoint = new Vector2(81f, 0f), MovementTimeMs = 450f, HasFrom = true }
            };

            FittsBlockMetrics metrics = FittsLawMath.ComputeBlockMetrics(trials, 40f);

            Assert.That(metrics.TrialCount, Is.EqualTo(1));
            Assert.That(metrics.AmplitudePixels, Is.EqualTo(80f).Within(1e-4f));
            Assert.That(float.IsNaN(metrics.EffectiveWidthPixels), Is.True);
            Assert.That(float.IsNaN(metrics.ThroughputBitsPerSecond), Is.True);
        }
    }
}
