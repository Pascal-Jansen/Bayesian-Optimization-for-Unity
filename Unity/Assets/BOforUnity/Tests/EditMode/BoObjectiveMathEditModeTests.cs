using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using NUnit.Framework;
using BOforUnity.Scripts;

namespace BOforUnity.Tests.EditMode
{
    public class BoObjectiveMathEditModeTests
    {
        [Test]
        public void SkipsNonFiniteSubMeasures()
        {
            var values = new List<float> { 2f, float.NaN, 4f, float.PositiveInfinity };

            ObjectiveAggregate result = BoObjectiveMath.Aggregate(values, 4, 0f, 10f);

            Assert.That(result.Value, Is.EqualTo(3f).Within(1e-6f), "Mean of the finite items only.");
            Assert.That(result.FiniteCount, Is.EqualTo(2));
            Assert.That(result.WindowCount, Is.EqualTo(4));
            Assert.That(result.UsedMidpointFallback, Is.False);
        }

        [Test]
        public void FallsBackToMidpointWhenNoSubMeasureIsFinite()
        {
            ObjectiveAggregate result = BoObjectiveMath.Aggregate(new List<float> { float.NaN, float.NaN }, 2, 0f, 10f);

            Assert.That(result.Value, Is.EqualTo(5f));
            Assert.That(result.UsedMidpointFallback, Is.True);
            Assert.That(float.IsNaN(result.UnclampedValue), Is.True);

            ObjectiveAggregate empty = BoObjectiveMath.Aggregate(null, 1, 2f, 4f);
            Assert.That(empty.Value, Is.EqualTo(3f));
            Assert.That(empty.UsedMidpointFallback, Is.True);
        }

        [Test]
        public void ClampsToBoundsInEitherOrder()
        {
            ObjectiveAggregate above = BoObjectiveMath.Aggregate(new List<float> { 12f }, 1, 0f, 10f);
            Assert.That(above.Value, Is.EqualTo(10f));
            Assert.That(above.UnclampedValue, Is.EqualTo(12f));
            Assert.That(above.Clamped, Is.True);

            ObjectiveAggregate reversed = BoObjectiveMath.Aggregate(new List<float> { -3f }, 1, 10f, 0f);
            Assert.That(reversed.Value, Is.EqualTo(0f));
            Assert.That(reversed.Clamped, Is.True);

            ObjectiveAggregate inside = BoObjectiveMath.Aggregate(new List<float> { 7f }, 1, 0f, 10f);
            Assert.That(inside.Clamped, Is.False);
        }

        [Test]
        public void UsesOnlyTheLastWindowAndReportsDroppedValues()
        {
            var values = new List<float> { 1f, 2f, 3f, 4f };

            ObjectiveAggregate result = BoObjectiveMath.Aggregate(values, 2, 0f, 10f);

            Assert.That(result.Value, Is.EqualTo(3.5f).Within(1e-6f));
            Assert.That(result.WindowCount, Is.EqualTo(2));
            Assert.That(result.DroppedCount, Is.EqualTo(2));

            ObjectiveAggregate zeroWindow = BoObjectiveMath.Aggregate(values, 0, 0f, 10f);
            Assert.That(zeroWindow.Value, Is.EqualTo(4f), "A window below 1 is treated as 1.");
            Assert.That(zeroWindow.DroppedCount, Is.EqualTo(3));
        }

        [Test]
        public void FormatsCsvFloatsRoundTripAndCultureInvariant()
        {
            CultureInfo previous = Thread.CurrentThread.CurrentCulture;
            try
            {
                Thread.CurrentThread.CurrentCulture = new CultureInfo("de-DE");

                Assert.That(BoObjectiveMath.FormatCsvFloat(0.7f), Is.EqualTo("0.7"));
                Assert.That(BoObjectiveMath.FormatCsvFloat(0.000274f), Is.EqualTo("0.000274"));
                Assert.That(BoObjectiveMath.FormatCsvFloat(1234.5f), Is.EqualTo("1234.5"));
                Assert.That(BoObjectiveMath.FormatCsvFloat(float.NaN), Is.EqualTo(string.Empty));
                Assert.That(BoObjectiveMath.FormatCsvFloat(float.NegativeInfinity), Is.EqualTo(string.Empty));

                float value = 0.1f + 0.2f;
                string text = BoObjectiveMath.FormatCsvFloat(value);
                Assert.That(float.Parse(text, CultureInfo.InvariantCulture), Is.EqualTo(value), "Must round-trip.");
            }
            finally
            {
                Thread.CurrentThread.CurrentCulture = previous;
            }
        }
    }
}
