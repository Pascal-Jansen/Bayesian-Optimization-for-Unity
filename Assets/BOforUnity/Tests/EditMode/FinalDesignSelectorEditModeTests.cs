using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using BOforUnity;
using BOforUnity.Scripts;

namespace BOforUnity.Tests.EditMode
{
    public class FinalDesignSelectorEditModeTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(), "bo4unity_fds_" + Path.GetRandomFileName());
            Directory.CreateDirectory(Path.Combine(_tempRoot, "run"));
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_tempRoot))
                Directory.Delete(_tempRoot, recursive: true);
        }

        private static List<ParameterEntry> MakeParameters()
        {
            return new List<ParameterEntry>
            {
                new ParameterEntry("p0", new ParameterArgs(0f, 1f))
            };
        }

        private static List<ObjectiveEntry> MakeObjectives()
        {
            return new List<ObjectiveEntry>
            {
                new ObjectiveEntry("o0", new ObjectiveArgs(0f, 10f, false, 1)),
                new ObjectiveEntry("o1", new ObjectiveArgs(0f, 10f, false, 1))
            };
        }

        private void WriteObservationCsv(params string[] rows)
        {
            var lines = new List<string>
            {
                "UserID;ConditionID;GroupID;Timestamp;Iteration;Phase;IsPareto;o0;o1;p0"
            };
            lines.AddRange(rows);
            File.WriteAllLines(
                Path.Combine(_tempRoot, "run", "ObservationsPerEvaluation.csv"),
                lines
            );
        }

        [Test]
        public void SelectsBalancedParetoRowClosestToUtopia()
        {
            WriteObservationCsv(
                "u;c;g;t;1;sampling;TRUE;2;8;0.1",
                "u;c;g;t;2;sampling;TRUE;8;2;0.9",
                "u;c;g;t;3;optimization;TRUE;6;6;0.5"
            );

            bool ok = FinalDesignSelector.TrySelectFromLatestObservationCsv(
                logRootPath: _tempRoot,
                userId: "u",
                conditionId: "c",
                groupId: "g",
                parameters: MakeParameters(),
                objectives: MakeObjectives(),
                distanceEpsilon: 1e-6f,
                maximinEpsilon: 1e-6f,
                aggressionEpsilon: 1e-6f,
                selection: out FinalDesignSelector.SelectionResult selection,
                selectedCsvPath: out string csvPath,
                error: out string error
            );

            Assert.That(ok, Is.True, "Selection failed: " + error);
            Assert.That(csvPath, Does.EndWith("ObservationsPerEvaluation.csv"));
            Assert.That(selection.Iteration, Is.EqualTo(3), "The balanced trade-off row should win.");
            Assert.That(selection.ParameterRaw[0], Is.EqualTo(0.5f).Within(1e-5f));
        }

        [Test]
        public void ExcludesFinaldesignRowsAndForeignContexts()
        {
            WriteObservationCsv(
                "u;c;g;t;1;sampling;TRUE;2;8;0.1",
                "u;c;g;t;2;sampling;TRUE;8;2;0.9",
                "u;c;g;t;3;optimization;TRUE;6;6;0.5",
                // A dominating finaldesign row must never be re-selected.
                "u;c;g;t;4;finaldesign;TRUE;9;9;0.4",
                // A dominating row from another participant must be filtered.
                "other;c;g;t;5;optimization;TRUE;9;9;0.3"
            );

            bool ok = FinalDesignSelector.TrySelectFromLatestObservationCsv(
                logRootPath: _tempRoot,
                userId: "u",
                conditionId: "c",
                groupId: "g",
                parameters: MakeParameters(),
                objectives: MakeObjectives(),
                distanceEpsilon: 1e-6f,
                maximinEpsilon: 1e-6f,
                aggressionEpsilon: 1e-6f,
                selection: out FinalDesignSelector.SelectionResult selection,
                selectedCsvPath: out _,
                error: out string error
            );

            Assert.That(ok, Is.True, "Selection failed: " + error);
            Assert.That(selection.Iteration, Is.EqualTo(3));
        }

        [Test]
        public void FailsWithClearErrorWhenNoContextRowsExist()
        {
            WriteObservationCsv("other;c;g;t;1;sampling;TRUE;2;8;0.1");

            bool ok = FinalDesignSelector.TrySelectFromLatestObservationCsv(
                logRootPath: _tempRoot,
                userId: "u",
                conditionId: "c",
                groupId: "g",
                parameters: MakeParameters(),
                objectives: MakeObjectives(),
                distanceEpsilon: 1e-6f,
                maximinEpsilon: 1e-6f,
                aggressionEpsilon: 1e-6f,
                selection: out _,
                selectedCsvPath: out _,
                error: out string error
            );

            Assert.That(ok, Is.False);
            Assert.That(error, Is.Not.Null.And.Not.Empty);
        }
    }
}
