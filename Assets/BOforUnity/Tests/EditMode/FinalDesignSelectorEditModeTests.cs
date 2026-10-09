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

        private void WriteCsv(string header, params string[] rows)
        {
            var lines = new List<string> { header };
            lines.AddRange(rows);
            File.WriteAllLines(Path.Combine(_tempRoot, "run", "ObservationsPerEvaluation.csv"), lines);
        }

        private bool Select(
            string userId,
            List<ParameterEntry> parameters,
            List<ObjectiveEntry> objectives,
            out FinalDesignSelector.SelectionResult selection,
            out string error,
            string conditionId = "c",
            string groupId = "g")
        {
            return FinalDesignSelector.TrySelectFromLatestObservationCsv(
                logRootPath: _tempRoot,
                userId: userId,
                conditionId: conditionId,
                groupId: groupId,
                parameters: parameters,
                objectives: objectives,
                distanceEpsilon: 1e-6f,
                maximinEpsilon: 1e-6f,
                aggressionEpsilon: 1e-6f,
                selection: out selection,
                selectedCsvPath: out _,
                error: out error
            );
        }

        [Test]
        public void SmallerIsBetterObjectivesAreFlipped()
        {
            // o0 = time (minimize), o1 = rating (maximize). Row 2 is best on both.
            WriteObservationCsv(
                "u;c;g;t;1;sampling;TRUE;900;6;0.1",
                "u;c;g;t;2;sampling;TRUE;300;8;0.9",
                "u;c;g;t;3;optimization;TRUE;600;7;0.5"
            );
            var objectives = new List<ObjectiveEntry>
            {
                new ObjectiveEntry("o0", new ObjectiveArgs(0f, 1000f, true, 1)),
                new ObjectiveEntry("o1", new ObjectiveArgs(0f, 10f, false, 1))
            };

            Assert.That(Select("u", MakeParameters(), objectives, out var selection, out string error), Is.True, error);
            Assert.That(selection.Iteration, Is.EqualTo(2));
        }

        [Test]
        public void SingleObjectiveIsBestFileUsesTheFlaggedRow()
        {
            WriteCsv(
                "UserID;ConditionID;GroupID;Timestamp;Iteration;Phase;IsBest;o0;p0",
                "u;c;g;t;1;sampling;FALSE;3;0.2",
                "u;c;g;t;2;optimization;TRUE;8;0.6",
                "u;c;g;t;3;optimization;FALSE;5;0.4"
            );
            var objectives = new List<ObjectiveEntry> { new ObjectiveEntry("o0", new ObjectiveArgs(0f, 10f, false, 1)) };

            Assert.That(Select("u", MakeParameters(), objectives, out var selection, out string error), Is.True, error);
            Assert.That(selection.Iteration, Is.EqualTo(2));
            Assert.That(selection.ParameterRaw[0], Is.EqualTo(0.6f).Within(1e-6f));
            Assert.That(selection.CandidateSource, Is.EqualTo("IsBest"));
        }

        [Test]
        public void MatchesQuotedAndZeroPaddedIdsExactly()
        {
            WriteObservationCsv(
                "\"007\";\"HITL;MOBO\";g;t;1;sampling;TRUE;2;8;0.1",
                "7;HITL;g;t;2;sampling;TRUE;9;9;0.9",
                "007;\"HITL;MOBO\";g;t;3;optimization;TRUE;6;6;0.5"
            );

            Assert.That(
                Select("007", MakeParameters(), MakeObjectives(), out var selection, out string error, conditionId: "HITL;MOBO"),
                Is.True, error);
            Assert.That(selection.Iteration, Is.EqualTo(3), "'7' is a different participant than '007'.");
        }

        [Test]
        public void FallsBackToTheParticipantsNonDominatedRowsWhenNoRowIsFlagged()
        {
            // Non-contextual warm start with a warm-start incumbent: every participant row is FALSE.
            WriteObservationCsv(
                "u;c;g;t;5;sampling;FALSE;2;8;0.1",
                "u;c;g;t;6;sampling;FALSE;8;2;0.9",
                "u;c;g;t;7;optimization;FALSE;6;6;0.5",
                "u;c;g;t;8;optimization;FALSE;1;1;0.3"
            );

            Assert.That(Select("u", MakeParameters(), MakeObjectives(), out var selection, out string error), Is.True, error);
            Assert.That(selection.Iteration, Is.EqualTo(7));
            Assert.That(selection.CandidateSource, Is.EqualTo("computed"));
        }

        [Test]
        public void FallbackSkipsNonFiniteRowsAndKeepsOnlyTheFirstCopyOfDuplicates()
        {
            WriteObservationCsv(
                "u;c;g;t;1;sampling;FALSE;5;5;0.9",
                // Same objective vector as iteration 1: IsPareto flags only the first copy, so must the fallback
                // (even though this row would win the aggression tie-break).
                "u;c;g;t;2;optimization;FALSE;5;5;0.5",
                // Unparsable / non-finite objectives: skipped although they would dominate.
                "u;c;g;t;3;optimization;FALSE;;9;0.5",
                "u;c;g;t;4;optimization;FALSE;NaN;9;0.5",
                "u;c;g;t;5;sampling;FALSE;1;1;0.5"
            );

            Assert.That(Select("u", MakeParameters(), MakeObjectives(), out var selection, out string error), Is.True, error);
            Assert.That(selection.CandidateSource, Is.EqualTo("computed"));
            Assert.That(selection.Iteration, Is.EqualTo(1));
        }

        [Test]
        public void SingleObjectiveFallbackUsesDirectionAndClampsToBounds()
        {
            // Minimize; 40000 and 50000 both clamp to the upper bound 30000 and are worse than 12000.
            WriteCsv(
                "UserID;ConditionID;GroupID;Timestamp;Iteration;Phase;IsBest;o0;p0",
                "u;c;g;t;1;sampling;FALSE;40000;0.2",
                "u;c;g;t;2;optimization;FALSE;12000;0.6",
                "u;c;g;t;3;optimization;FALSE;50000;0.4"
            );
            var objectives = new List<ObjectiveEntry> { new ObjectiveEntry("o0", new ObjectiveArgs(0f, 30000f, true, 1)) };

            Assert.That(Select("u", MakeParameters(), objectives, out var selection, out string error), Is.True, error);
            Assert.That(selection.Iteration, Is.EqualTo(2));
            Assert.That(selection.CandidateSource, Is.EqualTo("computed"));
        }

        [Test]
        public void AggressionTieBreakNormalizesByConfiguredBounds()
        {
            // Rows 1 and 2 tie on every objective. Row 2 is closer to the middle of the configured space
            // (p0 0.52 of [0,1], p1 50 of [0,100]); normalizing by the observed spread (p0 0.50..0.52) used to
            // make the tiny p0 difference dominate and pick row 1.
            WriteCsv(
                "UserID;ConditionID;GroupID;Timestamp;Iteration;Phase;IsPareto;o0;o1;p0;p1",
                "u;c;g;t;1;optimization;TRUE;5;5;0.50;80",
                "u;c;g;t;2;optimization;TRUE;5;5;0.52;50",
                "u;c;g;t;3;sampling;FALSE;1;1;0.51;100"
            );
            var parameters = new List<ParameterEntry>
            {
                new ParameterEntry("p0", new ParameterArgs(0f, 1f)),
                new ParameterEntry("p1", new ParameterArgs(0f, 100f))
            };

            Assert.That(Select("u", parameters, MakeObjectives(), out var selection, out string error), Is.True, error);
            Assert.That(selection.Iteration, Is.EqualTo(2));
            Assert.That(selection.Aggression, Is.EqualTo(0.02f).Within(1e-4f));
        }

        [Test]
        public void TrySelectFromLogRootsSkipsMissingRootsAndReportsTheRootUsed()
        {
            WriteObservationCsv("u;c;g;t;1;sampling;TRUE;2;8;0.1");
            string missing = Path.Combine(_tempRoot, "does-not-exist");

            bool ok = FinalDesignSelector.TrySelectFromLogRoots(
                primaryLogRoot: missing,
                fallbackLogRoots: new[] { _tempRoot + Path.DirectorySeparatorChar, _tempRoot },
                userId: "u",
                conditionId: "c",
                groupId: "g",
                parameters: MakeParameters(),
                objectives: MakeObjectives(),
                distanceEpsilon: 1e-6f,
                maximinEpsilon: 1e-6f,
                aggressionEpsilon: 1e-6f,
                selection: out var selection,
                selectedCsvPath: out _,
                selectedLogRoot: out string usedRoot,
                error: out string error
            );

            Assert.That(ok, Is.True, error);
            Assert.That(selection.Iteration, Is.EqualTo(1));
            Assert.That(usedRoot, Is.EqualTo(Path.GetFullPath(_tempRoot).TrimEnd(Path.DirectorySeparatorChar)));
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
