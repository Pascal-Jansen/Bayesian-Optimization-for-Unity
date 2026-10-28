using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using BOforUnity;
using QuestionnaireToolkit.Scripts;

namespace BOforUnity.Scripts
{
    /// <summary>
    /// Deterministic final-design selection from optimizer observation CSV.
    /// </summary>
    public static class FinalDesignSelector
    {
        public sealed class SelectionResult
        {
            public int Iteration;
            public float[] ParameterRaw;
            public float UtopiaDistance;
            public float Maximin;
            public float Aggression;
            /// <summary>
            /// Where the candidate pool came from: "IsPareto"/"IsBest" (the backend's flags), "computed"
            /// (no row was flagged, so the non-dominated/best rows were computed from the participant's own
            /// rows), or "all" (the log has no flag column).
            /// </summary>
            public string CandidateSource;
        }

        private sealed class CsvRow
        {
            public int Iteration;
            public float[] ObjectiveRaw;
            public float[] ObjectiveNormalized;
            public float[] ParameterRaw;
            public bool IsCandidate;
            public float UtopiaDistance;
            public float Maximin;
            public float Aggression;
        }

        public static bool TrySelectFromLatestObservationCsv(
            string logRootPath,
            string userId,
            string conditionId,
            string groupId,
            IList<ParameterEntry> parameters,
            IList<ObjectiveEntry> objectives,
            float distanceEpsilon,
            float maximinEpsilon,
            float aggressionEpsilon,
            out SelectionResult selection,
            out string selectedCsvPath,
            out string error
        )
        {
            selection = null;
            selectedCsvPath = null;
            error = null;

            var effectiveParameters = BuildEffectiveParameterEntries(parameters);
            var effectiveObjectives = BuildEffectiveObjectiveEntries(objectives);

            if (effectiveParameters.Count == 0)
            {
                error = "No parameters are defined.";
                return false;
            }
            if (effectiveObjectives.Count == 0)
            {
                error = "No objectives are defined.";
                return false;
            }
            if (string.IsNullOrWhiteSpace(logRootPath))
            {
                error = "Log root path is empty.";
                return false;
            }

            if (!TryGetLatestObservationCsvPath(logRootPath, userId, conditionId, groupId, out selectedCsvPath, out string[] lines, out error))
            {
                return false;
            }

            if (!TrySelectFinalDesign(
                    selectedCsvPath,
                    lines,
                    NormalizeContextToken(userId),
                    NormalizeContextToken(conditionId),
                    NormalizeContextToken(groupId),
                    effectiveParameters,
                    effectiveObjectives,
                    distanceEpsilon,
                    maximinEpsilon,
                    aggressionEpsilon,
                    out selection,
                    out error))
            {
                return false;
            }

            return true;
        }

        /// <summary>
        /// Tries <paramref name="primaryLogRoot"/> (<see cref="LogDataFolderUtility.LogDataRoot"/> when null — resolve
        /// it on the main thread) first, then each fallback root in order, and stops at the first root that yields a
        /// selection. Roots are compared in normalized form, so duplicates and missing folders are skipped without
        /// scanning. The search itself touches only the file system, so it may run off the main thread when the
        /// primary root is passed explicitly.
        /// </summary>
        public static bool TrySelectFromLogRoots(
            string primaryLogRoot,
            IEnumerable<string> fallbackLogRoots,
            string userId,
            string conditionId,
            string groupId,
            IList<ParameterEntry> parameters,
            IList<ObjectiveEntry> objectives,
            float distanceEpsilon,
            float maximinEpsilon,
            float aggressionEpsilon,
            out SelectionResult selection,
            out string selectedCsvPath,
            out string selectedLogRoot,
            out string error
        )
        {
            selection = null;
            selectedCsvPath = null;
            selectedLogRoot = null;
            error = null;

            var roots = new List<string>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            void AddRoot(string root)
            {
                if (string.IsNullOrWhiteSpace(root))
                    return;

                string normalized;
                try
                {
                    normalized = LogDataFolderUtility.NormalizeRoot(root);
                }
                catch (Exception)
                {
                    return;
                }

                if (seen.Add(normalized))
                    roots.Add(normalized);
            }

            AddRoot(primaryLogRoot ?? LogDataFolderUtility.LogDataRoot);
            if (fallbackLogRoots != null)
            {
                foreach (string root in fallbackLogRoots)
                    AddRoot(root);
            }

            var errors = new List<string>();
            foreach (string root in roots)
            {
                if (!Directory.Exists(root))
                    continue;

                if (TrySelectFromLatestObservationCsv(
                        root, userId, conditionId, groupId, parameters, objectives,
                        distanceEpsilon, maximinEpsilon, aggressionEpsilon,
                        out selection, out selectedCsvPath, out string rootError))
                {
                    selectedLogRoot = root;
                    return true;
                }

                errors.Add(root + ": " + rootError);
            }

            error = errors.Count > 0
                ? "No eligible observation log was found. " + string.Join(" | ", errors)
                : "None of the log roots exists: " + string.Join(", ", roots);
            return false;
        }

        private static List<ParameterEntry> BuildEffectiveParameterEntries(IList<ParameterEntry> parameters)
        {
            var result = new List<ParameterEntry>();
            if (parameters == null)
                return result;

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < parameters.Count; i++)
            {
                var parameter = parameters[i];
                if (parameter == null || parameter.value == null || string.IsNullOrWhiteSpace(parameter.key))
                    continue;

                string key = parameter.key.Trim();
                if (!seen.Add(key))
                    continue;

                result.Add(new ParameterEntry(key, parameter.value));
            }

            return result;
        }

        private static List<ObjectiveEntry> BuildEffectiveObjectiveEntries(IList<ObjectiveEntry> objectives)
        {
            var result = new List<ObjectiveEntry>();
            if (objectives == null)
                return result;

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < objectives.Count; i++)
            {
                var objective = objectives[i];
                if (objective == null || objective.value == null || string.IsNullOrWhiteSpace(objective.key))
                    continue;

                string key = objective.key.Trim();
                if (!seen.Add(key))
                    continue;

                result.Add(new ObjectiveEntry(key, objective.value));
            }

            return result;
        }

        private static bool TryGetLatestObservationCsvPath(
            string logRootPath,
            string userId,
            string conditionId,
            string groupId,
            out string csvPath,
            out string[] csvLines,
            out string error
        )
        {
            csvPath = null;
            csvLines = null;
            error = null;

            if (!Directory.Exists(logRootPath))
            {
                error = $"Log root does not exist: {logRootPath}";
                return false;
            }

            string rawPrefix = string.IsNullOrWhiteSpace(userId) ? "-1" : userId.Trim();
            string normalizedPrefix = NormalizeLogFolderToken(rawPrefix);
            var prefixCandidates = new HashSet<string>(StringComparer.Ordinal)
            {
                rawPrefix,
                normalizedPrefix
            };
            string expectedUserId = NormalizeContextToken(userId);
            string expectedConditionId = NormalizeContextToken(conditionId);
            string expectedGroupId = NormalizeContextToken(groupId);

            string normalizedConditionFolder = NormalizeLogFolderToken(conditionId);
            var candidateDirs = new List<string>();
            AddCandidateDirectory(candidateDirs, logRootPath);

            foreach (string dir in Directory.GetDirectories(logRootPath).OrderByDescending(Directory.GetLastWriteTimeUtc))
            {
                string name = Path.GetFileName(dir);
                bool matchesLegacyUserRun = false;
                foreach (string prefix in prefixCandidates)
                {
                    if (string.Equals(name, prefix, StringComparison.Ordinal) ||
                        name.StartsWith(prefix + "_", StringComparison.Ordinal))
                    {
                        matchesLegacyUserRun = true;
                        break;
                    }
                }

                if (matchesLegacyUserRun || string.Equals(name, normalizedConditionFolder, StringComparison.Ordinal))
                    AddCandidateDirectory(candidateDirs, dir);

                string nestedConditionDir = Path.Combine(dir, normalizedConditionFolder);
                if (Directory.Exists(nestedConditionDir))
                    AddCandidateDirectory(candidateDirs, nestedConditionDir);
            }

            for (int i = candidateDirs.Count - 1; i >= 0; i--)
            {
                string dir = candidateDirs[i];
                foreach (string child in Directory.GetDirectories(dir).OrderByDescending(Directory.GetLastWriteTimeUtc))
                {
                    string childName = Path.GetFileName(child);
                    if (string.Equals(childName, "run", StringComparison.Ordinal) ||
                        childName.StartsWith("run_", StringComparison.Ordinal) ||
                        string.Equals(childName, "single", StringComparison.Ordinal) ||
                        string.Equals(childName, "multi", StringComparison.Ordinal))
                    {
                        AddCandidateDirectory(candidateDirs, child);
                    }
                }
            }

            if (candidateDirs.Count == 0)
            {
                error =
                    $"No log directory found for user prefix(es) '{string.Join("', '", prefixCandidates)}' in {logRootPath}.";
                return false;
            }

            foreach (string dir in candidateDirs)
            {
                string candidate = Path.Combine(dir, "ObservationsPerEvaluation.csv");
                if (!File.Exists(candidate))
                    continue;

                // Read each candidate once; the selection works on the same lines.
                string[] lines;
                try
                {
                    lines = ReadAllLinesShared(candidate);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"FinalDesignSelector: could not read '{candidate}': {ex.Message}");
                    continue;
                }

                if (CsvContainsContextRows(lines, expectedUserId, expectedConditionId, expectedGroupId))
                {
                    csvPath = candidate;
                    csvLines = lines;
                    return true;
                }
            }

            error =
                "No observation CSV found for the current user/condition/group context. " +
                $"Context: UserID='{expectedUserId}', ConditionID='{expectedConditionId}', GroupID='{expectedGroupId}'.";
            return false;
        }

        private static void AddCandidateDirectory(List<string> candidateDirs, string directory)
        {
            if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
                return;

            if (!candidateDirs.Contains(directory))
                candidateDirs.Add(directory);
        }

        private static bool TrySelectFinalDesign(
            string csvPath,
            string[] lines,
            string expectedUserId,
            string expectedConditionId,
            string expectedGroupId,
            IList<ParameterEntry> parameters,
            IList<ObjectiveEntry> objectives,
            float distanceEpsilon,
            float maximinEpsilon,
            float aggressionEpsilon,
            out SelectionResult selection,
            out string error
        )
        {
            selection = null;
            error = null;

            if (lines == null || lines.Length < 2)
            {
                error = $"Observation CSV has no data rows: {csvPath}";
                return false;
            }

            string[] header = SplitCsvLine(lines[0], ';');
            var columnIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < header.Length; i++)
            {
                string key = header[i]?.Trim();
                if (!string.IsNullOrEmpty(key) && !columnIndex.ContainsKey(key))
                    columnIndex[key] = i;
            }

            if (!columnIndex.TryGetValue("Iteration", out int iterationIndex))
            {
                error = "CSV column 'Iteration' is missing.";
                return false;
            }

            int[] parameterIndices = new int[parameters.Count];
            for (int i = 0; i < parameters.Count; i++)
            {
                string key = parameters[i].key;
                if (!columnIndex.TryGetValue(key, out int idx))
                {
                    error = $"CSV column for parameter '{key}' is missing.";
                    return false;
                }
                parameterIndices[i] = idx;
            }

            int[] objectiveIndices = new int[objectives.Count];
            for (int i = 0; i < objectives.Count; i++)
            {
                string key = objectives[i].key;
                if (!columnIndex.TryGetValue(key, out int idx))
                {
                    error = $"CSV column for objective '{key}' is missing.";
                    return false;
                }
                objectiveIndices[i] = idx;
            }

            bool hasIsPareto = columnIndex.TryGetValue("IsPareto", out int isParetoIndex);
            bool hasIsBest = columnIndex.TryGetValue("IsBest", out int isBestIndex);
            bool hasPhase = columnIndex.TryGetValue("Phase", out int phaseIndex);
            bool hasUserId = columnIndex.TryGetValue("UserID", out int userIdIndex);
            bool hasConditionId = columnIndex.TryGetValue("ConditionID", out int conditionIdIndex);
            bool hasGroupId = columnIndex.TryGetValue("GroupID", out int groupIdIndex);
            if (!hasUserId || !hasConditionId || !hasGroupId)
            {
                error =
                    "CSV is missing required context columns for final-design selection. " +
                    "Expected columns: UserID, ConditionID, GroupID.";
                return false;
            }

            var rows = new List<CsvRow>(lines.Length - 1);
            var culture = CultureInfo.InvariantCulture;

            for (int lineIdx = 1; lineIdx < lines.Length; lineIdx++)
            {
                string line = lines[lineIdx];
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                string[] parts = SplitCsvLine(line, ';');
                if (parts.Length < header.Length)
                    continue;

                if (!int.TryParse(parts[iterationIndex], NumberStyles.Integer, culture, out int iterationValue))
                    continue;

                if (hasPhase)
                {
                    string phaseValue = parts[phaseIndex]?.Trim();
                    if (string.Equals(phaseValue, "finaldesign", StringComparison.OrdinalIgnoreCase))
                    {
                        // Final-design rows are post-hoc evaluations and must never be selected again.
                        continue;
                    }
                }

                bool userMatches =
                    string.Equals(
                        NormalizeContextToken(parts[userIdIndex]),
                        expectedUserId,
                        StringComparison.Ordinal);
                bool conditionMatches =
                    string.Equals(
                        NormalizeContextToken(parts[conditionIdIndex]),
                        expectedConditionId,
                        StringComparison.Ordinal);
                bool groupMatches =
                    string.Equals(
                        NormalizeContextToken(parts[groupIdIndex]),
                        expectedGroupId,
                        StringComparison.Ordinal);
                if (!userMatches || !conditionMatches || !groupMatches)
                    continue;

                var row = new CsvRow
                {
                    Iteration = iterationValue,
                    ObjectiveRaw = new float[objectives.Count],
                    ObjectiveNormalized = new float[objectives.Count],
                    ParameterRaw = new float[parameters.Count],
                    IsCandidate = true
                };

                bool numericParseFailed = false;
                for (int i = 0; i < objectives.Count; i++)
                {
                    if (!float.TryParse(parts[objectiveIndices[i]], NumberStyles.Float, culture, out row.ObjectiveRaw[i]))
                    {
                        numericParseFailed = true;
                        break;
                    }
                }
                if (numericParseFailed)
                    continue;

                for (int i = 0; i < parameters.Count; i++)
                {
                    if (!float.TryParse(parts[parameterIndices[i]], NumberStyles.Float, culture, out row.ParameterRaw[i]))
                    {
                        numericParseFailed = true;
                        break;
                    }
                }
                if (numericParseFailed)
                    continue;
                if (row.ObjectiveRaw.Any(v => !IsFinite(v)) || row.ParameterRaw.Any(v => !IsFinite(v)))
                    continue;

                if (hasIsPareto)
                {
                    row.IsCandidate = ParseBooleanLike(parts[isParetoIndex], false);
                }
                else if (hasIsBest)
                {
                    row.IsCandidate = ParseBooleanLike(parts[isBestIndex], false);
                }

                rows.Add(row);
            }

            if (rows.Count == 0)
            {
                error =
                    "No valid observation rows could be parsed from CSV for the current context. " +
                    $"Context: UserID='{expectedUserId}', ConditionID='{expectedConditionId}', GroupID='{expectedGroupId}'.";
                return false;
            }

            NormalizeObjectives(rows, objectives);

            string candidateSource = hasIsPareto ? "IsPareto" : hasIsBest ? "IsBest" : "all";
            List<CsvRow> candidateRows = rows.Where(r => r.IsCandidate).ToList();
            if (candidateRows.Count == 0)
            {
                if (hasIsPareto || hasIsBest)
                {
                    // The backend computes the flags over warm-start + live rows but writes only this
                    // participant's rows, so a warm-start incumbent leaves no flagged row here. Recompute the
                    // best/non-dominated set over the participant's own rows instead of skipping the round.
                    candidateRows = ComputeNonDominatedRows(rows, objectives);
                    candidateSource = "computed";
                }
                else
                {
                    candidateRows = rows;
                }
            }

            if (candidateRows.Count == 0)
            {
                error = "No candidate rows could be determined for the current context.";
                return false;
            }

            ComputeMetrics(candidateRows, rows, parameters, objectives);

            List<CsvRow> tieSet = ArgMinWithTies(candidateRows, r => r.UtopiaDistance, distanceEpsilon);
            if (tieSet.Count > 1)
                tieSet = ArgMaxWithTies(tieSet, r => r.Maximin, maximinEpsilon);
            if (tieSet.Count > 1)
                tieSet = ArgMinWithTies(tieSet, r => r.Aggression, aggressionEpsilon);

            CsvRow best = tieSet
                .OrderBy(r => r.Iteration)
                .ThenBy(r => r.UtopiaDistance)
                .ThenByDescending(r => r.Maximin)
                .ThenBy(r => r.Aggression)
                .First();

            selection = new SelectionResult
            {
                Iteration = best.Iteration,
                ParameterRaw = best.ParameterRaw.ToArray(),
                UtopiaDistance = best.UtopiaDistance,
                Maximin = best.Maximin,
                Aggression = best.Aggression,
                CandidateSource = candidateSource
            };
            return true;
        }

        /// <summary>
        /// Objective value as the backends compare it: clamped to the configured bounds (the backends normalize
        /// with clamping) and oriented so that larger is better.
        /// </summary>
        private static float DirectedClampedObjective(float raw, ObjectiveArgs objective)
        {
            float lo = Mathf.Min(objective.lowerBound, objective.upperBound);
            float hi = Mathf.Max(objective.lowerBound, objective.upperBound);
            float value = hi > lo ? Mathf.Clamp(raw, lo, hi) : raw;
            return objective.smallerIsBetter ? -value : value;
        }

        /// <summary>
        /// The rows the backends would flag: for one objective every row with the best value (IsBest ties), for
        /// several objectives the non-dominated rows, keeping only the first copy of duplicated objective vectors
        /// (IsPareto semantics).
        /// </summary>
        private static List<CsvRow> ComputeNonDominatedRows(List<CsvRow> rows, IList<ObjectiveEntry> objectives)
        {
            int nObjs = objectives.Count;
            var directed = new float[rows.Count][];
            for (int r = 0; r < rows.Count; r++)
            {
                directed[r] = new float[nObjs];
                for (int j = 0; j < nObjs; j++)
                    directed[r][j] = DirectedClampedObjective(rows[r].ObjectiveRaw[j], objectives[j].value);
            }

            var result = new List<CsvRow>();
            if (nObjs == 1)
            {
                float best = float.NegativeInfinity;
                for (int r = 0; r < rows.Count; r++)
                    best = Mathf.Max(best, directed[r][0]);

                for (int r = 0; r < rows.Count; r++)
                {
                    if (directed[r][0] == best)
                        result.Add(rows[r]);
                }

                return result;
            }

            for (int i = 0; i < rows.Count; i++)
            {
                bool keep = true;
                for (int k = 0; k < rows.Count && keep; k++)
                {
                    if (k == i)
                        continue;

                    bool allGreaterOrEqual = true;
                    bool anyGreater = false;
                    for (int j = 0; j < nObjs; j++)
                    {
                        if (directed[k][j] < directed[i][j])
                        {
                            allGreaterOrEqual = false;
                            break;
                        }

                        if (directed[k][j] > directed[i][j])
                            anyGreater = true;
                    }

                    if (!allGreaterOrEqual)
                        continue;

                    // Dominated, or an exact duplicate of an earlier row.
                    if (anyGreater || k < i)
                        keep = false;
                }

                if (keep)
                    result.Add(rows[i]);
            }

            return result;
        }

        private static void NormalizeObjectives(List<CsvRow> allRows, IList<ObjectiveEntry> objectives)
        {
            int nObjs = objectives.Count;
            float[] minV = new float[nObjs];
            float[] maxV = new float[nObjs];
            for (int j = 0; j < nObjs; j++)
            {
                minV[j] = float.PositiveInfinity;
                maxV[j] = float.NegativeInfinity;
            }

            foreach (var row in allRows)
            {
                for (int j = 0; j < nObjs; j++)
                {
                    float direction = objectives[j].value.smallerIsBetter ? -1f : 1f;
                    float directed = direction * row.ObjectiveRaw[j];
                    if (directed < minV[j]) minV[j] = directed;
                    if (directed > maxV[j]) maxV[j] = directed;
                }
            }

            foreach (var row in allRows)
            {
                for (int j = 0; j < nObjs; j++)
                {
                    float direction = objectives[j].value.smallerIsBetter ? -1f : 1f;
                    float directed = direction * row.ObjectiveRaw[j];
                    float denom = maxV[j] - minV[j];
                    float normalized = Mathf.Abs(denom) < 1e-12f ? 0.5f : (directed - minV[j]) / denom;
                    row.ObjectiveNormalized[j] = Mathf.Clamp01(normalized);
                }
            }
        }

        private static void ComputeMetrics(
            List<CsvRow> candidates,
            List<CsvRow> allRows,
            IList<ParameterEntry> parameters,
            IList<ObjectiveEntry> objectives
        )
        {
            int nObjs = objectives.Count;
            foreach (var row in candidates)
            {
                float distSq = 0f;
                float maximin = float.PositiveInfinity;
                for (int j = 0; j < nObjs; j++)
                {
                    float delta = 1f - row.ObjectiveNormalized[j];
                    distSq += delta * delta;
                    if (row.ObjectiveNormalized[j] < maximin)
                        maximin = row.ObjectiveNormalized[j];
                }
                row.UtopiaDistance = Mathf.Sqrt(distSq);
                row.Maximin = maximin;
            }

            // Aggression: distance to the middle of the configured design space, every dimension normalized by
            // its configured bounds. (Normalizing by the observed spread let a dimension the optimizer barely
            // varied dominate the tie-break.)
            int nParams = parameters.Count;
            foreach (var row in candidates)
            {
                float sq = 0f;
                for (int j = 0; j < nParams; j++)
                {
                    float lo = Mathf.Min(parameters[j].value.lowerBound, parameters[j].value.upperBound);
                    float hi = Mathf.Max(parameters[j].value.lowerBound, parameters[j].value.upperBound);
                    float d = Normalize01(row.ParameterRaw[j], lo, hi) - 0.5f;
                    sq += d * d;
                }
                row.Aggression = Mathf.Sqrt(sq);
            }
        }

        private static float Normalize01(float value, float min, float max)
        {
            float denom = max - min;
            if (Mathf.Abs(denom) < 1e-12f)
                return 0.5f;
            return Mathf.Clamp01((value - min) / denom);
        }

        private static List<CsvRow> ArgMinWithTies(List<CsvRow> items, Func<CsvRow, float> key, float epsilon)
        {
            float best = float.PositiveInfinity;
            for (int i = 0; i < items.Count; i++)
                best = Mathf.Min(best, key(items[i]));

            float eps = Mathf.Max(0f, epsilon);
            return items.Where(x => Mathf.Abs(key(x) - best) <= eps).ToList();
        }

        private static List<CsvRow> ArgMaxWithTies(List<CsvRow> items, Func<CsvRow, float> key, float epsilon)
        {
            float best = float.NegativeInfinity;
            for (int i = 0; i < items.Count; i++)
                best = Mathf.Max(best, key(items[i]));

            float eps = Mathf.Max(0f, epsilon);
            return items.Where(x => Mathf.Abs(key(x) - best) <= eps).ToList();
        }

        private static bool ParseBooleanLike(string raw, bool defaultValue)
        {
            if (string.IsNullOrWhiteSpace(raw))
                return defaultValue;

            string v = raw.Trim();
            if (string.Equals(v, "1", StringComparison.Ordinal)) return true;
            if (string.Equals(v, "0", StringComparison.Ordinal)) return false;
            if (bool.TryParse(v, out bool b)) return b;
            return defaultValue;
        }

        private static string NormalizeContextToken(string value, string defaultToken = "-1")
        {
            string token = string.IsNullOrWhiteSpace(value) ? defaultToken : value.Trim();
            return string.IsNullOrEmpty(token) ? defaultToken : token;
        }

        // File.ReadAllLines denies write sharing, so it fails while Excel holds the log open on Windows (and
        // while the backend replaces it); share read/write/delete like the Python writers expect.
        private static string[] ReadAllLinesShared(string path)
        {
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            using (var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                var lines = new List<string>();
                string line;
                while ((line = reader.ReadLine()) != null)
                    lines.Add(line);
                return lines.ToArray();
            }
        }

        private static bool CsvContainsContextRows(
            string[] lines,
            string expectedUserId,
            string expectedConditionId,
            string expectedGroupId)
        {
            if (lines == null || lines.Length < 2)
                return false;

            string[] header = SplitCsvLine(lines[0], ';');
            var colIdx = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < header.Length; i++)
            {
                string key = header[i]?.Trim();
                if (!string.IsNullOrEmpty(key) && !colIdx.ContainsKey(key))
                    colIdx[key] = i;
            }

            bool hasUserId = colIdx.TryGetValue("UserID", out int userIdx);
            bool hasConditionId = colIdx.TryGetValue("ConditionID", out int conditionIdx);
            bool hasGroupId = colIdx.TryGetValue("GroupID", out int groupIdx);
            if (!hasUserId || !hasConditionId || !hasGroupId)
                return false;

            for (int lineIdx = 1; lineIdx < lines.Length; lineIdx++)
            {
                string line = lines[lineIdx];
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                string[] parts = SplitCsvLine(line, ';');
                if (parts.Length < header.Length)
                    continue;

                bool userMatches =
                    string.Equals(
                        NormalizeContextToken(parts[userIdx]),
                        expectedUserId,
                        StringComparison.Ordinal);
                bool conditionMatches =
                    string.Equals(
                        NormalizeContextToken(parts[conditionIdx]),
                        expectedConditionId,
                        StringComparison.Ordinal);
                bool groupMatches =
                    string.Equals(
                        NormalizeContextToken(parts[groupIdx]),
                        expectedGroupId,
                        StringComparison.Ordinal);

                if (userMatches && conditionMatches && groupMatches)
                    return true;
            }

            return false;
        }

        private static string NormalizeLogFolderToken(string value)
        {
            // Must match the folder names the writers create (control characters 0x00-0x1F and 0x7F-0x9F too).
            return LogDataFolderUtility.NormalizeLogFolderToken(value);
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static string[] SplitCsvLine(string line, char separator)
        {
            var result = new List<string>();
            bool inQuotes = false;
            var current = new StringBuilder(line.Length);
            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (c == '"')
                {
                    if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        current.Append('"');
                        i++;
                        continue;
                    }
                    inQuotes = !inQuotes;
                    continue;
                }

                if (c == separator && !inQuotes)
                {
                    result.Add(current.ToString());
                    current.Length = 0;
                }
                else
                {
                    current.Append(c);
                }
            }
            result.Add(current.ToString());
            return result.ToArray();
        }
    }
}
