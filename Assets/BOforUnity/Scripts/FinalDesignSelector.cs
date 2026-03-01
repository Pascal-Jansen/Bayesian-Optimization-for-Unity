using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using BOforUnity;

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

            if (!TryGetLatestObservationCsvPath(logRootPath, userId, conditionId, groupId, out selectedCsvPath, out error))
            {
                return false;
            }

            if (!TrySelectFinalDesign(
                    selectedCsvPath,
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
            out string error
        )
        {
            csvPath = null;
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

            string[] candidateDirs = Directory.GetDirectories(logRootPath)
                .Where(d =>
                {
                    string name = Path.GetFileName(d);
                    foreach (string prefix in prefixCandidates)
                    {
                        if (string.Equals(name, prefix, StringComparison.Ordinal) ||
                            name.StartsWith(prefix + "_", StringComparison.Ordinal))
                        {
                            return true;
                        }
                    }

                    return false;
                })
                .OrderByDescending(Directory.GetLastWriteTimeUtc)
                .ToArray();

            if (candidateDirs.Length == 0)
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

                if (CsvContainsContextRows(candidate, expectedUserId, expectedConditionId, expectedGroupId))
                {
                    csvPath = candidate;
                    return true;
                }
            }

            error =
                "No observation CSV found for the current user/condition/group context. " +
                $"Context: UserID='{expectedUserId}', ConditionID='{expectedConditionId}', GroupID='{expectedGroupId}'.";
            return false;
        }

        private static bool TrySelectFinalDesign(
            string csvPath,
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

            string[] lines = File.ReadAllLines(csvPath);
            if (lines.Length < 2)
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

            List<CsvRow> candidateRows = rows.Where(r => r.IsCandidate).ToList();
            if (candidateRows.Count == 0)
            {
                if (hasIsPareto || hasIsBest)
                {
                    error =
                        "No candidate rows are marked in the current context. " +
                        "All rows are non-candidates according to IsPareto/IsBest.";
                    return false;
                }
                candidateRows = rows;
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
                Aggression = best.Aggression
            };
            return true;
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

            int nParams = parameters.Count;
            float[] minParam = new float[nParams];
            float[] maxParam = new float[nParams];
            for (int j = 0; j < nParams; j++)
            {
                minParam[j] = float.PositiveInfinity;
                maxParam[j] = float.NegativeInfinity;
            }

            foreach (var row in allRows)
            {
                for (int j = 0; j < nParams; j++)
                {
                    float v = row.ParameterRaw[j];
                    if (v < minParam[j]) minParam[j] = v;
                    if (v > maxParam[j]) maxParam[j] = v;
                }
            }

            float[] baselineNormalized = new float[nParams];
            for (int j = 0; j < nParams; j++)
            {
                float baselineRaw = 0.5f * (parameters[j].value.lowerBound + parameters[j].value.upperBound);
                baselineNormalized[j] = Normalize01(baselineRaw, minParam[j], maxParam[j]);
            }

            foreach (var row in candidates)
            {
                float sq = 0f;
                for (int j = 0; j < nParams; j++)
                {
                    float rowNorm = Normalize01(row.ParameterRaw[j], minParam[j], maxParam[j]);
                    float d = rowNorm - baselineNormalized[j];
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

        private static bool CsvContainsContextRows(
            string csvPath,
            string expectedUserId,
            string expectedConditionId,
            string expectedGroupId)
        {
            string[] lines;
            try
            {
                lines = File.ReadAllLines(csvPath);
            }
            catch
            {
                return false;
            }

            if (lines.Length < 2)
                return false;

            string[] header = SplitCsvLine(lines[0], ';');
            var colIdx = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < header.Length; i++)
            {
                string key = header[i]?.Trim();
                if (!string.IsNullOrEmpty(key) && !colIdx.ContainsKey(key))
                    colIdx[key] = i;
            }

            colIdx.TryGetValue("UserID", out int userIdx);
            colIdx.TryGetValue("ConditionID", out int conditionIdx);
            colIdx.TryGetValue("GroupID", out int groupIdx);
            if (userIdx < 0 || conditionIdx < 0 || groupIdx < 0)
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

        private static string NormalizeLogFolderToken(string value, string defaultToken = "-1")
        {
            string token = string.IsNullOrWhiteSpace(value) ? defaultToken : value.Trim();
            if (string.IsNullOrEmpty(token))
                token = defaultToken;

            var sb = new StringBuilder(token.Length);
            for (int i = 0; i < token.Length; i++)
            {
                char ch = token[i];
                bool invalid =
                    ch < 32 ||
                    ch == '/' || ch == '\\' ||
                    ch == ':' || ch == '*' || ch == '?' ||
                    ch == '"' || ch == '<' || ch == '>' || ch == '|';
                sb.Append(invalid ? '_' : ch);
            }

            string cleaned = sb.ToString().Trim().Trim('.');
            if (string.IsNullOrEmpty(cleaned) || cleaned == "." || cleaned == "..")
                return defaultToken;

            return cleaned;
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
