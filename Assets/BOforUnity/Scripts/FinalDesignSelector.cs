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

            if (parameters == null || parameters.Count == 0)
            {
                error = "No parameters are defined.";
                return false;
            }
            if (objectives == null || objectives.Count == 0)
            {
                error = "No objectives are defined.";
                return false;
            }
            if (string.IsNullOrWhiteSpace(logRootPath))
            {
                error = "Log root path is empty.";
                return false;
            }

            if (!TryGetLatestObservationCsvPath(logRootPath, userId, out selectedCsvPath, out error))
            {
                return false;
            }

            if (!TrySelectFinalDesign(
                    selectedCsvPath,
                    parameters,
                    objectives,
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

        private static bool TryGetLatestObservationCsvPath(
            string logRootPath,
            string userId,
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

            string prefix = string.IsNullOrWhiteSpace(userId) ? "-1" : userId.Trim();
            string[] candidateDirs = Directory.GetDirectories(logRootPath)
                .Where(d =>
                {
                    string name = Path.GetFileName(d);
                    return string.Equals(name, prefix, StringComparison.Ordinal) ||
                           name.StartsWith(prefix + "_", StringComparison.Ordinal);
                })
                .OrderByDescending(Directory.GetLastWriteTimeUtc)
                .ToArray();

            if (candidateDirs.Length == 0)
            {
                error = $"No log directory found for user prefix '{prefix}' in {logRootPath}.";
                return false;
            }

            string newestDir = candidateDirs[0];
            string candidate = Path.Combine(newestDir, "ObservationsPerEvaluation.csv");
            if (!File.Exists(candidate))
            {
                error =
                    "Latest log directory does not contain ObservationsPerEvaluation.csv. " +
                    "This can happen when no observations were logged for the current run.";
                return false;
            }

            csvPath = candidate;
            return true;
        }

        private static bool TrySelectFinalDesign(
            string csvPath,
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
                    row.IsCandidate = ParseBooleanLike(parts[isParetoIndex], true);
                }
                else if (hasIsBest)
                {
                    row.IsCandidate = ParseBooleanLike(parts[isBestIndex], true);
                }

                rows.Add(row);
            }

            if (rows.Count == 0)
            {
                error = "No valid observation rows could be parsed from CSV.";
                return false;
            }

            NormalizeObjectives(rows, objectives);

            List<CsvRow> candidateRows = rows.Where(r => r.IsCandidate).ToList();
            if (candidateRows.Count == 0)
                candidateRows = rows;

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
