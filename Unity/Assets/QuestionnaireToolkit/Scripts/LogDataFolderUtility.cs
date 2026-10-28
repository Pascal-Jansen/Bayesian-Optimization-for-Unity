using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace QuestionnaireToolkit.Scripts
{
    public static class LogDataFolderUtility
    {
        private static readonly object UserFolderLock = new object();
        private static readonly Dictionary<string, string> ReservedUserFoldersByCondition =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        public static string StreamingAssetsLogRoot =>
            Path.Combine(Application.streamingAssetsPath, "BOData", "LogData");

        public static string PersistentLogRoot =>
            Path.Combine(Application.persistentDataPath, "BOData", "LogData");

        private static string _resolvedLogDataRoot;

        /// <summary>
        /// The log root every writer (Python backends, questionnaires, example telemetry, final-design row)
        /// uses: <c>StreamingAssets/BOData/LogData</c> in the Editor and in builds where it is writable, otherwise
        /// <c>persistentDataPath/BOData/LogData</c> (installed or translocated players, where StreamingAssets is
        /// read-only). Resolved once per session so all writers agree.
        /// </summary>
        public static string LogDataRoot
        {
            get
            {
                if (_resolvedLogDataRoot != null)
                    return _resolvedLogDataRoot;
                _resolvedLogDataRoot = ResolveLogDataRoot();
                return _resolvedLogDataRoot;
            }
        }

        private static string ResolveLogDataRoot()
        {
            string streamingRoot = StreamingAssetsLogRoot;
            if (Application.isEditor || IsWritableDirectory(streamingRoot))
                return streamingRoot;

            string persistentRoot = PersistentLogRoot;
            Debug.LogWarning(
                $"LogData folder '{streamingRoot}' is not writable in this build; logging to '{persistentRoot}' instead."
            );
            return persistentRoot;
        }

        private static bool IsWritableDirectory(string directory)
        {
            try
            {
                Directory.CreateDirectory(directory);
                string probe = Path.Combine(directory, ".write_probe_" + Guid.NewGuid().ToString("N"));
                File.WriteAllText(probe, string.Empty);
                File.Delete(probe);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        // Static state survives play sessions when "Enter Play Mode Options" disable domain reload; a stale
        // reservation would hand the next session the previous run's folder without checking the disk.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void ResetStaticState()
        {
            lock (UserFolderLock)
            {
                ReservedUserFoldersByCondition.Clear();
            }

            _resolvedLogDataRoot = null;
        }

        /// <summary>
        /// Canonical form of a log root (absolute, no trailing separator), so that <c>.../LogData</c> and
        /// <c>.../LogData/</c> are the same root for folder reservations. Every caller that compares or keys
        /// log roots must go through this.
        /// </summary>
        public static string NormalizeRoot(string root)
        {
            if (string.IsNullOrWhiteSpace(root))
                return root;

            string fullPath = Path.GetFullPath(root.Trim());
            string pathRoot = Path.GetPathRoot(fullPath) ?? string.Empty;
            string trimmed = fullPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            // Keep the separator of a bare drive/file-system root ("/", "C:\").
            return trimmed.Length < pathRoot.Length ? pathRoot : trimmed;
        }

        public static bool IsSameRoot(string a, string b)
        {
            if (string.IsNullOrWhiteSpace(a) || string.IsNullOrWhiteSpace(b))
                return false;

            return string.Equals(NormalizeRoot(a), NormalizeRoot(b), StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Maps a directory that resolves to the project's <c>StreamingAssets/BOData/LogData</c> folder to
        /// <see cref="LogDataRoot"/> (which falls back to persistentDataPath where StreamingAssets is read-only),
        /// so questionnaires and the optimizer share one root. Other directories are returned normalized.
        /// </summary>
        public static string RedirectDefaultLogRoot(string directory)
        {
            if (string.IsNullOrWhiteSpace(directory))
                return NormalizeRoot(LogDataRoot);

            return IsSameRoot(directory, StreamingAssetsLogRoot)
                ? NormalizeRoot(LogDataRoot)
                : NormalizeRoot(directory);
        }

        public static string GetOrCreateUserFolderTokenForCondition(
            string logRoot,
            string requestedUserId,
            string conditionId,
            bool allowExistingRequestedUserFolder = false,
            bool allowExistingConditionFolder = false)
        {
            string normalizedRoot = NormalizeRoot(string.IsNullOrWhiteSpace(logRoot) ? LogDataRoot : logRoot);
            string baseToken = NormalizeLogFolderToken(requestedUserId);
            string conditionToken = NormalizeLogFolderToken(conditionId);
            string reservationKey = GetReservationKey(normalizedRoot, baseToken, conditionToken);

            lock (UserFolderLock)
            {
                if (ReservedUserFoldersByCondition.TryGetValue(reservationKey, out string reservedToken))
                    return reservedToken;

                Directory.CreateDirectory(normalizedRoot);

                string selectedToken = SelectUserFolderTokenForCondition(
                    normalizedRoot,
                    baseToken,
                    conditionToken,
                    allowExistingRequestedUserFolder,
                    allowExistingConditionFolder
                );
                Directory.CreateDirectory(Path.Combine(normalizedRoot, selectedToken, conditionToken));
                ReserveConditionFolder(normalizedRoot, baseToken, conditionToken, selectedToken);
                ReserveConditionFolder(normalizedRoot, selectedToken, conditionToken, selectedToken);
                return selectedToken;
            }
        }

        private static string SelectUserFolderTokenForCondition(
            string normalizedRoot,
            string baseToken,
            string conditionToken,
            bool allowExistingRequestedUserFolder,
            bool allowExistingConditionFolder)
        {
            int suffix = 0;
            while (true)
            {
                string candidateToken = suffix == 0
                    ? baseToken
                    : baseToken + "_" + suffix.ToString(System.Globalization.CultureInfo.InvariantCulture);
                string candidateUserPath = Path.Combine(normalizedRoot, candidateToken);
                string candidateConditionPath = Path.Combine(candidateUserPath, conditionToken);

                if (!File.Exists(candidateUserPath) && !Directory.Exists(candidateUserPath))
                    return candidateToken;

                bool canUseExistingUserFolder = suffix > 0 || allowExistingRequestedUserFolder;
                if (!File.Exists(candidateUserPath) &&
                    canUseExistingUserFolder &&
                    allowExistingConditionFolder &&
                    Directory.Exists(candidateConditionPath))
                    return candidateToken;

                if (!File.Exists(candidateUserPath) &&
                    canUseExistingUserFolder &&
                    !Directory.Exists(candidateConditionPath) &&
                    !File.Exists(candidateConditionPath))
                    return candidateToken;

                suffix++;
            }
        }

        private static void ReserveConditionFolder(
            string normalizedRoot,
            string requestedToken,
            string conditionToken,
            string selectedToken)
        {
            ReservedUserFoldersByCondition[GetReservationKey(normalizedRoot, requestedToken, conditionToken)] =
                selectedToken;
        }

        private static string GetReservationKey(string normalizedRoot, string requestedToken, string conditionToken)
        {
            return normalizedRoot + "\n" + requestedToken + "\n" + conditionToken;
        }

        public static string NormalizeLogFolderToken(string value)
        {
            string token = string.IsNullOrWhiteSpace(value) ? "-1" : value.Trim();
            char[] invalidChars = { '/', '\\', ':', '*', '?', '"', '<', '>', '|' };
            var builder = new StringBuilder(token.Length);
            for (int i = 0; i < token.Length; i++)
            {
                char c = token[i];
                builder.Append(Array.IndexOf(invalidChars, c) >= 0 || char.IsControl(c) ? '_' : c);
            }

            string cleaned = builder.ToString().Trim().Trim('.');
            if (string.IsNullOrWhiteSpace(cleaned) || cleaned == "." || cleaned == "..")
                return "-1";

            return cleaned;
        }
    }
}
