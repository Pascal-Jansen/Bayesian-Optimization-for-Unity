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

        public static string GetOrCreateUserFolderTokenForCondition(
            string logRoot,
            string requestedUserId,
            string conditionId,
            bool allowExistingRequestedUserFolder = false,
            bool allowExistingConditionFolder = false)
        {
            string root = string.IsNullOrWhiteSpace(logRoot) ? StreamingAssetsLogRoot : logRoot;
            string normalizedRoot = Path.GetFullPath(root);
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
