using System;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace BOforUnity.Editor
{
    /// <summary>
    /// Removes the other platforms' Python installers from a player build. StreamingAssets is copied as a whole, so
    /// without this every desktop build carries all three Installation folders (macOS pkg 71 MB, Windows Python
    /// 28 MB + VC++ runtime 25 MB). Only the built player's copy is touched, never the project's Assets folder.
    /// </summary>
    public class BoInstallationBuildPostprocessor : IPostprocessBuildWithReport
    {
        private static readonly string[] InstallationFolders = { "Windows", "MacOs", "Linux" };

        public int callbackOrder => 0;

        public void OnPostprocessBuild(BuildReport report)
        {
            string keep = InstallationFolderFor(report.summary.platform);
            if (keep == null)
                return; // not a desktop player: nothing of ours to prune

            string streamingAssets = FindPlayerStreamingAssets(report.summary.platform, report.summary.outputPath);
            if (streamingAssets == null)
            {
                Debug.LogWarning(
                    "BOforUnity: could not locate the built player's StreamingAssets folder for " +
                    $"'{report.summary.outputPath}'; the Python installers of all platforms stay in the build.");
                return;
            }

            string installationDir = Path.GetFullPath(Path.Combine(streamingAssets, "BOData", "Installation"));
            if (!Directory.Exists(installationDir))
                return;

            string projectAssets = Path.GetFullPath(Application.dataPath);
            if (IsSameOrInside(installationDir, projectAssets))
            {
                Debug.LogWarning($"BOforUnity: refusing to prune '{installationDir}': it is inside the project's Assets folder.");
                return;
            }

            // macOS: Unity has already code-signed the .app (ad hoc) when this callback runs. Removing files breaks the
            // signature's resource seal, and a downloaded (quarantined) copy is then reported as "damaged".
            string macApp = report.summary.platform == BuildTarget.StandaloneOSX ? MacAppPath(report.summary.outputPath) : null;
            bool resignAdHoc = false;
            if (macApp != null && !CanPruneSignedMacApp(macApp, out resignAdHoc))
                return;

            long freed = 0;
            foreach (string folder in InstallationFolders)
            {
                if (folder == keep)
                    continue;

                string path = Path.Combine(installationDir, folder);
                if (!Directory.Exists(path))
                    continue;

                try
                {
                    long size = DirectorySize(path);
                    Directory.Delete(path, true);
                    freed += size;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"BOforUnity: could not remove '{path}' from the build: {ex.Message}");
                }
            }

            if (freed > 0)
            {
                Debug.Log(
                    $"BOforUnity: removed the other platforms' Python installers from the {keep} build " +
                    $"({freed / (1024.0 * 1024.0):0.#} MB) in '{installationDir}'.");
            }

            if (resignAdHoc && freed > 0)
            {
                int exitCode = RunCodesign(
                    "--force --sign - --preserve-metadata=entitlements,requirements,flags,runtime " + Quote(macApp), out string output);
                if (exitCode != 0)
                {
                    Debug.LogWarning(
                        $"BOforUnity: could not renew the ad-hoc code signature of '{macApp}' after removing the other platforms' " +
                        $"installers (codesign exit code {exitCode}: {output.Trim()}). Sign the app again before distributing it: " +
                        $"codesign --force --sign - \"{macApp}\"");
                }
            }
        }

        private static string MacAppPath(string outputPath)
        {
            if (string.IsNullOrEmpty(outputPath))
                return null;
            return outputPath.EndsWith(".app", StringComparison.OrdinalIgnoreCase) ? outputPath : outputPath + ".app";
        }

        // An unsigned app (built on Windows/Linux) and an ad-hoc signed one (Unity's default on macOS; re-signed after
        // pruning) can be pruned. An app signed with an identity (by an earlier post-processor) is left as it is.
        private static bool CanPruneSignedMacApp(string app, out bool resignAdHoc)
        {
            resignAdHoc = false;
            if (!Directory.Exists(Path.Combine(app, "Contents", "_CodeSignature")))
                return true;

            if (Application.platform == RuntimePlatform.OSXEditor && File.Exists(Codesign) &&
                RunCodesign("-dv " + Quote(app), out string details) == 0 &&
                details.IndexOf("Signature=adhoc", StringComparison.Ordinal) >= 0)
            {
                resignAdHoc = true;
                return true;
            }

            Debug.Log(
                $"BOforUnity: '{app}' is already code-signed (not ad hoc, or the signature could not be read here); the " +
                "other platforms' Python installers are kept in the build so the signature stays valid.");
            return false;
        }

        private const string Codesign = "/usr/bin/codesign";

        private static int RunCodesign(string arguments, out string output)
        {
            output = string.Empty;
            try
            {
                using (var process = new Process
                {
                    StartInfo = new ProcessStartInfo(Codesign, arguments)
                    {
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        RedirectStandardError = true, // codesign reports on stderr; stdout stays unredirected (no pipe to fill)
                    },
                })
                {
                    process.Start();
                    output = process.StandardError.ReadToEnd();
                    process.WaitForExit();
                    return process.ExitCode;
                }
            }
            catch (Exception ex)
            {
                output = ex.Message;
                return -1;
            }
        }

        private static string Quote(string path)
        {
            return "\"" + path.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        }

        private static string InstallationFolderFor(BuildTarget target)
        {
            switch (target)
            {
                case BuildTarget.StandaloneWindows:
                case BuildTarget.StandaloneWindows64:
                    return "Windows";
                case BuildTarget.StandaloneOSX:
                    return "MacOs";
                case BuildTarget.StandaloneLinux64:
                    return "Linux";
                default:
                    return null;
            }
        }

        // Windows/Linux: <dir>/<name>.exe|.x86_64 + <dir>/<name>_Data/StreamingAssets.
        // macOS: <name>.app/Contents/Resources/Data/StreamingAssets (an Xcode-project export is not handled).
        private static string FindPlayerStreamingAssets(BuildTarget target, string outputPath)
        {
            if (string.IsNullOrEmpty(outputPath))
                return null;

            string candidate;
            if (target == BuildTarget.StandaloneOSX)
            {
                candidate = Path.Combine(MacAppPath(outputPath), "Contents", "Resources", "Data", "StreamingAssets");
            }
            else
            {
                string directory = Path.GetDirectoryName(outputPath);
                string name = Path.GetFileNameWithoutExtension(outputPath);
                if (string.IsNullOrEmpty(name))
                    return null;
                candidate = Path.Combine(directory ?? string.Empty, name + "_Data", "StreamingAssets");
            }

            return Directory.Exists(candidate) ? candidate : null;
        }

        private static bool IsSameOrInside(string path, string root)
        {
            string normalizedPath = path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            string normalizedRoot = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            return normalizedPath.StartsWith(normalizedRoot, StringComparison.OrdinalIgnoreCase);
        }

        private static long DirectorySize(string path)
        {
            long total = 0;
            foreach (string file in Directory.GetFiles(path, "*", SearchOption.AllDirectories))
            {
                try { total += new FileInfo(file).Length; }
                catch (Exception) { /* size is informational */ }
            }
            return total;
        }
    }
}
