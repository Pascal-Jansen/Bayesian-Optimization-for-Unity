using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using JetBrains.Annotations;
using UnityEngine;

namespace Packages.Rider.Editor.Util
{
  internal class UnityVersionUtils
  {
    private static string UnityApplicationVersion => Application.unityVersion;

    public static Version UnityVersion
    {
      get
      {
        var ver = UnityApplicationVersion.Split(".".ToCharArray()).Take(2).Aggregate((a, b) => a + "." + b);
        return new Version(ver);
      }
    }

    [CanBeNull]
    public static string FindClosestMatch(string prefix, Version unityVersion, string[] dllFiles)
    {
      
      // Get all matching DLLs based on the prefix

      // List to hold file and version pairs
      var versionedFiles = new List<(string file, Version version)>();

      // Regex to extract version (example: EditorPlugin_v2019.2.dll)
      var versionRegex = new Regex($"{prefix}(\\d+\\.\\d+)");

      foreach (var dllPath in dllFiles)
      {
        var fileName = Path.GetFileName(dllPath);
        var match = versionRegex.Match(fileName);

        if (match.Success && Version.TryParse(match.Groups[1].Value, out var version))
        {
          versionedFiles.Add((dllPath, version));
        }
      }

      // Find the closest version less than or equal to Unity version
      var closestMatch = versionedFiles
        .Where(vf => vf.version <= unityVersion)
        .OrderByDescending(vf => vf.version) // Closest version (descending order)
        .FirstOrDefault();

      return closestMatch.file;
    }
  }
}