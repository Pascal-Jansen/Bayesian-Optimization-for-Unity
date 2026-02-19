using System.Diagnostics;
using Debug = UnityEngine.Debug;

namespace Packages.Rider.Editor
{
  internal class LogFileOpener
  {
    public static void OpenFileAtLineExternal(string filePath)
    {
      try
      {
        switch (UnityEngine.SystemInfo.operatingSystemFamily)
        {
          case UnityEngine.OperatingSystemFamily.Windows:
            Process.Start(new ProcessStartInfo(filePath) { UseShellExecute = true });
            break;
          case UnityEngine.OperatingSystemFamily.MacOSX:
            Process.Start("open", $"\"{filePath}\"");
            break;
          case UnityEngine.OperatingSystemFamily.Linux:
            Process.Start("xdg-open", filePath);
            break;
          default:
            Debug.LogWarning("Platform not supported for opening files.");
            break;
        }
      }
      catch (System.Exception e)
      {
        Debug.Log($"Failed to open file: {e.Message}");
      }
    }
  }  
}
