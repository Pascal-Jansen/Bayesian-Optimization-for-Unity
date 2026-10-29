using System.Collections;
using BayesianOptimization.Scripts;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.TestTools;

namespace BayesianOptimization.Tests.EditMode
{
    public class SessionLifecycleEditModeTests
    {
        [UnityTest]
        public IEnumerator ClearsSingletonAndQueuedCallbacksAcrossSessionsWithoutDomainReload()
        {
            bool originalEnabled = EditorSettings.enterPlayModeOptionsEnabled;
            var originalOptions = EditorSettings.enterPlayModeOptions;
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            EditorSettings.enterPlayModeOptionsEnabled = true;
            EditorSettings.enterPlayModeOptions = EnterPlayModeOptions.DisableDomainReload;
            try
            {
                yield return new EnterPlayMode();
                var first = new GameObject("FirstSession", typeof(MainThreadDispatcher), typeof(BoForUnityManager));
                Assert.That(BoForUnityManager.Instance, Is.EqualTo(first.GetComponent<BoForUnityManager>()));
                yield return new ExitPlayMode();

                bool staleCallbackRan = false;
                MainThreadDispatcher.Execute(() => staleCallbackRan = true);
                yield return new EnterPlayMode();
                var second = new GameObject("SecondSession", typeof(MainThreadDispatcher), typeof(BoForUnityManager));
                Assert.That(BoForUnityManager.Instance, Is.EqualTo(second.GetComponent<BoForUnityManager>()));
                yield return null;
                Assert.That(staleCallbackRan, Is.False);
                yield return new ExitPlayMode();
            }
            finally
            {
                EditorSettings.enterPlayModeOptions = originalOptions;
                EditorSettings.enterPlayModeOptionsEnabled = originalEnabled;
            }
        }
    }
}
