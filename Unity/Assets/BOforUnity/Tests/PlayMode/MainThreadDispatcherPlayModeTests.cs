using System.Reflection;
using System.Threading;
using NUnit.Framework;
using UnityEngine;
using BOforUnity.Scripts;

namespace BOforUnity.Tests.PlayMode
{
    public class MainThreadDispatcherPlayModeTests
    {
        [Test]
        public void Execute_ProcessesQueuedActionsOnMainThreadInOrder()
        {
            var dispatcherGo = new GameObject("MainThreadDispatcherTest");
            var dispatcher = dispatcherGo.AddComponent<MainThreadDispatcher>();

            try
            {
                var mainThreadId = Thread.CurrentThread.ManagedThreadId;
                int? actionThreadId = null;
                int? genericActionThreadId = null;
                string executionTrace = string.Empty;

                var worker = new Thread(() =>
                {
                    MainThreadDispatcher.Execute(() =>
                    {
                        actionThreadId = Thread.CurrentThread.ManagedThreadId;
                        executionTrace += "A";
                    });

                    MainThreadDispatcher.Execute<string>(marker =>
                    {
                        genericActionThreadId = Thread.CurrentThread.ManagedThreadId;
                        executionTrace += marker;
                    }, "B");
                });

                worker.Start();
                worker.Join();

                InvokeDispatcherUpdate(dispatcher);

                Assert.That(executionTrace, Is.EqualTo("AB"), "Queued actions should execute in FIFO order.");
                Assert.That(actionThreadId, Is.EqualTo(mainThreadId), "Actions must execute on the main Unity thread.");
                Assert.That(genericActionThreadId, Is.EqualTo(mainThreadId), "Generic Execute overload must run on main thread.");
            }
            finally
            {
                Object.DestroyImmediate(dispatcherGo);
            }
        }

        private static void InvokeDispatcherUpdate(MainThreadDispatcher dispatcher)
        {
            MethodInfo updateMethod = typeof(MainThreadDispatcher).GetMethod(
                "Update",
                BindingFlags.Instance | BindingFlags.NonPublic
            );
            Assert.That(updateMethod, Is.Not.Null, "MainThreadDispatcher.Update should exist.");
            updateMethod.Invoke(dispatcher, null);
        }
    }
}
