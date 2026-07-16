using System;
using System.Collections.Generic;
using UnityEngine;

namespace BOforUnity.Scripts
{
    public class MainThreadDispatcher : MonoBehaviour
    {
        private static readonly Queue<Action> ExecutionQueue = new Queue<Action>();

        private void Update()
        {
            // Dequeue outside the lock so background Execute() calls are not
            // blocked while actions run, and isolate action failures so one
            // throwing action cannot abort the rest of this frame's queue.
            while (true)
            {
                Action action;
                lock (ExecutionQueue)
                {
                    if (ExecutionQueue.Count == 0)
                        break;
                    action = ExecutionQueue.Dequeue();
                }

                try
                {
                    action.Invoke();
                }
                catch (Exception e)
                {
                    Debug.LogException(e);
                }
            }
        }

        public static void Execute(Action action)
        {
            lock (ExecutionQueue)
            {
                ExecutionQueue.Enqueue(action);
            }
        }
        
        public static void Execute<T>(Action<T> action, T param)
        {
            lock (ExecutionQueue)
            {
                ExecutionQueue.Enqueue(() => action(param));
            }
        }
    }
}
