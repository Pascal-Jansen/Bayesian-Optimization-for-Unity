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
            lock (ExecutionQueue)
            {
                while (ExecutionQueue.Count > 0)
                {
                    ExecutionQueue.Dequeue().Invoke();
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
