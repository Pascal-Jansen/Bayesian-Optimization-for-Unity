using System;
using UnityEngine;
using UnityEngine.EventSystems;

namespace BayesianOptimization.Scripts
{
    public static class BoEventSystemUtility
    {
        public static EventSystem EnsureEventSystem()
        {
            var existing = UnityEngine.Object.FindAnyObjectByType<EventSystem>();
            if (existing != null) return existing;
            var target = new GameObject("EventSystem", typeof(EventSystem));
#if ENABLE_LEGACY_INPUT_MANAGER
            target.AddComponent<StandaloneInputModule>();
#else
            var module = Type.GetType("UnityEngine.InputSystem.UI.InputSystemUIInputModule, Unity.InputSystem", false);
            if (module != null && typeof(BaseInputModule).IsAssignableFrom(module))
                target.AddComponent(module);
            else
                Debug.LogWarning("Bayesian Optimization: install the Input System package or enable legacy input to use interactive UI.");
#endif
            return target.GetComponent<EventSystem>();
        }
    }
}
