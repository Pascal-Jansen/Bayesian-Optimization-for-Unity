using UnityEngine;
using UnityEngine.SceneManagement;

namespace __BO_FOR_UNITY.Scripts
{
    public class CursorManager : MonoBehaviour
    {
        public bool isCursorLocked = true; // Set this according to your needs

        void Awake()
        {
            DontDestroyOnLoad(gameObject); // Prevent this object from being destroyed on scene load
            SceneManager.sceneLoaded += OnSceneLoaded; // Subscribe to the sceneLoaded event
        }

        void OnDestroy()
        {
            SceneManager.sceneLoaded -= OnSceneLoaded; // Unsubscribe from the sceneLoaded event
        }

        void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            ApplyCursorSettings(); // Apply cursor settings when a new scene is loaded
        }

        void ApplyCursorSettings()
        {
            if (isCursorLocked)
            {
                LockCursor();
            }
            else
            {
                UnlockCursor();
            }
        }

        private void UnlockCursor()
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        private void LockCursor()
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
    }
}