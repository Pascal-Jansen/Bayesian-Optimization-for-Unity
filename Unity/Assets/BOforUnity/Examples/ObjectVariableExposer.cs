using BOforUnity.Scripts;
using QuestionnaireToolkit.Scripts;
using UnityEngine;

namespace BOforUnity.Examples
{
    /// <summary>
    /// Applies five design parameters (color R, G, B, transparency, visibility) to this object's renderer and
    /// exposes the applied values in the fields below.
    /// </summary>
    public class ObjectVariableExposer : MonoBehaviour
    {
        // Applied renderer values (read-only output, normalized to [0,1] by each parameter's bounds).
        public float colorR;
        public float colorG;
        public float colorB;
        public float colorA;

        // Applied visibility (renderer enabled).
        public bool isActive;

        private MeshRenderer _renderer;

        [Tooltip("Selects the parameter set: '<prefix>ColorR', '<prefix>ColorG', '<prefix>ColorB', " +
                 "'<prefix>Transparency', '<prefix>Visibility' with prefix 'Cube' or 'Cylinder'. When a key is " +
                 "missing, the parameter at list position 1-5 (cube) or 6-10 (cylinder) is used.")]
        public bool isCube;

        [Tooltip("Overrides the key prefix ('Cube'/'Cylinder' from Is Cube when empty).")]
        public string parameterKeyPrefix = "";

        private static readonly string[] KeySuffixes = { "ColorR", "ColorG", "ColorB", "Transparency", "Visibility" };

        void Start()
        {
            _renderer = GetComponent<MeshRenderer>();
            if (_renderer == null)
            {
                Debug.LogWarning("ObjectVariableExposer: MeshRenderer is missing.");
                return;
            }

            var manager = BoParameterReader.FindManager();
            if (manager == null || manager.parameters == null)
            {
                Debug.LogWarning("ObjectVariableExposer: BoForUnityManager or parameters are missing.");
                return;
            }

            string prefix = string.IsNullOrWhiteSpace(parameterKeyPrefix)
                ? (isCube ? "Cube" : "Cylinder")
                : parameterKeyPrefix.Trim();
            int offset = isCube ? 0 : 5;
            var values = new float[KeySuffixes.Length];
            for (int k = 0; k < KeySuffixes.Length; k++)
            {
                if (!BoParameterReader.TryGetNormalized(manager, prefix + KeySuffixes[k], offset + k, out values[k]))
                {
                    Debug.LogWarning("ObjectVariableExposer: Not enough valid BO parameters to expose object variables.");
                    return;
                }
            }

            colorR = values[0];
            colorG = values[1];
            colorB = values[2];
            colorA = values[3];
            isActive = values[4] >= 0.5f;

            _renderer.material.color = new Color(colorR, colorG, colorB, colorA);
            _renderer.enabled = isActive;
        }

        public void StartQuestionnaire()
        {
            var questionnaire = FindAnyObjectByType<QTQuestionnaireManager>();
            if (questionnaire == null)
            {
                Debug.LogWarning("ObjectVariableExposer: QTQuestionnaireManager is missing.");
                return;
            }

            questionnaire.StartQuestionnaire();
        }
    }
}
