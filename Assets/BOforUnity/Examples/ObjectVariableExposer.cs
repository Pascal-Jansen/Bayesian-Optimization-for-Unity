using QuestionnaireToolkit.Scripts;
using UnityEngine;

namespace BOforUnity.Examples
{
    public class ObjectVariableExposer : MonoBehaviour
    {
        // Renderer variables
        public float colorR;
        public float colorG;
        public float colorB;
        public float colorA;

        // GameObject active state
        public bool isActive;

        private MeshRenderer _renderer;

        public bool isCube;
        
        void Start()
        {
            _renderer = GetComponent<MeshRenderer>();
            if (_renderer == null)
            {
                Debug.LogWarning("ObjectVariableExposer: MeshRenderer is missing.");
                return;
            }

            var manager = FindObjectOfType<BoForUnityManager>();
            if (manager == null || manager.parameters == null)
            {
                Debug.LogWarning("ObjectVariableExposer: BoForUnityManager or parameters are missing.");
                return;
            }
            var bo = manager.parameters;
            
            var i = isCube? 0 : 5;
            if (bo.Count <= i + 4 ||
                bo[i] == null || bo[i].value == null ||
                bo[i + 1] == null || bo[i + 1].value == null ||
                bo[i + 2] == null || bo[i + 2].value == null ||
                bo[i + 3] == null || bo[i + 3].value == null ||
                bo[i + 4] == null || bo[i + 4].value == null)
            {
                Debug.LogWarning("ObjectVariableExposer: Not enough valid BO parameters to expose object variables.");
                return;
            }

            _renderer.material.color = new Color(bo[i].value.Value, bo[i+1].value.Value, bo[i+2].value.Value, bo[i+3].value.Value);
            
            _renderer.enabled = bo[i+4].value.Value >= 0.5f;
        }

        public void StartQuestionnaire()
        {
            var questionnaire = FindObjectOfType<QTQuestionnaireManager>();
            if (questionnaire == null)
            {
                Debug.LogWarning("ObjectVariableExposer: QTQuestionnaireManager is missing.");
                return;
            }

            questionnaire.StartQuestionnaire();
        }
    }
}
