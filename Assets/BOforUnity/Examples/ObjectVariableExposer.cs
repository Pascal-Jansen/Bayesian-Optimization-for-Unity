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

            var bo = GameObject.FindWithTag("BOforUnityManager").GetComponent<BoForUnityManager>().parameters;
            
            var i = isCube? 0 : 5;

            _renderer.material.color = new Color(bo[i].value.Value, bo[i+1].value.Value, bo[i+2].value.Value, bo[i+3].value.Value);
            
            _renderer.enabled = bo[i+4].value.Value >= 0.5f;
        }

        public void StartQuestionnaire()
        {
            GameObject.FindWithTag("QTQuestionnaireManager").GetComponent<QTQuestionnaireManager>()
                .StartQuestionnaire();
        }
    }
}