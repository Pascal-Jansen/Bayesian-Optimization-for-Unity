using UnityEditor;
using UnityEngine;

namespace QuestionnaireToolkit.Scripts
{
    [ExecuteInEditMode]
    public class QTImage : MonoBehaviour
    {
        [HideInInspector]
        public QTQuestionnaireManager _questionnaireManager;

        private void Start()
        {
//#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                
            }
            _questionnaireManager = transform.parent.parent.parent.parent.parent.GetComponent<QTQuestionnaireManager>();
//#endif
        }
    }
}
