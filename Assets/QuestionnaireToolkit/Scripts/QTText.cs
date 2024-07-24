using UnityEditor;
using UnityEngine;

namespace QuestionnaireToolkit.Scripts
{
    [ExecuteInEditMode]
    public class QTText : MonoBehaviour
    {
        [HideInInspector]
        public QTQuestionnaireManager _questionnaireManager;

//#if UNITY_EDITOR
        private void Start()
        {
            if (!Application.isPlaying)
            {
                
            }
            _questionnaireManager = transform.parent.parent.parent.parent.parent.GetComponent<QTQuestionnaireManager>();
        }
//#endif
    }
}
