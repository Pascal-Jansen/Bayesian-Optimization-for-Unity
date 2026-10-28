using UnityEngine;

namespace QuestionnaireToolkit.Scripts
{
    [CreateAssetMenu(fileName = "QTMetaData", menuName = "QTMetaData", order = 1)]
    public class QuestionnaireMeta : ScriptableObject
    {
        public int currentResponseId = 0;
        public int currentUserRun = 0;
    }
}
