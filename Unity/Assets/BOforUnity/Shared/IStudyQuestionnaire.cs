using System.Collections.Generic;
using UnityEngine;

namespace BayesianOptimization.Integration
{
    /// <summary>Optional questionnaire integration used by the legacy study examples.</summary>
    public interface IStudyQuestionnaire
    {
        bool StartQuestionnaire();
        void BuildHeaderItems();
        IEnumerable<string> SliderHeaderNames { get; }
        string ContextUserId { get; set; }
        string ContextConditionId { get; set; }
        string ContextGroupId { get; set; }
        void ResolveBoContextForLogging(out string userId, out string conditionId, out string groupId);
        bool EnsureAdditionalCsvItem(string headerName, Object source, string componentName, string memberName, int order);
    }

    public static class StudyQuestionnaires
    {
        public static IStudyQuestionnaire Resolve(MonoBehaviour preferred = null)
        {
            if (preferred != null && preferred is IStudyQuestionnaire questionnaire)
                return questionnaire;
            return FindComponent() as IStudyQuestionnaire;
        }

        public static MonoBehaviour FindComponent()
        {
#if UNITY_6000_6_OR_NEWER
            var behaviours = Object.FindObjectsByType<MonoBehaviour>();
#else
            var behaviours = Object.FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None);
#endif
            foreach (var behaviour in behaviours)
            {
                if (behaviour.isActiveAndEnabled && behaviour is IStudyQuestionnaire)
                    return behaviour;
            }
            return null;
        }
    }
}
