namespace QuestionnaireToolkit.Scripts
{
    public interface IQuestionnaireOptimizationBridge
    {
        bool UsesExternalIterationSignal { get; }
        bool EnablePriorRatingHints { get; }
        float PriorRatingHintAlpha { get; }
        string UserId { get; }
        string ConditionId { get; }
        string GroupId { get; }

        void OptimizationStart();
        void RequestNextIteration();
        void SubmitQuestionnaireObjectiveValue(string headerName, string rawValue, string sourceName);

        void SetPriorSliderRatingHint(string questionKey, float sliderValue);
        bool TryGetPriorSliderRatingHint(string questionKey, out float sliderValue);
        void RemovePriorSliderRatingHint(string questionKey);

        void SetPriorLinearScaleRatingHint(string questionKey, string answerValue);
        bool TryGetPriorLinearScaleRatingHint(string questionKey, out string answerValue);
        void RemovePriorLinearScaleRatingHint(string questionKey);
    }
}
