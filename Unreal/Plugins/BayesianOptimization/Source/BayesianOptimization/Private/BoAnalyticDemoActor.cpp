#include "BoAnalyticDemoActor.h"
#include "BayesianOptimizerComponent.h"

ABoAnalyticDemoActor::ABoAnalyticDemoActor()
{
    PrimaryActorTick.bCanEverTick = false;
    Optimizer = CreateDefaultSubobject<UBayesianOptimizerComponent>(TEXT("Optimizer"));
    FBoParameter Parameter;
    Parameter.Key = TEXT("x");
    Optimizer->Settings.Parameters.Add(Parameter);
    FBoObjective Objective;
    Objective.Key = TEXT("cost");
    Objective.bMinimize = true;
    Optimizer->Settings.Objectives.Add(Objective);
    Optimizer->Settings.SamplingIterations = 3;
    Optimizer->Settings.OptimizationIterations = 5;
    Optimizer->Settings.UserId = TEXT("analytic-demo");
    Optimizer->Settings.ConditionId = TEXT("quadratic");
}

void ABoAnalyticDemoActor::BeginPlay()
{
    Super::BeginPlay();
    Optimizer->OnDesignSuggested.AddUniqueDynamic(this, &ABoAnalyticDemoActor::EvaluateDesign);
    if (bStartOnBeginPlay) Optimizer->StartStudy();
}

void ABoAnalyticDemoActor::EvaluateDesign(const FBoSuggestion& Suggestion)
{
    const double X = Suggestion.Values.FindRef(TEXT("x"));
    TMap<FString, double> Objectives;
    Objectives.Add(TEXT("cost"), FMath::Square(X - 0.3));
    Optimizer->SubmitObjectives(Objectives);
}
