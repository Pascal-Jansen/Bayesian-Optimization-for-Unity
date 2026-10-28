#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BoStudySettings.h"
#include "BoAnalyticDemoActor.generated.h"

class UBayesianOptimizerComponent;

// Add this actor to any map and start launch_backend.py --backend botorch --objectives 1.
UCLASS(Blueprintable)
class BAYESIANOPTIMIZATION_API ABoAnalyticDemoActor : public AActor
{
    GENERATED_BODY()

public:
    ABoAnalyticDemoActor();

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Optimization")
    TObjectPtr<UBayesianOptimizerComponent> Optimizer;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    bool bStartOnBeginPlay = true;

protected:
    virtual void BeginPlay() override;

private:
    UFUNCTION()
    void EvaluateDesign(const FBoSuggestion& Suggestion);
};
