#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "BoStudySettings.h"
#include "BayesianOptimizerComponent.generated.h"

class FBoTransport;
struct FBoTransportDeleter
{
    void operator()(FBoTransport* Transport) const;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FBoDesignEvent, const FBoSuggestion&, Suggestion);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FBoMetricEvent, const FString&, Metric, double, Value);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FBoErrorEvent, const FString&, Message);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FBoStateEvent, EBoSessionState, State);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FBoCompletedEvent);

UCLASS(ClassGroup = (Optimization), meta = (BlueprintSpawnableComponent))
class BAYESIANOPTIMIZATION_API UBayesianOptimizerComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UBayesianOptimizerComponent();
    virtual ~UBayesianOptimizerComponent() override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    FBoStudySettings Settings;

    // Start a matching external Python backend before StartStudy. Connections stay on loopback.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Connection", meta = (ClampMin = "1", ClampMax = "65535"))
    int32 ServerPort = 56001;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Connection", meta = (ClampMin = "0.1"))
    double ConnectionTimeoutSeconds = 10.0;

    // Includes participant response time. Set this above the longest trial duration.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Connection", meta = (ClampMin = "1"))
    double IdleTimeoutSeconds = 3600.0;

    UPROPERTY(BlueprintReadOnly, Category = "Optimization")
    EBoSessionState State = EBoSessionState::Disconnected;

    UPROPERTY(BlueprintReadOnly, Category = "Optimization")
    FBoSuggestion LastSuggestion;

    UPROPERTY(BlueprintReadOnly, Category = "Optimization")
    FString LastError;

    UPROPERTY(BlueprintAssignable, Category = "Optimization")
    FBoDesignEvent OnDesignSuggested;

    // coverage is backend-specific. tempCoverage is sampling progress in [0,1].
    UPROPERTY(BlueprintAssignable, Category = "Optimization")
    FBoMetricEvent OnMetric;

    UPROPERTY(BlueprintAssignable, Category = "Optimization")
    FBoErrorEvent OnError;

    UPROPERTY(BlueprintAssignable, Category = "Optimization")
    FBoStateEvent OnStateChanged;

    UPROPERTY(BlueprintAssignable, Category = "Optimization")
    FBoCompletedEvent OnCompleted;

    // Validates settings and connects on a worker thread. Returns false on local validation failure.
    UFUNCTION(BlueprintCallable, Category = "Optimization")
    bool StartStudy();

    // Supply every configured objective once, in its original units, after completing the current trial.
    UFUNCTION(BlueprintCallable, Category = "Optimization")
    bool SubmitObjectives(const TMap<FString, double>& Values);

    UFUNCTION(BlueprintCallable, Category = "Optimization")
    void StopStudy(const FString& Reason = TEXT("user_requested"));

    // Immediate local cleanup. EndPlay and destruction also call this.
    UFUNCTION(BlueprintCallable, Category = "Optimization")
    void Disconnect();

    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    TUniquePtr<FBoTransport, FBoTransportDeleter> Transport;
    FBoStudySettings ActiveSettings;
    uint64 SessionGeneration = 0;
    bool bEndingPlay = false;

    void HandleMessage(const FString& Message);
    void SetState(EBoSessionState NewState);
    void Fail(const FString& Message);
    void ReportInputError(const FString& Message);
};
