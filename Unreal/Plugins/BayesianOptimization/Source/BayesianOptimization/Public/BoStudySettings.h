#pragma once

#include "CoreMinimal.h"
#include "BoStudySettings.generated.h"

UENUM(BlueprintType)
enum class EBoBackend : uint8
{
    BoTorch,
    CABOP,
    DynamicBO,
    MetaTAF
};

UENUM(BlueprintType)
enum class EBoSessionState : uint8
{
    Disconnected,
    Connecting,
    WaitingForDesign,
    Evaluating,
    Stopping,
    Completed,
    Failed
};

USTRUCT(BlueprintType)
struct BAYESIANOPTIMIZATION_API FBoParameter
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    FString Key = TEXT("x");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    double Low = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    double High = 1.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CABOP")
    FString Group = TEXT("default");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CABOP", meta = (ClampMin = "0"))
    double Tolerance = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CABOP")
    TArray<double> PrefabricatedValues;
};

USTRUCT(BlueprintType)
struct BAYESIANOPTIMIZATION_API FBoObjective
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    FString Key = TEXT("score");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    double Low = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    double High = 1.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    bool bMinimize = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CABOP", meta = (ClampMin = "0"))
    double Weight = 1.0;
};

USTRUCT(BlueprintType)
struct BAYESIANOPTIMIZATION_API FBoStudySettings
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    EBoBackend Backend = EBoBackend::BoTorch;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    TArray<FBoParameter> Parameters;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Optimization")
    TArray<FBoObjective> Objectives;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Budget", meta = (ClampMin = "0"))
    int32 SamplingIterations = 5;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Budget", meta = (ClampMin = "0"))
    int32 OptimizationIterations = 20;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model", meta = (ClampMin = "1"))
    int32 NumRestarts = 10;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model", meta = (ClampMin = "1"))
    int32 RawSamples = 256;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model", meta = (ClampMin = "1"))
    int32 McSamples = 128;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model")
    int32 Seed = 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Warm Start")
    bool bWarmStart = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Warm Start")
    FString InitialParametersDataPath;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Warm Start")
    FString InitialObjectivesDataPath;

    // raw, normalized_max, normalized_native, or auto.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Warm Start")
    FString WarmStartObjectiveFormat = TEXT("raw");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Study")
    FString UserId = TEXT("-1");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Study")
    FString ConditionId = TEXT("-1");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Study")
    FString GroupId = TEXT("-1");

    // Optional object containing config fields, context, and cabopGroupCosts.
    // Core config fields are protected. See README for allowed advanced settings.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Advanced", meta = (MultiLine = true))
    FString AdvancedInitJson = TEXT("{}");
};

USTRUCT(BlueprintType)
struct BAYESIANOPTIMIZATION_API FBoSuggestion
{
    GENERATED_BODY()

    // Values use the original parameter units. Apply these before starting the trial.
    UPROPERTY(BlueprintReadOnly, Category = "Optimization")
    TMap<FString, double> Values;

    // Matches the backend CSV Iteration column.
    UPROPERTY(BlueprintReadOnly, Category = "Optimization")
    int32 Iteration = -1;

    // Full message preserves backend-specific metadata for C++ and Blueprint parsing.
    UPROPERTY(BlueprintReadOnly, Category = "Optimization")
    FString MessageJson;
};
