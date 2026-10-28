#include "BoProtocol.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace
{
    bool ValidateKey(const FString& Key, TSet<FString>& Used, FString& Error)
    {
        static const TSet<FString> Reserved = {
            TEXT("userid"), TEXT("conditionid"), TEXT("groupid"), TEXT("timestamp"),
            TEXT("iteration"), TEXT("phase"), TEXT("isbest"), TEXT("ispareto"), TEXT("context")
        };
        const FString Folded = Key.ToLower();
        if (Key.IsEmpty() || Key != Key.TrimStartAndEnd() || Key.Contains(TEXT("\n")) || Key.Contains(TEXT("\r")))
        {
            Error = TEXT("Keys must be nonempty and have no surrounding whitespace or line breaks.");
            return false;
        }
        if (Used.Contains(Folded) || Reserved.Contains(Folded))
        {
            Error = FString::Printf(TEXT("Duplicate or reserved parameter/objective key: %s"), *Key);
            return false;
        }
        Used.Add(Folded);
        return true;
    }

    TSharedRef<FJsonObject> Bounds(double Low, double High)
    {
        const TSharedRef<FJsonObject> Object = MakeShared<FJsonObject>();
        Object->SetNumberField(TEXT("low"), Low);
        Object->SetNumberField(TEXT("high"), High);
        return Object;
    }

    FString BackendName(EBoBackend Backend)
    {
        switch (Backend)
        {
        case EBoBackend::CABOP: return TEXT("cabop");
        case EBoBackend::DynamicBO: return TEXT("dbo");
        case EBoBackend::MetaTAF: return TEXT("meta-taf");
        default: return TEXT("botorch");
        }
    }
}

FString BoProtocol::Serialize(const TSharedRef<FJsonObject>& Object)
{
    FString Json;
    const TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
        TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Json);
    FJsonSerializer::Serialize(Object, Writer);
    return Json;
}

bool BoProtocol::BuildInit(const FBoStudySettings& Settings, FString& Json, FString& Error)
{
    if (Settings.Parameters.IsEmpty() || Settings.Objectives.IsEmpty())
    {
        Error = TEXT("Configure at least one parameter and one objective.");
        return false;
    }
    if (Settings.SamplingIterations < 0 || Settings.OptimizationIterations < 0 ||
        (!Settings.bWarmStart && Settings.SamplingIterations < 1) ||
        Settings.NumRestarts < 1 || Settings.RawSamples < Settings.NumRestarts || Settings.McSamples < 1)
    {
        Error = TEXT("Invalid iteration budget or model settings. RawSamples must be at least NumRestarts.");
        return false;
    }
    if ((Settings.Backend == EBoBackend::DynamicBO && Settings.Objectives.Num() != 1) ||
        (Settings.Backend == EBoBackend::MetaTAF && Settings.Objectives.Num() < 2))
    {
        Error = TEXT("DynamicBO requires one objective. MetaTAF requires at least two objectives.");
        return false;
    }
    if (Settings.bWarmStart &&
        (Settings.InitialParametersDataPath.IsEmpty() || Settings.InitialObjectivesDataPath.IsEmpty()))
    {
        Error = TEXT("Warm start requires parameter and objective CSV paths.");
        return false;
    }
    if (Settings.bWarmStart && Settings.Backend == EBoBackend::MetaTAF)
    {
        Error = TEXT("MetaTAF uses its source library and does not support this warm start mode.");
        return false;
    }
    const TSet<FString> Formats = {TEXT("auto"), TEXT("raw"), TEXT("normalized_max"), TEXT("normalized_native")};
    if (!Formats.Contains(Settings.WarmStartObjectiveFormat))
    {
        Error = TEXT("Invalid WarmStartObjectiveFormat.");
        return false;
    }

    TSharedPtr<FJsonObject> Advanced;
    const FString AdvancedText = Settings.AdvancedInitJson.TrimStartAndEnd().IsEmpty() ? TEXT("{}") : Settings.AdvancedInitJson;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(AdvancedText), Advanced) || !Advanced.IsValid())
    {
        Error = TEXT("AdvancedInitJson must be a JSON object.");
        return false;
    }
    const TSet<FString> AllowedRoot = {TEXT("config"), TEXT("context"), TEXT("cabopGroupCosts")};
    if (Advanced->HasField(TEXT("context")))
    {
        const TSharedPtr<FJsonObject>* Context = nullptr;
        if (!Advanced->TryGetObjectField(TEXT("context"), Context))
        {
            Error = TEXT("AdvancedInitJson.context must be an object.");
            return false;
        }
        bool bContextEnabled = false;
        if ((*Context)->HasField(TEXT("enabled")) && !(*Context)->TryGetBoolField(TEXT("enabled"), bContextEnabled))
        {
            Error = TEXT("context.enabled must be a boolean.");
            return false;
        }
        if (bContextEnabled && Settings.Backend != EBoBackend::BoTorch)
        {
            Error = TEXT("Contextual optimization requires the BoTorch backend.");
            return false;
        }
    }
    for (const auto& Field : Advanced->Values)
    {
        const FString FieldKey(*Field.Key);
        if (!AllowedRoot.Contains(FieldKey))
        {
            Error = FString::Printf(TEXT("Unsupported AdvancedInitJson root field: %s"), *FieldKey);
            return false;
        }
    }
    const TSet<FString> ProtectedConfig = {
        TEXT("nParameters"), TEXT("nObjectives"), TEXT("batchSize"), TEXT("numSamplingIterations"),
        TEXT("numOptimizationIterations"), TEXT("numRestarts"), TEXT("rawSamples"), TEXT("mcSamples"),
        TEXT("seed"), TEXT("warmStart"), TEXT("initialParametersDataPath"), TEXT("initialObjectivesDataPath"),
        TEXT("warmStartObjectiveFormat"), TEXT("optimizerBackend"), TEXT("cabopObjectiveMode")
    };
    TSharedRef<FJsonObject> Config = MakeShared<FJsonObject>();
    if (Advanced->HasField(TEXT("config")))
    {
        const TSharedPtr<FJsonObject>* AdvancedConfig = nullptr;
        if (!Advanced->TryGetObjectField(TEXT("config"), AdvancedConfig))
        {
            Error = TEXT("AdvancedInitJson.config must be an object.");
            return false;
        }
        for (const auto& Field : (*AdvancedConfig)->Values)
        {
            const FString FieldKey(*Field.Key);
            if (ProtectedConfig.Contains(FieldKey))
            {
                Error = FString::Printf(TEXT("Set the typed study setting for config field: %s"), *FieldKey);
                return false;
            }
            Config->SetField(FieldKey, Field.Value);
        }
    }
    Config->SetNumberField(TEXT("nParameters"), Settings.Parameters.Num());
    Config->SetNumberField(TEXT("nObjectives"), Settings.Objectives.Num());
    Config->SetNumberField(TEXT("batchSize"), 1);
    Config->SetNumberField(TEXT("numSamplingIterations"), Settings.SamplingIterations);
    Config->SetNumberField(TEXT("numOptimizationIterations"), Settings.OptimizationIterations);
    Config->SetNumberField(TEXT("numRestarts"), Settings.NumRestarts);
    Config->SetNumberField(TEXT("rawSamples"), Settings.RawSamples);
    Config->SetNumberField(TEXT("mcSamples"), Settings.McSamples);
    Config->SetNumberField(TEXT("seed"), Settings.Seed);
    Config->SetBoolField(TEXT("warmStart"), Settings.bWarmStart);
    Config->SetStringField(TEXT("initialParametersDataPath"), Settings.InitialParametersDataPath);
    Config->SetStringField(TEXT("initialObjectivesDataPath"), Settings.InitialObjectivesDataPath);
    Config->SetStringField(TEXT("warmStartObjectiveFormat"), Settings.WarmStartObjectiveFormat);
    Config->SetStringField(TEXT("optimizerBackend"), BackendName(Settings.Backend));
    Config->SetStringField(TEXT("cabopObjectiveMode"), Settings.Objectives.Num() == 1 ? TEXT("single") : TEXT("multi"));

    TSet<FString> Used;
    TArray<TSharedPtr<FJsonValue>> Parameters;
    for (int32 Index = 0; Index < Settings.Parameters.Num(); ++Index)
    {
        const FBoParameter& P = Settings.Parameters[Index];
        if (!ValidateKey(P.Key, Used, Error)) return false;
        if (!FMath::IsFinite(P.Low) || !FMath::IsFinite(P.High) || !FMath::IsFinite(P.High - P.Low) || P.Low >= P.High ||
            !FMath::IsFinite(P.Tolerance) || P.Tolerance < 0 || P.Tolerance > 1)
        {
            Error = FString::Printf(TEXT("Invalid bounds or CABOP tolerance for parameter: %s"), *P.Key);
            return false;
        }
        TArray<TSharedPtr<FJsonValue>> Prefabs;
        for (double Value : P.PrefabricatedValues)
        {
            if (!FMath::IsFinite(Value) || Value < P.Low || Value > P.High)
            {
                Error = FString::Printf(TEXT("Invalid prefabricated value for parameter: %s"), *P.Key);
                return false;
            }
            Prefabs.Add(MakeShared<FJsonValueNumber>(Value));
        }
        const TSharedRef<FJsonObject> PObject = MakeShared<FJsonObject>();
        PObject->SetStringField(TEXT("key"), P.Key);
        PObject->SetObjectField(TEXT("init"), Bounds(P.Low, P.High));
        PObject->SetNumberField(TEXT("optSeqOrder"), Index);
        PObject->SetStringField(TEXT("group"), P.Group);
        PObject->SetNumberField(TEXT("tolerance"), P.Tolerance);
        PObject->SetArrayField(TEXT("prefabValues"), Prefabs);
        Parameters.Add(MakeShared<FJsonValueObject>(PObject));
    }
    TArray<TSharedPtr<FJsonValue>> Objectives;
    for (int32 Index = 0; Index < Settings.Objectives.Num(); ++Index)
    {
        const FBoObjective& O = Settings.Objectives[Index];
        if (!ValidateKey(O.Key, Used, Error)) return false;
        if (!FMath::IsFinite(O.Low) || !FMath::IsFinite(O.High) || !FMath::IsFinite(O.High - O.Low) || O.Low >= O.High ||
            !FMath::IsFinite(O.Weight) || O.Weight <= 0)
        {
            Error = FString::Printf(TEXT("Invalid bounds or weight for objective: %s"), *O.Key);
            return false;
        }
        const TSharedRef<FJsonObject> Init = Bounds(O.Low, O.High);
        Init->SetNumberField(TEXT("minimize"), O.bMinimize ? 1 : 0);
        const TSharedRef<FJsonObject> OObject = MakeShared<FJsonObject>();
        OObject->SetStringField(TEXT("key"), O.Key);
        OObject->SetObjectField(TEXT("init"), Init);
        OObject->SetNumberField(TEXT("optSeqOrder"), Index);
        OObject->SetNumberField(TEXT("weight"), O.Weight);
        Objectives.Add(MakeShared<FJsonValueObject>(OObject));
    }
    const TSharedRef<FJsonObject> User = MakeShared<FJsonObject>();
    User->SetStringField(TEXT("userId"), Settings.UserId);
    User->SetStringField(TEXT("conditionId"), Settings.ConditionId);
    User->SetStringField(TEXT("groupId"), Settings.GroupId);
    Advanced->RemoveField(TEXT("config"));
    Advanced->SetStringField(TEXT("type"), TEXT("init"));
    Advanced->SetObjectField(TEXT("config"), Config);
    Advanced->SetArrayField(TEXT("parameters"), Parameters);
    Advanced->SetArrayField(TEXT("objectives"), Objectives);
    Advanced->SetObjectField(TEXT("user"), User);
    Json = Serialize(Advanced.ToSharedRef());
    if (FTCHARToUTF8(*Json).Length() > MaxMessageBytes)
    {
        Error = TEXT("Init message exceeds the 64 MiB protocol limit.");
        return false;
    }
    return true;
}

bool BoProtocol::BuildObjectives(const FBoStudySettings& Settings, const TMap<FString, double>& Values, FString& Json, FString& Error)
{
    if (Values.Num() != Settings.Objectives.Num())
    {
        Error = TEXT("Supply exactly one value for every configured objective.");
        return false;
    }
    const TSharedRef<FJsonObject> Payload = MakeShared<FJsonObject>();
    for (const FBoObjective& Objective : Settings.Objectives)
    {
        const double* Value = Values.Find(Objective.Key);
        if (!Value || !FMath::IsFinite(*Value) || *Value < Objective.Low || *Value > Objective.High)
        {
            Error = FString::Printf(TEXT("Objective is missing, nonfinite, or outside its configured bounds: %s"), *Objective.Key);
            return false;
        }
        Payload->SetNumberField(Objective.Key, *Value);
    }
    const TSharedRef<FJsonObject> Message = MakeShared<FJsonObject>();
    Message->SetStringField(TEXT("type"), TEXT("objectives"));
    Message->SetObjectField(TEXT("values"), Payload);
    Json = Serialize(Message);
    return true;
}

bool BoProtocol::ReadSuggestion(const FBoStudySettings& Settings, const TSharedPtr<FJsonObject>& Object, FBoSuggestion& Suggestion, FString& Error)
{
    const TSharedPtr<FJsonObject>* Values = nullptr;
    if (!Object->TryGetObjectField(TEXT("values"), Values) || (*Values)->Values.Num() != Settings.Parameters.Num())
    {
        Error = TEXT("Backend returned an incomplete or unexpected parameter set.");
        return false;
    }
    FBoSuggestion Candidate;
    double Iteration = 0;
    if (!Object->TryGetNumberField(TEXT("iteration"), Iteration) || !FMath::IsFinite(Iteration) ||
        Iteration < 0 || Iteration > MAX_int32 || Iteration != FMath::FloorToDouble(Iteration))
    {
        Error = TEXT("Backend returned an invalid iteration number.");
        return false;
    }
    Candidate.Iteration = static_cast<int32>(Iteration);
    for (const FBoParameter& Parameter : Settings.Parameters)
    {
        double Value = 0;
        if (!(*Values)->TryGetNumberField(Parameter.Key, Value) || !FMath::IsFinite(Value) ||
            Value < Parameter.Low - 1e-9 || Value > Parameter.High + 1e-9)
        {
            Error = FString::Printf(TEXT("Backend parameter is missing, nonfinite, or outside its bounds: %s"), *Parameter.Key);
            return false;
        }
        Candidate.Values.Add(Parameter.Key, FMath::Clamp(Value, Parameter.Low, Parameter.High));
    }
    Suggestion = MoveTemp(Candidate);
    return true;
}
