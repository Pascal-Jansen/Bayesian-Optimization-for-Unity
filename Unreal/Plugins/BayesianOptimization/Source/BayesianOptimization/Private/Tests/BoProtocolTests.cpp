#if WITH_DEV_AUTOMATION_TESTS
#include "BoProtocol.h"
#include "BayesianOptimizerComponent.h"
#include "Misc/AutomationTest.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include <limits>

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoProtocolTest, "BayesianOptimization.Protocol.Validation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FBoProtocolTest::RunTest(const FString& Parameters)
{
    UBayesianOptimizerComponent* Component = NewObject<UBayesianOptimizerComponent>();
    Component->ServerPort = 0;
    TestFalse(TEXT("Zero server port refused before connecting"), Component->StartStudy());
    Component->ServerPort = 65536;
    TestFalse(TEXT("Port overflow refused before connecting"), Component->StartStudy());
    FBoStudySettings Settings;
    FBoParameter Parameter;
    Parameter.Key = TEXT("size");
    Parameter.Low = 10;
    Parameter.High = 100;
    Settings.Parameters.Add(Parameter);
    FBoObjective Objective;
    Objective.Key = TEXT("duration");
    Objective.High = 60;
    Objective.bMinimize = true;
    Settings.Objectives.Add(Objective);
    FString Json, Error;
    TestTrue(TEXT("Valid settings"), BoProtocol::BuildInit(Settings, Json, Error));
    TSharedPtr<FJsonObject> Init;
    TestTrue(TEXT("Init JSON parses"), FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Init));
    if (!Init) return false;
    TestEqual(TEXT("Parameter count"), Init->GetObjectField(TEXT("config"))->GetIntegerField(TEXT("nParameters")), 1);
    TestEqual(TEXT("Minimization flag"), Init->GetArrayField(TEXT("objectives"))[0]->AsObject()->GetObjectField(TEXT("init"))->GetIntegerField(TEXT("minimize")), 1);
    FBoStudySettings Invalid = Settings;
    Invalid.Parameters[0].High = Invalid.Parameters[0].Low;
    TestFalse(TEXT("Degenerate bounds refused"), BoProtocol::BuildInit(Invalid, Json, Error));
    Invalid = Settings;
    Invalid.Parameters[0].Low = -std::numeric_limits<double>::max();
    Invalid.Parameters[0].High = std::numeric_limits<double>::max();
    TestFalse(TEXT("Overflowing span refused"), BoProtocol::BuildInit(Invalid, Json, Error));
    Invalid = Settings;
    Invalid.Objectives[0].Key = TEXT("SIZE");
    TestFalse(TEXT("Case-insensitive overlap refused"), BoProtocol::BuildInit(Invalid, Json, Error));
    Invalid = Settings;
    Invalid.Parameters[0].Key = TEXT("Iteration");
    TestFalse(TEXT("CSV column collision refused"), BoProtocol::BuildInit(Invalid, Json, Error));
    Invalid = Settings;
    Invalid.AdvancedInitJson = TEXT("{\"config\":{\"nParameters\":9}}");
    TestFalse(TEXT("Core config override refused"), BoProtocol::BuildInit(Invalid, Json, Error));
    Invalid.AdvancedInitJson = TEXT("{\"config\":{\"dboStationaryBaseline\":true}}");
    TestTrue(TEXT("Advanced settings accepted"), BoProtocol::BuildInit(Invalid, Json, Error));
    Invalid.AdvancedInitJson = TEXT("{\"context\":{\"enabled\":true}}");
    Invalid.Backend = EBoBackend::CABOP;
    TestFalse(TEXT("CABOP context refused"), BoProtocol::BuildInit(Invalid, Json, Error));
    Invalid.Backend = EBoBackend::DynamicBO;
    TestFalse(TEXT("DynamicBO context refused"), BoProtocol::BuildInit(Invalid, Json, Error));
    TMap<FString, double> Values;
    Values.Add(TEXT("duration"), 12.5);
    TestTrue(TEXT("Raw objective accepted"), BoProtocol::BuildObjectives(Settings, Values, Json, Error));
    TSharedPtr<FJsonObject> Message;
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Message);
    if (Message) TestEqual(TEXT("Original objective units preserved"), Message->GetObjectField(TEXT("values"))->GetNumberField(TEXT("duration")), 12.5);
    Values[TEXT("duration")] = 61;
    TestFalse(TEXT("Out of bounds refused"), BoProtocol::BuildObjectives(Settings, Values, Json, Error));
    Values[TEXT("duration")] = std::numeric_limits<double>::quiet_NaN();
    TestFalse(TEXT("NaN refused"), BoProtocol::BuildObjectives(Settings, Values, Json, Error));
    Values.Reset();
    TestFalse(TEXT("Missing objective refused"), BoProtocol::BuildObjectives(Settings, Values, Json, Error));
    TSharedPtr<FJsonObject> Incoming;
    const FString SuggestionJson = TEXT("{\"type\":\"parameters\",\"values\":{\"size\":25.5},\"iteration\":7}");
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(SuggestionJson), Incoming);
    if (!Incoming) return false;
    FBoSuggestion Suggestion;
    TestTrue(TEXT("Valid suggestion"), BoProtocol::ReadSuggestion(Settings, Incoming, Suggestion, Error));
    TestEqual(TEXT("Original parameter units preserved"), Suggestion.Values.FindRef(TEXT("size")), 25.5);
    TestEqual(TEXT("Log iteration preserved"), Suggestion.Iteration, 7);
    Incoming->SetNumberField(TEXT("iteration"), 7.5);
    TestFalse(TEXT("Fractional iteration refused"), BoProtocol::ReadSuggestion(Settings, Incoming, Suggestion, Error));
    Incoming->SetNumberField(TEXT("iteration"), 8);
    Incoming->GetObjectField(TEXT("values"))->SetNumberField(TEXT("extra"), 0);
    TestFalse(TEXT("Unexpected parameters refused"), BoProtocol::ReadSuggestion(Settings, Incoming, Suggestion, Error));
    return true;
}
#endif
