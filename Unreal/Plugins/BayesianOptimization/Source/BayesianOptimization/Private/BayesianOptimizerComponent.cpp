#include "BayesianOptimizerComponent.h"
#include "BoProtocol.h"
#include "BoTransport.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

void FBoTransportDeleter::operator()(FBoTransport* Transport) const { delete Transport; }

UBayesianOptimizerComponent::UBayesianOptimizerComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;
    PrimaryComponentTick.bTickEvenWhenPaused = true;
}

UBayesianOptimizerComponent::~UBayesianOptimizerComponent() = default;

void UBayesianOptimizerComponent::SetState(EBoSessionState NewState)
{
    if (State == NewState) return;
    State = NewState;
    OnStateChanged.Broadcast(State);
}

void UBayesianOptimizerComponent::ReportInputError(const FString& Message)
{
    LastError = Message;
    OnError.Broadcast(Message);
}

void UBayesianOptimizerComponent::Fail(const FString& Message)
{
    if (Transport) Transport->Stop();
    LastError = Message;
    const uint64 Generation = SessionGeneration;
    SetState(EBoSessionState::Failed);
    if (SessionGeneration == Generation) OnError.Broadcast(Message);
}

bool UBayesianOptimizerComponent::StartStudy()
{
    check(IsInGameThread());
    if (bEndingPlay) return false;
    if (State == EBoSessionState::Connecting || State == EBoSessionState::WaitingForDesign ||
        State == EBoSessionState::Evaluating || State == EBoSessionState::Stopping)
    {
        ReportInputError(TEXT("A study is already active. Stop or disconnect before starting another study."));
        return false;
    }
    FString Json;
    FString Error;
    if (ServerPort < 1 || ServerPort > 65535)
    {
        Fail(TEXT("ServerPort must lie between 1 and 65535."));
        return false;
    }
    if (!FMath::IsFinite(ConnectionTimeoutSeconds) || ConnectionTimeoutSeconds <= 0 ||
        !FMath::IsFinite(IdleTimeoutSeconds) || IdleTimeoutSeconds <= 0)
    {
        Fail(TEXT("Connection and idle timeouts must be finite and positive."));
        return false;
    }
    if (!BoProtocol::BuildInit(Settings, Json, Error))
    {
        Fail(Error);
        return false;
    }
    Transport.Reset();
    ++SessionGeneration;
    ActiveSettings = Settings;
    LastSuggestion = FBoSuggestion();
    LastError.Reset();
    Transport.Reset(new FBoTransport(ConnectionTimeoutSeconds, IdleTimeoutSeconds, ServerPort));
    if (!Transport->Start(Json))
    {
        Transport.Reset();
        Fail(TEXT("Cannot start the optimizer socket worker."));
        return false;
    }
    SetState(EBoSessionState::Connecting);
    return true;
}

bool UBayesianOptimizerComponent::SubmitObjectives(const TMap<FString, double>& Values)
{
    check(IsInGameThread());
    if (State != EBoSessionState::Evaluating || !Transport)
    {
        ReportInputError(TEXT("SubmitObjectives requires an active design awaiting evaluation."));
        return false;
    }
    FString Json;
    FString Error;
    if (!BoProtocol::BuildObjectives(ActiveSettings, Values, Json, Error))
    {
        // Keep the trial active so instrumentation can correct an invalid value.
        ReportInputError(Error);
        return false;
    }
    Transport->Send(Json);
    SetState(EBoSessionState::WaitingForDesign);
    return true;
}

void UBayesianOptimizerComponent::StopStudy(const FString& Reason)
{
    check(IsInGameThread());
    if (!Transport || State == EBoSessionState::Completed || State == EBoSessionState::Failed ||
        State == EBoSessionState::Disconnected || State == EBoSessionState::Stopping) return;
    const TSharedRef<FJsonObject> Message = MakeShared<FJsonObject>();
    Message->SetStringField(TEXT("type"), TEXT("stop"));
    Message->SetStringField(TEXT("reason"), Reason);
    Transport->Send(BoProtocol::Serialize(Message));
    Transport->CloseAfterFlush();
    SetState(EBoSessionState::Stopping);
}

void UBayesianOptimizerComponent::Disconnect()
{
    check(IsInGameThread());
    ++SessionGeneration;
    Transport.Reset();
    SetState(EBoSessionState::Disconnected);
}

void UBayesianOptimizerComponent::BeginPlay()
{
    bEndingPlay = false;
    Super::BeginPlay();
}

void UBayesianOptimizerComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    bEndingPlay = true;
    Disconnect();
    Super::EndPlay(EndPlayReason);
}

void UBayesianOptimizerComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    FBoTransportEvent Event;
    // A callback can disconnect or restart. Stop draining when transport ownership changes.
    const uint64 Generation = SessionGeneration;
    while (Transport && SessionGeneration == Generation && Transport->Poll(Event))
    {
        switch (Event.Type)
        {
        case EBoTransportEvent::Connected:
            if (State == EBoSessionState::Connecting) SetState(EBoSessionState::WaitingForDesign);
            break;
        case EBoTransportEvent::Line:
            if (State != EBoSessionState::Stopping && State != EBoSessionState::Failed)
                HandleMessage(Event.Text);
            break;
        case EBoTransportEvent::Error:
            if (State != EBoSessionState::Stopping && State != EBoSessionState::Completed && State != EBoSessionState::Failed)
                Fail(Event.Text);
            break;
        case EBoTransportEvent::Closed:
            if (State == EBoSessionState::Stopping) SetState(EBoSessionState::Disconnected);
            else if (State != EBoSessionState::Completed && State != EBoSessionState::Failed)
                Fail(TEXT("Backend disconnected before optimization_finished. Inspect the Python console and run logs."));
            break;
        }
    }
}

void UBayesianOptimizerComponent::HandleMessage(const FString& Message)
{
    TSharedPtr<FJsonObject> Object;
    FString Type;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Message), Object) ||
        !Object.IsValid() || !Object->TryGetStringField(TEXT("type"), Type))
    {
        Fail(TEXT("Backend sent malformed JSON or a message without a type."));
        return;
    }
    if (Type == TEXT("parameters"))
    {
        if (State != EBoSessionState::WaitingForDesign)
        {
            Fail(TEXT("Backend sent a design while another trial is awaiting objectives."));
            return;
        }
        FBoSuggestion Suggestion;
        FString Error;
        if (!BoProtocol::ReadSuggestion(ActiveSettings, Object, Suggestion, Error))
        {
            Fail(Error);
            return;
        }
        Suggestion.MessageJson = Message;
        LastSuggestion = MoveTemp(Suggestion);
        const FBoSuggestion DeliveredSuggestion = LastSuggestion;
        const uint64 Generation = SessionGeneration;
        SetState(EBoSessionState::Evaluating);
        if (SessionGeneration == Generation && State == EBoSessionState::Evaluating)
            OnDesignSuggested.Broadcast(DeliveredSuggestion);
    }
    else if (Type == TEXT("coverage") || Type == TEXT("tempCoverage"))
    {
        double Value = 0;
        if (!Object->TryGetNumberField(TEXT("value"), Value) || !FMath::IsFinite(Value))
        {
            Fail(TEXT("Backend sent an invalid metric value."));
            return;
        }
        OnMetric.Broadcast(Type, Value);
    }
    else if (Type == TEXT("optimization_finished"))
    {
        if (State == EBoSessionState::Evaluating)
        {
            Fail(TEXT("Backend finished while a design is awaiting evaluation."));
            return;
        }
        if (Transport) Transport->CloseAfterFlush();
        const uint64 Generation = SessionGeneration;
        SetState(EBoSessionState::Completed);
        if (SessionGeneration == Generation && State == EBoSessionState::Completed) OnCompleted.Broadcast();
    }
    else
    {
        Fail(FString::Printf(TEXT("Unknown backend message type: %s"), *Type));
    }
}
