#pragma once

#include "CoreMinimal.h"
#include "Containers/Queue.h"
#include "HAL/Runnable.h"
#include <atomic>

class FSocket;
class FRunnableThread;
class ISocketSubsystem;

enum class EBoTransportEvent : uint8 { Connected, Line, Error, Closed };

struct FBoTransportEvent
{
    EBoTransportEvent Type = EBoTransportEvent::Line;
    FString Text;
};

// Owns the socket on one worker thread. Contains no UObject references.
class FBoTransport final : public FRunnable
{
public:
    FBoTransport(double InConnectionTimeout, double InIdleTimeout, int32 InServerPort = 56001);
    virtual ~FBoTransport() override;
    bool Start(const FString& InitJson);
    void Send(const FString& Json);
    void CloseAfterFlush();
    bool Poll(FBoTransportEvent& Event);

    virtual uint32 Run() override;
    virtual void Stop() override;

private:
    double ConnectionTimeout;
    double IdleTimeout;
    int32 ServerPort;
    FRunnableThread* Thread = nullptr;
    ISocketSubsystem* Subsystem = nullptr;
    FSocket* Socket = nullptr;
    std::atomic<bool> bStop{false};
    TQueue<FString, EQueueMode::Mpsc> Outgoing;
    TQueue<FBoTransportEvent, EQueueMode::Spsc> Incoming;
    TArray<uint8> PartialLine;

    bool Connect();
    bool SendLine(const FString& Json);
    bool FeedBytes(const uint8* Data, int32 Count);
    void Push(EBoTransportEvent Type, const FString& Text = FString());
    void CloseSocket();
};
