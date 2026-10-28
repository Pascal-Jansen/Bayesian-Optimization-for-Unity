#include "BoTransport.h"
#include "BoProtocol.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/RunnableThread.h"
#include "IPAddress.h"
#include "Sockets.h"
#include "SocketSubsystem.h"

FBoTransport::FBoTransport(double InConnectionTimeout, double InIdleTimeout, int32 InServerPort)
    : ConnectionTimeout(InConnectionTimeout), IdleTimeout(InIdleTimeout), ServerPort(InServerPort)
{
}

FBoTransport::~FBoTransport()
{
    Stop();
    if (Thread)
    {
        Thread->WaitForCompletion();
        delete Thread;
    }
}

bool FBoTransport::Start(const FString& InitJson)
{
    check(IsInGameThread());
    // Resolve the module before entering the worker. Module loading requires the game thread.
    Subsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!Subsystem) return false;
    Outgoing.Enqueue(InitJson);
    Thread = FRunnableThread::Create(this, TEXT("BayesianOptimizationIO"));
    return Thread != nullptr;
}

void FBoTransport::Send(const FString& Json) { Outgoing.Enqueue(Json); }
// A queue sentinel orders closure after every previously queued write.
void FBoTransport::CloseAfterFlush() { Outgoing.Enqueue(FString()); }
bool FBoTransport::Poll(FBoTransportEvent& Event) { return Incoming.Dequeue(Event); }
void FBoTransport::Stop() { bStop.store(true); }

void FBoTransport::Push(EBoTransportEvent Type, const FString& Text)
{
    FBoTransportEvent Event;
    Event.Type = Type;
    Event.Text = Text;
    Incoming.Enqueue(MoveTemp(Event));
}

void FBoTransport::CloseSocket()
{
    if (Socket)
    {
        Socket->Shutdown(ESocketShutdownMode::ReadWrite);
        Socket->Close();
        Subsystem->DestroySocket(Socket);
        Socket = nullptr;
    }
}

bool FBoTransport::Connect()
{
    if (!Subsystem)
    {
        Push(EBoTransportEvent::Error, TEXT("Platform socket subsystem is unavailable."));
        return false;
    }
    const TSharedRef<FInternetAddr> Address = Subsystem->CreateInternetAddr();
    bool bValid = false;
    Address->SetIp(TEXT("127.0.0.1"), bValid);
    Address->SetPort(ServerPort);
    const double Deadline = FPlatformTime::Seconds() + ConnectionTimeout;
    while (!bStop.load() && FPlatformTime::Seconds() < Deadline)
    {
        Socket = Subsystem->CreateSocket(NAME_Stream, TEXT("BayesianOptimization"), false);
        if (!Socket || !Socket->SetNonBlocking(true))
        {
            CloseSocket();
            Push(EBoTransportEvent::Error, TEXT("Cannot create a nonblocking TCP socket."));
            return false;
        }
        Socket->SetNoDelay(true);
        Socket->SetLinger(false, 0);
        // On Unreal's BSD implementation true also means the nonblocking connect is pending.
        const bool bConnectStarted = Socket->Connect(*Address);
        const ESocketErrors ConnectError = Subsystem->GetLastErrorCode();
        if (bConnectStarted || ConnectError == SE_EWOULDBLOCK || ConnectError == SE_EINPROGRESS || ConnectError == SE_EALREADY)
        {
            const double AttemptDeadline = FMath::Min(Deadline, FPlatformTime::Seconds() + 0.5);
            while (!bStop.load() && FPlatformTime::Seconds() < AttemptDeadline)
            {
                if (Socket->Wait(ESocketWaitConditions::WaitForWrite, FTimespan::FromMilliseconds(50)))
                {
                    if (Socket->GetConnectionState() == SCS_Connected) return true;
                    break;
                }
            }
        }
        CloseSocket();
        if (!bStop.load()) FPlatformProcess::SleepNoStats(0.05f);
    }
    if (!bStop.load())
        Push(EBoTransportEvent::Error, FString::Printf(TEXT("Connection timed out. Start the Python backend on 127.0.0.1:%d before StartStudy."), ServerPort));
    return false;
}

bool FBoTransport::SendLine(const FString& Json)
{
    const FString Framed = Json + TEXT("\n");
    const FTCHARToUTF8 Encoded(*Framed);
    if (Encoded.Length() - 1 > BoProtocol::MaxMessageBytes)
    {
        Push(EBoTransportEvent::Error, TEXT("Outgoing message exceeds the 64 MiB limit."));
        return false;
    }
    const uint8* Bytes = reinterpret_cast<const uint8*>(Encoded.Get());
    int32 Offset = 0;
    const double Deadline = FPlatformTime::Seconds() + ConnectionTimeout;
    while (!bStop.load() && Offset < Encoded.Length())
    {
        if (FPlatformTime::Seconds() >= Deadline)
        {
            Push(EBoTransportEvent::Error, TEXT("Sending a message timed out."));
            return false;
        }
        if (!Socket->Wait(ESocketWaitConditions::WaitForWrite, FTimespan::FromMilliseconds(50))) continue;
        int32 Sent = 0;
        if (!Socket->Send(Bytes + Offset, Encoded.Length() - Offset, Sent))
        {
            if (Subsystem->GetLastErrorCode() == SE_EWOULDBLOCK) continue;
            Push(EBoTransportEvent::Error, TEXT("The backend connection closed while sending."));
            return false;
        }
        Offset += Sent;
    }
    return Offset == Encoded.Length();
}

bool FBoTransport::FeedBytes(const uint8* Data, int32 Count)
{
    // Split bytes before decoding, so a UTF-8 code point may cross socket reads.
    int32 Start = 0;
    for (int32 Index = 0; Index < Count; ++Index)
    {
        if (Data[Index] != '\n') continue;
        if (PartialLine.Num() + Index - Start > BoProtocol::MaxMessageBytes)
        {
            Push(EBoTransportEvent::Error, TEXT("Incoming message exceeds the 64 MiB limit."));
            return false;
        }
        PartialLine.Append(Data + Start, Index - Start);
        if (PartialLine.Num() > 0 && PartialLine.Last() == '\r') PartialLine.Pop();
        if (!PartialLine.IsEmpty())
        {
            const FUTF8ToTCHAR Decoded(reinterpret_cast<const ANSICHAR*>(PartialLine.GetData()), PartialLine.Num());
            Push(EBoTransportEvent::Line, FString(Decoded.Length(), Decoded.Get()));
        }
        PartialLine.Reset();
        Start = Index + 1;
    }
    if (PartialLine.Num() + Count - Start > BoProtocol::MaxMessageBytes)
    {
        Push(EBoTransportEvent::Error, TEXT("Incoming message exceeds the 64 MiB limit."));
        return false;
    }
    PartialLine.Append(Data + Start, Count - Start);
    return true;
}

uint32 FBoTransport::Run()
{
    if (Connect())
    {
        Push(EBoTransportEvent::Connected);
        double LastActivity = FPlatformTime::Seconds();
        uint8 Buffer[65536];
        while (!bStop.load())
        {
            FString Json;
            bool bSendFailed = false;
            bool bGracefulClose = false;
            while (!bStop.load() && Outgoing.Dequeue(Json))
            {
                if (Json.IsEmpty()) { bGracefulClose = true; break; }
                if (!SendLine(Json)) { bSendFailed = true; break; }
                LastActivity = FPlatformTime::Seconds();
            }
            if (bSendFailed || bGracefulClose) break;
            if (FPlatformTime::Seconds() - LastActivity > IdleTimeout)
            {
                Push(EBoTransportEvent::Error, TEXT("The optimizer connection exceeded IdleTimeoutSeconds."));
                break;
            }
            if (!Socket->Wait(ESocketWaitConditions::WaitForRead, FTimespan::FromMilliseconds(50))) continue;
            int32 Read = 0;
            if (!Socket->Recv(Buffer, sizeof(Buffer), Read, ESocketReceiveFlags::None))
            {
                // Recv returns false on EOF. Its true/zero result means would-block.
                // Do not inspect a stale platform error after EOF.
                break;
            }
            if (Read == 0) continue;
            LastActivity = FPlatformTime::Seconds();
            if (!FeedBytes(Buffer, Read)) break;
        }
    }
    CloseSocket();
    Push(EBoTransportEvent::Closed);
    return 0;
}
