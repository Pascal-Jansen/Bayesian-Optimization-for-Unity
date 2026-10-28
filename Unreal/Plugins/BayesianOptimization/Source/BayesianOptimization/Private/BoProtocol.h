#pragma once

#include "CoreMinimal.h"
#include "BoStudySettings.h"
#include "Dom/JsonObject.h"

namespace BoProtocol
{
    constexpr int32 Port = 56001;
    constexpr int32 MaxMessageBytes = 64 * 1024 * 1024;

    bool BuildInit(const FBoStudySettings& Settings, FString& Json, FString& Error);
    bool BuildObjectives(const FBoStudySettings& Settings, const TMap<FString, double>& Values, FString& Json, FString& Error);
    bool ReadSuggestion(const FBoStudySettings& Settings, const TSharedPtr<FJsonObject>& Object, FBoSuggestion& Suggestion, FString& Error);
    FString Serialize(const TSharedRef<FJsonObject>& Object);
}
