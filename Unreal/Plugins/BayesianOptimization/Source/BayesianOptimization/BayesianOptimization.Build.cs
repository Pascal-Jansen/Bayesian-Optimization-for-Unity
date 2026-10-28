using UnrealBuildTool;

public class BayesianOptimization : ModuleRules
{
    public BayesianOptimization(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine" });
        PrivateDependencyModuleNames.AddRange(new[] { "Sockets", "Json" });
    }
}
