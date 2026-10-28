"""Create an isolated Unity project and validate example materials with GPU readback."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--pipeline", choices=("BuiltIn", "Universal", "HighDefinition"), required=True)
    parser.add_argument("--both-packages", action="store_true")
    parser.add_argument("--full-package", action="store_true", help="Also compile the QT-free optimizer and generate its analytic sample")
    parser.add_argument("--project", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = args.project or Path(tempfile.mkdtemp(prefix="bo-render-" + args.pipeline.lower() + "-"))
    project = project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    if any(project.iterdir()):
        parser.error("Validation requires an empty project directory.")
    editor = project / "Assets" / "Editor"
    editor.mkdir(parents=True)
    source = root / "Unity" / "Assets" / "iBO" / "Editor" / "Rendering" / "BoDemoRendering.cs"
    if not source.is_file():
        source = root / "Assets" / "iBO" / "Editor" / "Rendering" / "BoDemoRendering.cs"
    paths = source.parents[1] / "BoEditorPaths.cs"
    shutil.copyfile(source, editor / source.name)
    shutil.copyfile(paths, editor / paths.name)
    shutil.copyfile(Path(__file__).with_name("BoRenderingValidation.cs"), editor / "BoRenderingValidation.cs")
    dependencies = {}
    defines = []
    if args.full_package:
        defines.append("BO_VALIDATION_FULL")
        (editor / source.name).unlink()
        (editor / paths.name).unlink()
        asset_root = source.parents[2]
        for folder in ("Scripts", "Shared", "Examples", "Editor", "Tests", "Samples"):
            shutil.copytree(asset_root / folder, project / "Assets" / "iBO" / folder)
        dependencies.update({"com.unity.nuget.newtonsoft-json": "3.2.2", "com.unity.ugui": "2.6.0", "com.unity.test-framework": "1.8.0"})
        dependencies.update({"com.unity.modules." + module: "1.0.0" for module in ("audio", "physics", "physics2d", "imgui", "animation", "jsonserialize")})
    for name, package, define in (
        ("Universal", "com.unity.render-pipelines.universal", "BO_VALIDATION_URP"),
        ("HighDefinition", "com.unity.render-pipelines.high-definition", "BO_VALIDATION_HDRP"),
    ):
        if args.pipeline == name or args.both_packages:
            dependencies[package] = "17.6.0"
            defines.append(define)
    (project / "Packages").mkdir()
    (project / "Packages" / "manifest.json").write_text(json.dumps({"dependencies": dependencies}, indent=2), encoding="utf-8")
    (project / "Assets" / "csc.rsp").write_text("-define:" + ";".join(defines) if defines else "", encoding="utf-8")
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings" / "ProjectVersion.txt").write_text("m_EditorVersion: 6000.6.7f1\n", encoding="utf-8")
    print("Validation project:", project, flush=True)
    command = [str(args.unity), "-batchmode", "-projectPath", str(project), "-executeMethod", "BoRenderingValidation.Run", "-logFile", str(project / "unity.log")]
    result = subprocess.run(command, env=dict(os.environ, BO_RENDER_PIPELINE=args.pipeline), check=False)
    print("Unity exit:", result.returncode, flush=True)
    if (project / "render-validation.json").is_file():
        print((project / "render-validation.json").read_text(encoding="utf-8"), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
