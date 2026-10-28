"""Build independently installable engine packages from the common source repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile


# The QuestionnaireToolkit add-on lives in the Unity project but ships separately from the core packages.
ADDON_FOLDERS = {"iBO-QuestionnaireToolkit"}
EXCLUDED_PARTS = {".git", ".DS_Store", ".idea", ".vs", ".vscode", "Library", "Temp", "Logs", "UserSettings", "Binaries", "Intermediate", "Saved", "DerivedDataCache", "__pycache__", ".venv", "LogData", "wheels", "ContextEmbeddingCache"} | ADDON_FOLDERS
FORBIDDEN_PARTS = {"QuestionnaireToolkit", "TextMesh Pro", "Installation_Objects", "wheels", "LogData"}
FORBIDDEN_SUFFIXES = {".exe", ".dll", ".so", ".dylib", ".pkg", ".msi", ".pyc", ".pyo"}


def tracked_files(folder: Path) -> set[Path] | None:
    """Files Git tracks below folder, or None outside a Git work tree (then every file is a candidate)."""
    try:
        listing = subprocess.run(["git", "ls-files", "-z"], cwd=folder, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {folder / name for name in listing.decode("utf-8").split("\0") if name}


def distribution_files(folder: Path) -> list[Path]:
    # In a checkout only tracked files ship: local logs, IDE projects and caches never reach a release.
    tracked = tracked_files(folder)
    files = []
    for path in folder.rglob("*"):
        relative = path.relative_to(folder)
        if not path.is_file() or EXCLUDED_PARTS.intersection(relative.parts) or (path.suffix == ".meta" and path.stem in EXCLUDED_PARTS):
            continue
        if tracked is not None and path not in tracked:
            continue
        if FORBIDDEN_PARTS.intersection(relative.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Unexpected distribution content: {relative}")
        files.append(path)
    return sorted(files)


def archive_folder(folder: Path, output: Path) -> None:
    if folder.resolve() == output.resolve() or folder.resolve() in output.resolve().parents:
        raise ValueError("Archive output must be outside the directory being packaged")
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in distribution_files(folder):
            archive.write(path, (Path(folder.name) / path.relative_to(folder)).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Corrupt archive: {output}")


def inspect_unitypackage(package: Path) -> int:
    count = 0
    with tarfile.open(package, "r|gz") as archive:
        for item in archive:
            if item.isfile() and item.name.endswith("/pathname"):
                stream = archive.extractfile(item)
                name = stream.read().decode("utf-8").strip()
                relative = Path(name)
                if FORBIDDEN_PARTS.intersection(relative.parts) or relative.suffix.lower() in FORBIDDEN_SUFFIXES:
                    raise ValueError(f"Forbidden Unity package asset: {name}")
                count += 1
    if count == 0:
        raise ValueError("Unity export contains no assets")
    return count


def build(root: Path, output: Path, version: str, unity_editor: Path | None) -> dict:
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError("version must be a semantic version")
    for engine in ("Unity", "Unreal"):
        folder = (root / engine).resolve()
        if output.resolve() == folder or folder in output.resolve().parents:
            raise ValueError("Release output must be outside the engine directories")
    subprocess.run([sys.executable, "-B", str(root / "tools/sync_backends.py"), "--check"], cwd=root, check=True)
    unity_project = root / "Unity"
    if not (unity_project / "Assets").is_dir():
        raise ValueError("Run packaging from the clean common repository with Unity/Assets")
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    if unity_editor is not None:
        target = output / f"iBO-Unity-{version}.unitypackage"
        if target.exists():
            raise FileExistsError(target)
        environment = dict(os.environ, BO_RELEASE_DIRECTORY=str(output.resolve()), BO_RELEASE_VERSION=version)
        subprocess.run([
            str(unity_editor), "-batchmode", "-quit", "-projectPath", str(unity_project),
            "-executeMethod", "BayesianOptimization.Distribution.ReleaseExporter.Build",
            "-logFile", str(output / "unity-export.log"),
        ], env=environment, check=True)
        if not target.is_file():
            raise RuntimeError("Unity did not produce its export; inspect unity-export.log")
        inspect_unitypackage(target)
        outputs.append(target)
    for engine in ("Unity", "Unreal"):
        package = output / f"iBO-{engine}-{version}.zip"
        archive_folder(root / engine, package)
        outputs.append(package)
    checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs}
    manifest = {"version": version, "sha256": checksums}
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS.txt").write_text("".join(f"{value}  {name}\n" for name, value in checksums.items()), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--unity-editor", type=Path, help="Also produce a Unity .unitypackage using this Editor")
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.output.resolve(), args.version, args.unity_editor), indent=2))


if __name__ == "__main__":
    main()
