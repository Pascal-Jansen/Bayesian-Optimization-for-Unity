"""Generate and verify the Unity and Unreal Python distributions from Backend/."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath

ENGINES = ("unity", "unreal")
DATA_FOLDERS = ("InitData", "MetaSources")
# Written by the backends at run time (context_support.py), never part of a distribution.
RUNTIME_FOLDERS = ("__pycache__", "ContextEmbeddingCache")
UNITY_META = (
    "fileFormatVersion: 2\nguid: {guid}\n{folder}DefaultImporter:\n  externalObjects: {{}}\n  userData: \n"
    "  assetBundleName: \n  assetBundleVariant: \n"
)


def canonical_bytes(path: Path) -> bytes:
    """Canonical text bytes keep manifests stable across Git line ending settings."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def repository_root(repository: Path | None = None) -> Path:
    return (repository or Path(__file__).resolve().parents[1]).resolve()


def unity_data_root(repository: Path) -> Path:
    """Support the common repository and the original Unity authoring project."""
    assets = repository / "Unity" / "Assets"
    if not assets.is_dir():
        assets = repository / "Assets"
    return assets / "StreamingAssets" / "BOData"


def source_files(engine: str, repository: Path | None = None) -> tuple[Path, dict[Path, Path]]:
    """Return the distribution root and canonical-source to runtime-target mapping."""
    repository = repository_root(repository)
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine: {engine}")
    source = repository / "Backend"
    if not source.is_dir():
        raise RuntimeError("Backend/ is missing. Synchronization requires the common source repository.")
    destination = unity_data_root(repository) if engine == "unity" else repository / "Unreal"
    code_folder = "BayesianOptimization" if engine == "unity" else "Backend"
    mapping: dict[Path, Path] = {}
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if not item.is_file() or set(RUNTIME_FOLDERS) & set(relative.parts) or item.suffix in (".pyc", ".pyo", ".meta"):
            continue
        if relative.parts[0] in DATA_FOLDERS:
            mapping[item] = destination / relative
        elif relative == Path("requirements.txt"):
            suffix = Path("Installation/requirements.txt") if engine == "unity" else relative
            mapping[item] = destination / suffix
        elif item.suffix in (".py", ".md", ".txt", ".json") or item.name == "LICENSE":
            mapping[item] = destination / code_folder / relative
    if engine == "unreal":
        mapping[source / "LICENSE"] = destination / "LICENSE"
    return destination, mapping


def managed_target(destination: Path, name: str, engine: str) -> Path:
    windows = PureWindowsPath(name)
    if windows.drive or windows.root or "\\" in name or ".." in windows.parts:
        raise ValueError(f"Invalid generated path: {name}")
    target = (destination / name).resolve()
    if destination.resolve() not in target.parents:
        raise ValueError(f"Generated path leaves its distribution: {name}")
    code_folder = "BayesianOptimization" if engine == "unity" else "Backend"
    allowed = name.startswith((f"{code_folder}/", "InitData/", "MetaSources/"))
    allowed = allowed or name == ("Installation/requirements.txt" if engine == "unity" else "requirements.txt")
    allowed = allowed or (engine == "unreal" and name == "LICENSE")
    if not allowed:
        raise ValueError(f"Unmanaged generated path: {name}")
    return target


def unity_meta_paths(names) -> list[str]:
    """Every generated file and folder below BOData needs a .meta file, or Unity creates one on import."""
    paths = set()
    for name in names:
        parts = name.split("/")
        paths.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    return sorted(paths)


def write_unity_meta(destination: Path, name: str) -> None:
    """Create a missing .meta with a GUID derived from the path, so every checkout generates the same one."""
    target = destination / (name + ".meta")
    if target.exists():
        return
    guid = hashlib.md5(f"iBO/StreamingAssets/BOData/{name}".encode("utf-8")).hexdigest()
    folder = "folderAsset: yes\n" if (destination / name).is_dir() else ""
    target.write_text(UNITY_META.format(guid=guid, folder=folder), encoding="utf-8", newline="\n")


def manifest_path(engine: str, repository: Path | None = None) -> Path:
    repository = repository_root(repository)
    if engine == "unreal":
        return repository / "Unreal" / "backend-manifest.json"
    return repository / "tools" / "manifests" / "unity-backend.json"


def sync_engine(engine: str, check: bool = False, repository: Path | None = None) -> dict[str, str]:
    """Copy one engine mirror or verify it without modifying any files."""
    destination, mapping = source_files(engine, repository)
    manifest = {
        target.relative_to(destination).as_posix(): hashlib.sha256(canonical_bytes(original)).hexdigest()
        for original, target in sorted(mapping.items())
    }
    record = manifest_path(engine, repository)
    previous = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
    if not isinstance(previous, dict):
        raise ValueError(f"Expected an object in {record}")
    # Check all deletion and write targets before the first filesystem mutation.
    previous_targets = {name: managed_target(destination, name, engine) for name in previous}
    targets = {name: managed_target(destination, name, engine) for name in manifest}
    if check:
        errors = []
        if previous != manifest:
            errors.append(f"{record} differs from Backend/")
        for name, expected in manifest.items():
            target = targets[name]
            if not target.is_file() or hashlib.sha256(canonical_bytes(target)).hexdigest() != expected:
                errors.append(f"Generated {engine} file differs from Backend/: {name}")
        code_folder = "BayesianOptimization" if engine == "unity" else "Backend"
        for folder in (code_folder, *DATA_FOLDERS):
            for target in (destination / folder).rglob("*"):
                if target.is_file() and not set(RUNTIME_FOLDERS) & set(target.parts) and target.suffix not in (".pyc", ".pyo", ".meta"):
                    name = target.relative_to(destination).as_posix()
                    if name not in manifest:
                        errors.append(f"Unexpected generated {engine} file: {name}")
        if engine == "unity":
            for name in unity_meta_paths(manifest):
                if not (destination / (name + ".meta")).is_file():
                    errors.append(f"Missing Unity .meta file (run tools/sync_backends.py): {name}.meta")
            for folder in (code_folder, *DATA_FOLDERS):
                for metadata in (destination / folder).rglob("*.meta"):
                    if not set(RUNTIME_FOLDERS) & set(metadata.parts) and not metadata.with_suffix("").exists():
                        errors.append(f"Unity .meta file without its asset: {metadata.relative_to(destination).as_posix()}")
        if errors:
            raise RuntimeError("\n".join(errors))
        return manifest
    for name in sorted(set(previous) - set(manifest)):
        target = previous_targets[name]
        if target.is_file():
            target.unlink()
        if engine == "unity":
            metadata = managed_target(destination, name + ".meta", engine)
            if metadata.is_file():
                metadata.unlink()
        # A folder emptied by the removal goes too, with its Unity .meta.
        folder = target.parent
        while folder != destination and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            folder_meta = folder.with_name(folder.name + ".meta")
            if engine == "unity" and folder_meta.is_file():
                folder_meta.unlink()
            folder = folder.parent
    for original, target in sorted(mapping.items()):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_bytes(original))
    if engine == "unity":
        for name in unity_meta_paths(manifest):
            write_unity_meta(destination, name)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def sync(engine: str = "all", check: bool = False, repository: Path | None = None) -> dict[str, dict[str, str]]:
    engines = ENGINES if engine == "all" else (engine,)
    return {name: sync_engine(name, check=check, repository=repository) for name in engines}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=(*ENGINES, "all"), default="all")
    parser.add_argument("--check", action="store_true", help="Check generated source drift without writing files")
    parser.add_argument("--root", type=Path, help="Common repository root (defaults to this script's repository)")
    args = parser.parse_args()
    for engine, manifest in sync(args.engine, check=args.check, repository=args.root).items():
        print(f"{'Verified' if args.check else 'Generated'} {engine}: {len(manifest)} files.")


if __name__ == "__main__":
    main()
