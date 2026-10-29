"""Compatibility command for generating Unreal's copy from the common Backend/ source."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPOSITORY = Path(__file__).resolve().parents[2]
_TOOLS = _REPOSITORY / "tools"

def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _shared():
    if not (_TOOLS / "sync_backends.py").is_file():
        raise RuntimeError("Refresh requires the common source repository. The distribution runs without it.")
    sys.path.insert(0, str(_TOOLS))
    import sync_backends
    return sync_backends


def source_files():
    return _shared().source_files("unreal", _REPOSITORY)


def managed_target(unreal: Path, name: str) -> Path:
    return _shared().managed_target(unreal, name, "unreal")


def refresh(check: bool = False) -> dict[str, str]:
    return _shared().sync_engine("unreal", check=check, repository=_REPOSITORY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated source drift without writing files")
    args = parser.parse_args()
    print(f"{'Verified' if args.check else 'Generated'} {len(refresh(check=args.check))} files.")
