"""Launch the copied optimization backend with writable, engine-independent data paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def backend_entrypoint(backend: str, objectives: int) -> str:
    if objectives < 1:
        raise ValueError("objectives must be at least 1")
    if backend == "botorch":
        return "bo.py" if objectives == 1 else "mobo.py"
    if backend == "cabop":
        return "cabop_bo.py" if objectives == 1 else "cabop_mobo.py"
    if backend == "dbo":
        if objectives != 1:
            raise ValueError("DBO requires exactly one objective")
        return "dbo.py"
    if backend == "meta-taf":
        if objectives < 2:
            raise ValueError("MetaTAF requires at least two objectives")
        return "meta_mobo.py"
    raise ValueError(f"Unknown backend: {backend}")


def validate_init_identity(init: dict, backend: str, objectives: int) -> dict:
    config = init.get("config") or {}
    if config.get("optimizerBackend") != backend:
        raise ValueError(
            f"Launcher backend {backend!r} differs from Unreal optimizerBackend {config.get('optimizerBackend')!r}. "
            "Restart the launcher with the component's selected backend."
        )
    if config.get("nObjectives") != objectives:
        raise ValueError(
            f"Launcher expects {objectives} objectives; Unreal configured {config.get('nObjectives')!r}. "
            "Restart the launcher with the component's objective count."
        )
    return init


def validate_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    return port


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("botorch", "cabop", "dbo", "meta-taf"), default="botorch")
    parser.add_argument("--objectives", type=int, default=1)
    parser.add_argument("--port", type=int, default=56001, help="Loopback TCP port; match the Unreal component Server Port")
    parser.add_argument("--log-root", type=Path, default=Path.home() / "BayesianOptimizationRuns")
    parser.add_argument("--init-root", type=Path, default=root / "InitData")
    parser.add_argument("--model-library", choices=("botorch", "botorch-rust"), default="botorch")
    args = parser.parse_args(argv)
    try:
        entrypoint = backend_entrypoint(args.backend, args.objectives)
        validate_port(args.port)
    except ValueError as error:
        parser.error(str(error))
    backend_dir = root / "Backend"
    entry = backend_dir / entrypoint
    if not entry.is_file():
        parser.error(f"Missing backend source: {entry}. Run tools/refresh_backend.py from the source repository.")
    log_root = args.log_root.expanduser().resolve()
    init_root = args.init_root.expanduser().resolve()
    if not init_root.is_dir():
        parser.error(f"Init root does not exist: {init_root}")
    log_root.mkdir(parents=True, exist_ok=True)
    os.environ["BO_LOG_ROOT"] = str(log_root)
    os.environ["BO_INIT_ROOT"] = str(init_root)
    os.environ["BO_MODEL_LIBRARY"] = args.model_library
    os.environ["BO_PORT"] = str(args.port)
    os.environ.setdefault("BO_TORCH_THREADS", "1")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(backend_dir))
    import bo_protocol
    receive_init = bo_protocol.receive_init_message
    previous_port = bo_protocol.PORT
    bo_protocol.PORT = args.port

    def receive_checked_init(connection, timeout):
        return validate_init_identity(receive_init(connection, timeout), args.backend, args.objectives)

    bo_protocol.receive_init_message = receive_checked_init
    # Several backend-relative paths use the current working directory.
    os.chdir(root)
    print(f"Backend: {entrypoint}; objectives: {args.objectives}; logs: {log_root}", flush=True)
    print(f"Connect Unreal to 127.0.0.1:{args.port} after the server readiness message.", flush=True)
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        bo_protocol.receive_init_message = receive_init
        bo_protocol.PORT = previous_port


if __name__ == "__main__":
    main()
