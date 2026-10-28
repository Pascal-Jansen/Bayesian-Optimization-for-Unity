"""Start the iBO server (ibo_server.py) as a child process, the way Unity's PythonStarter does.

The server needs a Python with the optimizer stack (torch, botorch, ...; Backend/requirements.txt). Pass that interpreter as ``python``; the client itself needs none of it.
"""

from __future__ import annotations

import collections
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Mapping, Optional, Union

READY_LINE = "Server starts, waiting for connection..."  # printed by bo_protocol before it accepts

PathLike = Union[str, os.PathLike]


def default_backend_dir() -> Path:
    """IBO_BACKEND_DIR; else Backend/ of the repository this package lives in (source checkout or
    editable install); else Backend/ in the current folder (repository root after `pip install ./Python`)."""
    configured = os.environ.get("IBO_BACKEND_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    # Python/interactive_bo/server.py -> repository root
    candidates = [Path(__file__).resolve().parents[2] / "Backend", Path.cwd() / "Backend"]
    for candidate in candidates:
        if (candidate / "ibo_server.py").is_file():
            return candidate
    return candidates[-1]


def data_dir_for(backend_dir: Path) -> Path:
    """Folder holding InitData and MetaSources: Backend/ itself, or the parent of an engine copy
    (Unity's StreamingAssets/BOData/BayesianOptimization, Unreal's Unreal/Backend)."""
    return backend_dir if (backend_dir / "InitData").is_dir() else backend_dir.parent


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerProcess:
    """A running iBO server. Use as a context manager, or call stop()."""

    def __init__(self, process: subprocess.Popen, port: int, log_root: Path, echo: bool = False):
        self.process = process
        self.echo = echo
        self.port = port
        self.log_root = log_root
        self.output = collections.deque(maxlen=200)  # last lines of the server's console output
        self._ready = threading.Event()
        self._seen_ready = False
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        for line in self.process.stdout:
            line = line.rstrip("\n")
            self.output.append(line)
            if self.echo:
                print(line, flush=True)
            if line.strip() == READY_LINE:
                self._seen_ready = True
                self._ready.set()
        self.process.stdout.close()
        self._ready.set()  # the process ended: wake wait_ready()

    def wait_ready(self, timeout: float) -> None:
        """Wait for the ready line; the event is also set when the process ends without it."""
        self._ready.wait(timeout)
        if not self._seen_ready:
            tail = "\n".join(list(self.output)[-30:])
            self.stop()
            raise RuntimeError(f"The iBO server did not start within {timeout:g} s "
                               f"(exit code {self.process.poll()}). Last output:\n{tail}")

    def wait(self, timeout: Optional[float] = None) -> int:
        code = self.process.wait(timeout)
        self._reader.join(timeout=5)
        return code

    def stop(self, timeout: float = 10.0) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._reader.join(timeout=5)

    def __enter__(self) -> "ServerProcess":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def launch_server(python: Optional[PathLike] = None, port: Optional[int] = None,
                  log_root: Optional[PathLike] = None, init_root: Optional[PathLike] = None,
                  backend_dir: Optional[PathLike] = None, env: Optional[Mapping[str, str]] = None,
                  ready_timeout: float = 120.0, echo: bool = False) -> ServerProcess:
    """Start ibo_server.py and wait until it accepts a connection.

    python:      interpreter with the optimizer stack (default: this one)
    port:        TCP port on 127.0.0.1 (default: a free one; Unity always uses 56001)
    log_root:    root of the run folders <userId>/<conditionId>/run/ (default: ./LogData)
    init_root:   warm-start and context data (default: the InitData folder of the backend)
    backend_dir: folder holding ibo_server.py (default: default_backend_dir())
    echo:        print the server's console output as it arrives
    """
    backend_dir = Path(backend_dir).resolve() if backend_dir else default_backend_dir()
    script = backend_dir / "ibo_server.py"
    if not script.is_file():
        raise FileNotFoundError(f"ibo_server.py not found in {backend_dir}: run from the repository root, "
                                "set IBO_BACKEND_DIR or pass backend_dir (--backend-dir).")
    port = int(port) if port else free_port()
    log_root = Path(log_root).resolve() if log_root else Path.cwd() / "LogData"
    data_dir = data_dir_for(backend_dir)
    init_root = Path(init_root).resolve() if init_root else data_dir / "InitData"
    log_root.mkdir(parents=True, exist_ok=True)

    child_env = dict(os.environ)
    child_env.update({
        "BO_PORT": str(port),
        "BO_LOG_ROOT": str(log_root),
        "BO_INIT_ROOT": str(init_root),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",  # torch and numpy may each load an OpenMP runtime
    })
    if env:
        child_env.update(env)

    process = subprocess.Popen(
        [str(python or sys.executable), "-u", str(script)],
        cwd=str(data_dir), env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    server = ServerProcess(process, port, log_root, echo=echo)
    server.wait_ready(ready_timeout)
    return server
