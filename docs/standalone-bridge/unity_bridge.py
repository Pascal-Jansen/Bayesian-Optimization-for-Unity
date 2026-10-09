"""TCP/JSON server exposing a :class:`~dbo_torch.optimizer.DynamicBO` to Unity.

Unity cannot host PyTorch, so the optimiser runs as a small local server and
the Unity scene talks to it over a socket. The protocol is newline-delimited
JSON: one request object per line, one response object per line. The matching
C# client is ``DboClient.cs`` in this folder.

Run it from this folder, with ``dbo_torch`` importable (a dbo-torch checkout or
installation, or BOforUnity's vendored copy: add
``Assets/StreamingAssets/BOData/BayesianOptimization`` to ``PYTHONPATH``)::

    python unity_bridge.py --host 127.0.0.1 --port 8756 --save-dir runs

Requests
--------
``{"cmd": "reset", "bounds": [[lo, hi], ...], ...}``
    Start a new run. Optional: ``seed_points``, ``exploration_ratio``,
    ``validation_every``, ``stationary``, ``seed``.
``{"cmd": "suggest"}``
    Next input to evaluate. Returns ``x``, ``iteration``, ``is_validation``.
``{"cmd": "suggest_validation"}``
    Current best estimate, without consuming an acquisition step.
``{"cmd": "observe", "x": [...], "y": 1.23}``
    Record a measured cost.
``{"cmd": "predict", "x": [...]}``
    Posterior ``mean`` and ``std`` at a point.
``{"cmd": "state"}``
    Full run history and fitted ``alpha``.
``{"cmd": "save", "path": "run.json"}``
    Write the history to ``path`` inside the server's save directory
    (``--save-dir``, default: the working directory the server was started
    in). Absolute paths and ``..`` components are refused, so a client cannot
    write anywhere else on the machine.
``{"cmd": "ping"}``
    Liveness check.

Every response carries ``ok``. On failure it carries ``ok: false`` and
``error``. The server keeps running after a failed request: a study should not
die because one iteration hit a numerical problem.

The server binds to 127.0.0.1 by default and performs no authentication. It is
intended for a lab machine, not a shared network. Binding to a non-loopback
address requires ``--allow-remote``, so it cannot happen by accident. Like
BOforUnity's backends it never shares its port: on Windows it binds with
``SO_EXCLUSIVEADDRUSE`` (``SO_REUSEADDR`` there would let a second server bind
the same port and receive the client's connections), elsewhere with
``SO_REUSEADDR``, which only permits an immediate restart while the previous
server's connections linger in ``TIME_WAIT``.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import socketserver
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # imported lazily at runtime, see _load_dbo()
    from dbo_torch.optimizer import DynamicBO

__all__ = ["DBOServer", "configure_listener_socket", "resolve_save_path", "serve", "main"]

log = logging.getLogger("dbo_torch.bridge")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8756


def _load_dbo():
    """Import dbo_torch on first use, so the transport works (and is testable) without torch."""
    from dbo_torch.model import DBOModelConfig
    from dbo_torch.optimizer import DBOConfig, DynamicBO

    return DBOModelConfig, DBOConfig, DynamicBO


def configure_listener_socket(sock: socket.socket) -> None:
    """Socket options for the listening socket (same policy as BOforUnity's backends).

    On Windows, SO_REUSEADDR lets a second process bind a port that is still being
    listened on, so a stale server could receive the client's connection;
    SO_EXCLUSIVEADDRUSE refuses that and still allows an immediate restart. On POSIX,
    SO_REUSEADDR only permits rebinding while an earlier connection lingers in TIME_WAIT.
    """
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def resolve_save_path(save_dir: Path, requested: Any) -> Path:
    """Map a client-supplied ``save`` path to a file inside ``save_dir``.

    The bridge is unauthenticated, so the client must not choose where on the machine
    the server writes: absolute paths, drive-qualified paths and ``..`` components are
    refused, and the resolved target (after following symlinks) must stay inside
    ``save_dir``.
    """
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("'save' requires 'path', a file name relative to the server's save directory")
    text = requested.strip()
    windows = PureWindowsPath(text)
    if PurePosixPath(text).is_absolute() or windows.drive or windows.root:
        raise ValueError(
            f"'save' path must be relative to the server's save directory, got {requested!r}"
        )
    parts = windows.parts  # splits on both '/' and '\\'
    if any(part == ".." for part in parts):
        raise ValueError(f"'save' path must not contain '..', got {requested!r}")

    root = Path(save_dir).resolve()
    target = root.joinpath(*parts).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"'save' path {requested!r} does not name a file inside the save directory")
    return target


class _Session:
    """Holds the optimiser, with a lock serialising request handling.

    The lock makes each individual request atomic; it does not arbitrate
    between multiple clients, which all share the single run. One client per
    run is the intended deployment."""

    def __init__(self, save_dir: Path | str | None = None) -> None:
        self.lock = threading.Lock()
        self.optimizer: DynamicBO | None = None
        self.save_dir = Path(save_dir) if save_dir is not None else Path.cwd()

    def reset(self, req: dict[str, Any]) -> dict[str, Any]:
        bounds = req.get("bounds")
        if not bounds:
            raise ValueError("'reset' requires 'bounds', e.g. [[-5, 9]]")

        DBOModelConfig, DBOConfig, DynamicBO = _load_dbo()
        parsed = [(float(lo), float(hi)) for lo, hi in bounds]

        model_cfg = DBOModelConfig(
            stationary=bool(req.get("stationary", False)),
        )
        if "normalize_inputs" in req:
            model_cfg.normalize_inputs = bool(req["normalize_inputs"])
        if "initial_alpha" in req:
            model_cfg.initial_alpha = float(req["initial_alpha"])

        cfg = DBOConfig(
            model=model_cfg,
            seed_points=[[float(v) for v in p] for p in req["seed_points"]]
            if req.get("seed_points")
            else None,
            exploration_ratio=float(req.get("exploration_ratio", 0.1)),
            validation_every=req.get("validation_every"),
            validation_confidence=float(req.get("validation_confidence", 0.01)),
            acquisition_time_offset=float(req.get("acquisition_time_offset", 0.0)),
            seed=req.get("seed"),
        )
        if cfg.validation_every is not None:
            cfg.validation_every = int(cfg.validation_every)

        self.optimizer = DynamicBO(bounds=parsed, config=cfg)
        log.info("reset: %d parameter(s), bounds=%s", len(parsed), parsed)
        return {"dim": len(parsed)}

    def require(self) -> DynamicBO:
        if self.optimizer is None:
            raise ValueError("No active run. Send 'reset' first.")
        return self.optimizer


def _handle(session: _Session, req: dict[str, Any]) -> dict[str, Any]:
    cmd = req.get("cmd")

    if cmd == "ping":
        return {"pong": True}

    if cmd == "reset":
        return session.reset(req)

    if cmd == "suggest":
        opt = session.require()
        iteration = opt.next_iteration
        is_val = opt.is_validation_iteration(iteration)
        x = opt.suggest()
        return {"x": x, "iteration": iteration, "is_validation": is_val}

    if cmd == "suggest_validation":
        opt = session.require()
        return {
            "x": opt.suggest_validation(),
            "iteration": opt.next_iteration,
            "is_validation": True,
        }

    if cmd == "observe":
        opt = session.require()
        if "x" not in req or "y" not in req:
            raise ValueError("'observe' requires 'x' and 'y'")
        obs = opt.observe(
            [float(v) for v in req["x"]],
            float(req["y"]),
            is_validation=req.get("is_validation"),
        )
        return {"iteration": obs.iteration, "alpha": opt.alpha}

    if cmd == "predict":
        opt = session.require()
        model = opt._ensure_model()
        if model is None:
            raise ValueError("Not enough observations yet to predict.")
        mean, std = opt._predict_at(model, [float(v) for v in req["x"]])
        return {"mean": mean, "std": std}

    if cmd == "state":
        opt = session.require()
        best = opt.best_observed()
        return {
            "iteration": opt.num_observations,
            "alpha": opt.alpha,
            "observations": opt.history(),
            "best": best.as_dict() if best else None,
            "prediction_error": opt.prediction_error(),
        }

    if cmd == "save":
        # Validate the path first: a refused path is reported even before 'reset'.
        target = resolve_save_path(session.save_dir, req.get("path"))
        opt = session.require()
        return {"path": str(opt.save(target))}

    raise ValueError(f"Unknown command: {cmd!r}")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        peer = self.client_address
        log.info("client connected: %s", peer)

        for raw in self.rfile:
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                self._reply({"ok": False, "error": f"Bad request: {exc}"})
                continue
            if not line:
                continue

            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("Each request must be a JSON object.")
            except (json.JSONDecodeError, ValueError) as exc:
                self._reply({"ok": False, "error": f"Bad request: {exc}"})
                continue

            try:
                with self.server.session.lock:
                    payload = _handle(self.server.session, req)
                self._reply({"ok": True, **payload})
            except Exception as exc:  # noqa: BLE001 - never drop the study
                log.exception("command %r failed", req.get("cmd"))
                self._reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        log.info("client disconnected: %s", peer)

    def _reply(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.wfile.flush()


class DBOServer(socketserver.ThreadingTCPServer):
    # Socket options are set in server_bind (configure_listener_socket); the stock
    # allow_reuse_address would set SO_REUSEADDR, which shares the port on Windows.
    allow_reuse_address = False
    daemon_threads = True

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        save_dir: Path | str | None = None,
    ) -> None:
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        self.session = _Session(save_dir)
        super().__init__((host, port), _Handler)

    def server_bind(self) -> None:
        configure_listener_socket(self.socket)
        super().server_bind()


def serve(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, save_dir: Path | str | None = None
) -> None:
    server = DBOServer(host, port, save_dir)
    log.info("listening on %s:%d, saving runs under %s", host, port, server.session.save_dir.resolve())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="unity_bridge.py", description="Serve a Dynamic Bayesian optimiser to Unity."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory 'save' requests write into (relative paths only). "
        "Default: the current working directory.",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit binding to a non-loopback address. The server has no "
        "authentication, so only do this on a trusted lab network.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
        parser.error(
            f"Refusing to bind to {args.host} without --allow-remote: the bridge "
            "is unauthenticated."
        )

    try:
        _load_dbo()
    except ImportError as exc:
        parser.error(
            f"dbo_torch is not importable ({exc}). Install dbo-torch, or add BOforUnity's "
            "Assets/StreamingAssets/BOData/BayesianOptimization folder to PYTHONPATH."
        )

    try:
        serve(args.host, args.port, args.save_dir)
    except OSError as exc:
        log.error(
            "Cannot listen on %s:%d (%s). Is another unity_bridge.py still running?",
            args.host,
            args.port,
            exc,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
