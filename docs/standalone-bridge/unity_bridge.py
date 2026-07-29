"""TCP/JSON server exposing a :class:`~dbo_torch.optimizer.DynamicBO` to Unity.

Unity cannot host PyTorch, so the optimiser runs as a small local server and
the Unity scene talks to it over a socket. The protocol is newline-delimited
JSON: one request object per line, one response object per line. The matching
C# client is in ``unity/DboClient.cs``.

Run it with::

    dbo-serve --host 127.0.0.1 --port 8756

or::

    python -m dbo_torch.unity_bridge --port 8756

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
    Write the history to disk.
``{"cmd": "ping"}``
    Liveness check.

Every response carries ``ok``. On failure it carries ``ok: false`` and
``error``. The server keeps running after a failed request: a study should not
die because one iteration hit a numerical problem.

The server binds to localhost by default and performs no authentication. It is
intended for a lab machine, not a shared network. Binding to a non-loopback
address requires ``--allow-remote``, so it cannot happen by accident.
"""

from __future__ import annotations

import argparse
import json
import logging
import socketserver
import threading
from typing import Any

from dbo_torch.model import DBOModelConfig
from dbo_torch.optimizer import DBOConfig, DynamicBO

__all__ = ["DBOServer", "serve", "main"]

log = logging.getLogger("dbo_torch.bridge")


class _Session:
    """Holds the optimiser, with a lock serialising request handling.

    The lock makes each individual request atomic; it does not arbitrate
    between multiple clients, which all share the single run. One client per
    run is the intended deployment."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.optimizer: DynamicBO | None = None

    def reset(self, req: dict[str, Any]) -> dict[str, Any]:
        bounds = req.get("bounds")
        if not bounds:
            raise ValueError("'reset' requires 'bounds', e.g. [[-5, 9]]")

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
        opt = session.require()
        path = req.get("path")
        if not path:
            raise ValueError("'save' requires 'path'")
        return {"path": str(opt.save(path))}

    raise ValueError(f"Unknown command: {cmd!r}")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        peer = self.client_address
        log.info("client connected: %s", peer)

        for raw in self.rfile:
            line = raw.decode("utf-8").strip()
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
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int) -> None:
        super().__init__((host, port), _Handler)
        self.session = _Session()


def serve(host: str = "127.0.0.1", port: int = 8756) -> None:
    server = DBOServer(host, port)
    log.info("listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dbo-serve", description="Serve a Dynamic Bayesian optimiser to Unity."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8756)
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

    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
