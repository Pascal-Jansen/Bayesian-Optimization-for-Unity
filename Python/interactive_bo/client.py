"""Python client of the IBO protocol v1 (protocol/README.md).

Standard library only, so it runs inside experiment software (PsychoPy, robot controllers, scripts)
without the optimizer's dependencies. The server proposes designs; the client evaluates them with a
person and reports the objective values back:

    study = Study(parameters=[Parameter("size", 0.1, 1.0)],
                  objectives=[Objective("rating", 1, 7), Objective("time", 0, 30, minimize=True)])
    with IboClient(port=server.port) as client:
        client.start(study)
        while (design := client.next_design()) is not None:
            client.report(evaluate_with_participant(design.values))
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 56001

# Unity's defaults (BoForUnityManager), so a study behaves the same in every integration.
DEFAULT_CONFIG = {
    "batchSize": 1,
    "numRestarts": 10,
    "rawSamples": 1024,
    "mcSamples": 128,
    "warmStart": False,
}


class ProtocolError(RuntimeError):
    """The server broke the protocol or ended the session before the study finished."""


@dataclass(frozen=True)
class Parameter:
    """A design parameter the optimizer may set, within [low, high] in your own units."""

    key: str
    low: float
    high: float

    def to_message(self) -> Dict[str, Any]:
        return {"key": self.key, "init": {"low": float(self.low), "high": float(self.high)}}


@dataclass(frozen=True)
class Objective:
    """A score the optimizer improves; expected values lie in [low, high]."""

    key: str
    low: float
    high: float
    minimize: bool = False

    def to_message(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "init": {"low": float(self.low), "high": float(self.high), "minimize": 1 if self.minimize else 0},
        }


@dataclass
class Study:
    """Everything the init message describes. ``config`` and ``extra`` pass further settings through.

    ``config`` holds further ``init.config`` keys (backend settings, warm start, ...); ``extra`` holds
    further top-level init fields (``context``, ``cabopGroupCosts``). protocol/README.md lists both.
    """

    parameters: List[Parameter]
    objectives: List[Objective]
    backend: str = "botorch"
    sampling_iterations: Optional[int] = None  # None: 2 * (number of parameters + 1), as in Unity
    optimization_iterations: int = 10
    seed: int = 3
    user_id: str = "-1"
    condition_id: str = "-1"
    group_id: str = "-1"
    config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.parameters or not self.objectives:
            raise ValueError("A study needs at least one parameter and one objective.")
        keys = [p.key for p in self.parameters] + [o.key for o in self.objectives]
        if any(not isinstance(k, str) or not k.strip() for k in keys):
            raise ValueError("Parameter and objective keys must be non-empty strings.")
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"Keys must be unique across parameters and objectives: {duplicates}")
        for item in list(self.parameters) + list(self.objectives):
            if not (math.isfinite(item.low) and math.isfinite(item.high)):
                raise ValueError(f"'{item.key}' needs finite bounds, got ({item.low}, {item.high}).")
            if item.high < item.low:
                raise ValueError(f"'{item.key}' has low {item.low} > high {item.high}.")

    def init_message(self) -> Dict[str, Any]:
        self.validate()
        sampling = self.sampling_iterations
        if sampling is None:
            sampling = 2 * (len(self.parameters) + 1)
        config = dict(DEFAULT_CONFIG)
        config.update({
            "optimizerBackend": self.backend,
            "numSamplingIterations": int(sampling),
            "numOptimizationIterations": int(self.optimization_iterations),
            "seed": int(self.seed),
            "nParameters": len(self.parameters),
            "nObjectives": len(self.objectives),
        })
        config.update(self.config)
        message = {
            "type": "init",
            "protocolVersion": PROTOCOL_VERSION,
            "config": config,
            "parameters": [p.to_message() for p in self.parameters],
            "objectives": [o.to_message() for o in self.objectives],
            "user": {"userId": str(self.user_id), "conditionId": str(self.condition_id), "groupId": str(self.group_id)},
        }
        message.update(self.extra)
        return message


@dataclass(frozen=True)
class Design:
    """A design proposed by the server: parameter values in your units, and its log iteration."""

    values: Dict[str, float]
    iteration: Optional[int] = None


class IboClient:
    """One session with an IBO server: start(), then alternate next_design() and report().

    ``on_message`` is called with every server message (dict), including progress messages.
    Unknown message types are ignored, so newer servers can add messages.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 3600.0,
                 on_message: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.on_message = on_message
        self.coverage: Optional[float] = None           # hypervolume (several objectives) or best value
        self.sampling_progress: Optional[float] = None  # 0..1 during the sampling phase
        self.finished = False
        self._sock: Optional[socket.socket] = None
        self._buffer = b""
        self._awaiting_report = False

    # -- connection --------------------------------------------------------------------------
    def connect(self, connect_timeout: float = 30.0) -> "IboClient":
        """Connect, retrying while the server is still starting up."""
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=connect_timeout)
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        self._sock.settimeout(self.timeout)
        return self

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "IboClient":
        if self._sock is None:
            self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- protocol ----------------------------------------------------------------------------
    def start(self, study: "Study | Mapping[str, Any]") -> None:
        """Send the init message (a Study, or a ready init dict)."""
        message = study.init_message() if isinstance(study, Study) else dict(study)
        if message.get("type") != "init":
            raise ValueError("The init message needs \"type\": \"init\".")
        self._send(message)

    def next_design(self) -> Optional[Design]:
        """Block for the next design; None once the server reports that the study is finished."""
        if self._awaiting_report:
            raise RuntimeError("Report the objectives of the current design before asking for the next one.")
        while True:
            message = self._receive()
            if message is None:
                if self.finished:
                    return None
                raise ProtocolError("The server closed the connection before the study finished "
                                    "(see its console output for the reason).")
            kind = message.get("type")
            if kind == "parameters":
                values = message.get("values")
                if not isinstance(values, dict):
                    raise ProtocolError("Malformed 'parameters' message: 'values' is not an object.")
                iteration = message.get("iteration")
                self._awaiting_report = True
                return Design({str(k): float(v) for k, v in values.items()},
                              int(iteration) if iteration is not None else None)
            if kind == "optimization_finished":
                self.finished = True
                return None
            if kind == "coverage":
                self.coverage = float(message.get("value"))
            elif kind == "tempCoverage":
                self.sampling_progress = float(message.get("value"))

    def report(self, values: Mapping[str, float]) -> None:
        """Send the objective values (in their own units) measured for the current design."""
        if not self._awaiting_report:
            raise RuntimeError("There is no design to report on: call next_design() first.")
        cleaned = {}
        for key, value in values.items():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Objective '{key}' is not a finite number: {value!r}")
            cleaned[str(key)] = number
        self._send({"type": "objectives", "values": cleaned})
        self._awaiting_report = False

    def stop(self, reason: str = "") -> None:
        """End the study early; every completed evaluation stays logged."""
        self._send({"type": "stop", "reason": reason})

    # -- framing -----------------------------------------------------------------------------
    def _send(self, message: Mapping[str, Any]) -> None:
        if self._sock is None:
            raise RuntimeError("Not connected: call connect() first.")
        line = json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n"
        self._sock.sendall(line.encode("utf-8"))

    def _receive(self) -> Optional[Dict[str, Any]]:
        if self._sock is None:
            raise RuntimeError("Not connected: call connect() first.")
        while True:
            cut = self._buffer.find(b"\n")
            if cut >= 0:
                line, self._buffer = self._buffer[:cut], self._buffer[cut + 1:]
                text = line.decode("utf-8").strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError as e:
                    raise ProtocolError(f"Malformed JSON from the server: {e}: {text[:200]!r}") from e
                if not isinstance(message, dict):
                    continue
                if self.on_message is not None:
                    self.on_message(message)
                return message
            chunk = self._sock.recv(65536)
            if not chunk:
                return None
            self._buffer += chunk


def run_study(study: Study, evaluate: Callable[[Dict[str, float]], Mapping[str, float]],
              host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, **client_options) -> List[Dict[str, Any]]:
    """Run a whole study with ``evaluate(parameter values) -> objective values``; returns the history."""
    history = []
    with IboClient(host, port, **client_options) as client:
        client.start(study)
        while (design := client.next_design()) is not None:
            objectives = dict(evaluate(dict(design.values)))
            client.report(objectives)
            history.append({"iteration": design.iteration, "parameters": design.values, "objectives": objectives})
    return history
