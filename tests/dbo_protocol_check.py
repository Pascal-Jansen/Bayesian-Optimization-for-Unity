"""End-to-end test of the BOforUnity DBO backend against a mock Unity client.

Starts ``dbo.py`` (the launcher shim for ``dbo_runtime.py``) exactly the way
BOforUnity's PythonStarter does — ``python "<abs path>"`` with the working
directory set to ``<StreamingAssets>/BOData`` and no other argv — then drives it
over a real TCP socket with a client that speaks the wire protocol byte for
byte:

  * Python is the SERVER on 127.0.0.1:56001; this client connects to it.
  * NDJSON: one UTF-8 JSON object per line, terminated by '\\n'.
  * Unity speaks first with {"type": "init", ...}; Python then DRIVES, sending
    {"type": "parameters", ...} and blocking for {"type": "objectives", ...}.
  * Python -> Unity types: parameters, coverage, tempCoverage,
    optimization_finished.

Two things here are deliberate rather than incidental:

  * Two messages are split across two TCP writes (the init message and one
    objectives reply), with a pause in between and Nagle disabled, so the
    backend genuinely has to buffer a partial line across recv() calls.
  * Every value that crosses the wire is narrowed to float32, because Unity's
    ``Dictionary<string, float>`` payloads are float32. A backend that compared
    a returned point against the point it sent for exact equality would break.

The objective drifts with the iteration index, which is what DBO is for: the
optimum moves while the study runs.

Run it directly::

    python tests/dbo_protocol_check.py
    python tests/dbo_protocol_check.py --backend <path to dbo.py>

Deliberately NOT named ``test_*.py``: it launches the real backend with real
torch/botorch and takes ~15s, so it must not be picked up by the stubbed
``python -m unittest discover tests`` CI run. Run it manually after touching
dbo_runtime.py or refreshing the vendored dbo_torch/ package.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HOST = "127.0.0.1"
PORT = 56001

# Small on purpose: the point is protocol fidelity, not optimisation quality. Long
# enough, though, that the GP has something to fit alpha against.
N_SAMPLING = 4
N_OPTIMIZATION = 10
N_TOTAL = N_SAMPLING + N_OPTIMIZATION
VALIDATION_EVERY = 5  # global iterations 5 and 10 become validation iterations

PARAM_BOUNDS = {"speed": (0.0, 10.0), "gain": (-2.0, 2.0)}
OBJ_KEY = "discomfort"
OBJ_LOW, OBJ_HIGH = 0.0, 100.0
OBJ_MINIMIZE = 1  # smallerIsBetter: exercises the sign flip in the canonical frame

CONNECT_TIMEOUT_SEC = 120.0
SOCKET_TIMEOUT_SEC = 600.0
RUN_DEADLINE_SEC = 900.0


class ProtocolError(AssertionError):
    """A protocol expectation was violated."""


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def f32(value):
    """Narrow to float32, exactly as Unity's float fields do."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


# -------------------- mock Unity client --------------------
class MockUnity:
    """A line-oriented NDJSON client with its own receive buffer.

    The buffer mirrors SocketNetwork.cs: bytes are appended as they arrive and
    only complete '\\n'-terminated lines are parsed, so messages may split or
    coalesce across TCP segments in either direction.
    """

    def __init__(self, sock):
        self.sock = sock
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(SOCKET_TIMEOUT_SEC)
        self._buf = ""

    def send(self, obj, split_at=None):
        """Send one NDJSON message, optionally as two separate TCP writes."""
        data = (json.dumps(obj) + "\n").encode("utf-8")
        if split_at is None:
            self.sock.sendall(data)
            return
        require(0 < split_at < len(data), "split point must fall inside the message")
        self.sock.sendall(data[:split_at])
        time.sleep(0.05)  # force the remainder into a distinct TCP segment
        self.sock.sendall(data[split_at:])

    def recv(self):
        """Return the next message, or None when the backend closes the socket."""
        while True:
            idx = self._buf.find("\n")
            if idx >= 0:
                line = self._buf[:idx].rstrip("\r")
                self._buf = self._buf[idx + 1:]
                if not line.strip():
                    continue
                return json.loads(line)
            chunk = self.sock.recv(4096)
            if not chunk:
                require(not self._buf.strip(),
                        f"backend closed the socket mid-message: {self._buf!r}")
                return None
            self._buf += chunk.decode("utf-8")


def build_init_message():
    """The init payload SocketNetwork.cs serialises, plus the dbo* settings.

    Every field the C# InitConfig carries is included — including the cabop*
    and meta* fields other backends own — because a real Unity build always
    sends the whole struct and the backend must ignore what it does not use.
    """
    return {
        "type": "init",
        "config": {
            "batchSize": 1,
            "numRestarts": 2,
            "rawSamples": 32,
            "numOptimizationIterations": N_OPTIMIZATION,
            "mcSamples": 64,
            "numSamplingIterations": N_SAMPLING,
            "seed": 3,
            "nParameters": len(PARAM_BOUNDS),
            "nObjectives": 1,
            "warmStart": False,
            "initialParametersDataPath": "",
            "initialObjectivesDataPath": "",
            "warmStartObjectiveFormat": "auto",
            "optimizerBackend": "dbo",
            "cabopObjectiveMode": "single",
            "cabopUpdateRule": "actual",
            "cabopUseCostAwareAcquisition": False,
            "cabopEnableCostBudget": False,
            "cabopMaxCumulativeCost": 0.0,
            "metaSourceDir": "MetaSources",
            "metaWeightMode": "taf_r",
            "metaRequireSources": True,
            "metaRho": 1.0,
            "metaTargetWeight": 1.0,
            "metaDecayRate": 0.3,
            "metaWarmupIters": 1,
            "metaDecayStartIter": 2,
            "dboSpatialKernel": "rbf",
            "dboAlphaParameterization": "decay",
            "dboInitialAlpha": 0.99,
            "dboAcquisitionTimeOffset": 1.0,
            "dboValidationEvery": VALIDATION_EVERY,
            "dboValidationConfidence": 0.01,
            "dboValidationVisitedOnly": True,
            "dboStationaryBaseline": False,
            "explorationRatio": 0.1,
        },
        "parameters": [
            {
                "key": key,
                "init": {"low": lo, "high": hi},
                "optSeqOrder": i,
                "group": "default",
                "tolerance": 0.0,
                "prefabValues": [],
            }
            for i, (key, (lo, hi)) in enumerate(PARAM_BOUNDS.items())
        ],
        "objectives": [
            {
                "key": OBJ_KEY,
                "init": {"low": OBJ_LOW, "high": OBJ_HIGH, "minimize": OBJ_MINIMIZE},
                "optSeqOrder": 0,
                "weight": 1.0,
            }
        ],
        "cabopGroupCosts": [
            {
                "group": "default",
                "cost": {"unchanged": 1.0, "swapped": 10.0, "acquired": 100.0},
                "actualCost": {"unchanged": 1.0, "swapped": 10.0, "acquired": 100.0},
            }
        ],
        "context": None,
        "user": {
            "userId": "dbo_protocol_test",
            "conditionId": "c1",
            "groupId": "g1",
        },
    }


def drifting_objective(values, iteration):
    """A cost whose optimum walks across the design space as iterations pass.

    Returns a float32 value inside [OBJ_LOW, OBJ_HIGH]; smaller is better, so
    it is reported under minimize=1.
    """
    u = []
    for key, (lo, hi) in PARAM_BOUNDS.items():
        u.append((values[key] - lo) / (hi - lo))

    # The optimum sweeps most of the box over the run: a stationary GP fitted to
    # this trace is wrong, which is the whole point of the temporal kernel.
    centre = (
        min(1.0, max(0.0, 0.30 + 0.05 * iteration)),
        min(1.0, max(0.0, 0.70 - 0.05 * iteration)),
    )
    d2 = sum((a - b) ** 2 for a, b in zip(u, centre, strict=True))
    return f32(min(OBJ_HIGH, OBJ_HIGH * d2 / 0.5))


# -------------------- backend process --------------------
def default_backend_path():
    """The backend inside this repository, overridable via env or --backend."""
    override = os.environ.get("BOFORUNITY_BODATA")
    if override:
        return Path(override) / "BayesianOptimization" / "dbo.py"
    # tests/dbo_protocol_check.py -> <repo>/Assets/StreamingAssets/BOData/...
    repo = Path(__file__).resolve().parents[1]
    return (repo / "Assets" / "StreamingAssets" / "BOData"
            / "BayesianOptimization" / "dbo.py")


def assert_port_free():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        probe.connect((HOST, PORT))
    except OSError:
        return
    finally:
        probe.close()
    raise ProtocolError(
        f"Something is already listening on {HOST}:{PORT}. Stop the stale optimizer "
        "process (or Unity) and re-run."
    )


def start_backend(backend, log_root):
    """Launch the backend the way BOforUnity does: cwd = <StreamingAssets>/BOData."""
    bodata_dir = backend.parent.parent
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Redirect the run logs into a scratch directory so the test never writes
    # into the Unity project. Everything else about the launch is unchanged.
    env["BO_LOG_ROOT"] = str(log_root)
    env["BO_ACCEPT_TIMEOUT_SEC"] = str(int(CONNECT_TIMEOUT_SEC))
    env["BO_SOCKET_TIMEOUT_SEC"] = str(int(SOCKET_TIMEOUT_SEC))

    proc = subprocess.Popen(
        [sys.executable, str(backend)],
        cwd=str(bodata_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output = []

    def pump():
        for line in proc.stdout:
            output.append(line.rstrip("\r\n"))

    threading.Thread(target=pump, daemon=True).start()
    return proc, output


def connect_with_retry(proc):
    deadline = time.time() + CONNECT_TIMEOUT_SEC
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ProtocolError(f"Backend exited before listening (code {proc.returncode}).")
        try:
            return socket.create_connection((HOST, PORT), timeout=5.0)
        except OSError:
            time.sleep(0.2)
    raise ProtocolError(f"Backend did not start listening within {CONNECT_TIMEOUT_SEC}s.")


# -------------------- the conversation --------------------
def run_session(client):
    """Drive one full run and return the observed message tallies."""
    # Split #1: the init message goes out as two TCP writes, so the backend must
    # buffer a partial line before it can parse anything at all.
    client.send(build_init_message(), split_at=40)

    parameters_seen = 0
    temp_coverage = []
    coverage = []
    finished = False
    sent_params = []
    deadline = time.time() + RUN_DEADLINE_SEC

    while True:
        require(time.time() < deadline, f"run exceeded {RUN_DEADLINE_SEC}s")
        msg = client.recv()
        if msg is None:
            break

        require(isinstance(msg, dict) and "type" in msg,
                f"message without a 'type' field: {msg!r}")
        kind = msg["type"]

        if kind == "parameters":
            require(not finished, "received 'parameters' after 'optimization_finished'")
            values = msg.get("values")
            require(isinstance(values, dict), f"'parameters' without a values dict: {msg!r}")
            require(set(values) == set(PARAM_BOUNDS),
                    f"parameter keys {sorted(values)} != {sorted(PARAM_BOUNDS)}")

            narrowed = {}
            for key, (lo, hi) in PARAM_BOUNDS.items():
                value = f32(values[key])  # Unity narrows every parameter to float32
                require(lo - 1e-5 <= value <= hi + 1e-5,
                        f"parameter '{key}' = {value} outside [{lo}, {hi}]")
                narrowed[key] = value

            parameters_seen += 1
            sent_params.append(narrowed)
            cost = drifting_objective(narrowed, parameters_seen)

            # Split #2: one objectives reply is also written in two pieces, this
            # time cutting the JSON body rather than the header.
            reply = {"type": "objectives", "values": {OBJ_KEY: cost}}
            split_at = None
            if parameters_seen == 3:
                split_at = len(json.dumps(reply).encode("utf-8")) // 2
            client.send(reply, split_at=split_at)

        elif kind == "tempCoverage":
            temp_coverage.append(float(msg["value"]))

        elif kind == "coverage":
            coverage.append(float(msg["value"]))

        elif kind == "optimization_finished":
            finished = True

        else:
            raise ProtocolError(f"unknown message type from backend: {kind!r}")

    return {
        "parameters": parameters_seen,
        "temp_coverage": temp_coverage,
        "coverage": coverage,
        "finished": finished,
        "sent_params": sent_params,
    }


def read_csv(path):
    require(path.exists(), f"expected log file is missing: {path}")
    with open(path, newline="") as handle:
        return list(csv.reader(handle, delimiter=";"))


def check_protocol(result):
    require(result["finished"], "backend never sent 'optimization_finished'")
    require(result["parameters"] == N_TOTAL,
            f"expected {N_TOTAL} 'parameters' messages, got {result['parameters']}")
    require(len(result["temp_coverage"]) == N_SAMPLING,
            f"expected {N_SAMPLING} 'tempCoverage' messages, got {len(result['temp_coverage'])}")
    require(abs(result["temp_coverage"][-1] - 1.0) < 1e-6,
            f"final tempCoverage should be 1.0, got {result['temp_coverage'][-1]}")
    # One coverage after the sampling phase (Run 0) plus one per optimization step.
    require(len(result["coverage"]) == N_OPTIMIZATION + 1,
            f"expected {N_OPTIMIZATION + 1} 'coverage' messages, got {len(result['coverage'])}")
    require(all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in result["coverage"]),
            f"coverage outside the normalized [-1,1] frame: {result['coverage']}")
    # Pairwise over consecutive entries, so the two sequences differ in length
    # by one on purpose.
    require(all(b >= a - 1e-9
                for a, b in zip(result["coverage"], result["coverage"][1:], strict=False)),
            f"coverage (best-so-far) must not decrease: {result['coverage']}")

    designs = {tuple(sorted(p.items())) for p in result["sent_params"]}
    require(len(designs) > 1,
            "backend proposed the same design every iteration; the sampling phase "
            "alone should produce distinct Sobol points")


def check_logs(log_root, result):
    run_dir = log_root / "dbo_protocol_test" / "c1" / "run"
    require(run_dir.is_dir(), f"run folder was not created: {run_dir}")

    # -- ObservationsPerEvaluation.csv: the iteration/time axis ----------------
    rows = read_csv(run_dir / "ObservationsPerEvaluation.csv")
    header = rows[0]
    require(header == ["UserID", "ConditionID", "GroupID", "Timestamp", "Iteration",
                       "Phase", "IsBest", OBJ_KEY] + list(PARAM_BOUNDS),
            f"unexpected observations header: {header}")
    body = rows[1:]
    require(len(body) == N_TOTAL, f"expected {N_TOTAL} observation rows, got {len(body)}")

    iterations = [int(r[4]) for r in body]
    phases = [r[5] for r in body]
    require(iterations == list(range(1, N_TOTAL + 1)),
            f"iteration/time counter is not continuous across phases: {iterations}")
    require(phases == ["sampling"] * N_SAMPLING + ["optimization"] * N_OPTIMIZATION,
            f"unexpected phase sequence: {phases}")
    require(sum(1 for r in body if r[6] == "TRUE") >= 1, "no row is flagged IsBest")

    # The objective column is written back in ORIGINAL units, and 'minimize' means
    # the best row is the smallest one.
    logged = [float(r[7]) for r in body]
    best_row = min(range(len(logged)), key=lambda i: logged[i])
    require(body[best_row][6] == "TRUE",
            f"IsBest is not on the lowest logged '{OBJ_KEY}' row (minimize=1): {logged}")

    # -- DboDiagnosticsPerEvaluation.csv: the fitted alpha ---------------------
    rows = read_csv(run_dir / "DboDiagnosticsPerEvaluation.csv")
    header = rows[0]
    require(header[:4] == ["Iteration", "Phase", "IsValidation", "Alpha"],
            f"unexpected diagnostics header: {header}")
    body = rows[1:]
    require(len(body) == N_TOTAL, f"expected {N_TOTAL} diagnostics rows, got {len(body)}")
    require([int(r[0]) for r in body] == list(range(1, N_TOTAL + 1)),
            "diagnostics iteration column is not continuous")

    sampling_alpha = [r[3] for r in body[:N_SAMPLING]]
    require(all(a == "" for a in sampling_alpha),
            f"alpha should be empty while sampling (no model yet): {sampling_alpha}")

    opt_alpha = [r[3] for r in body[N_SAMPLING:]]
    require(all(a != "" for a in opt_alpha), f"alpha missing on optimization rows: {opt_alpha}")
    alphas = [float(a) for a in opt_alpha]
    require(all(0.0 < a <= 1.0 for a in alphas), f"alpha outside (0, 1]: {alphas}")

    validation_rows = [int(r[0]) for r in body if r[2] == "TRUE"]
    expected_validation = [i for i in range(N_SAMPLING + 1, N_TOTAL + 1)
                           if i % VALIDATION_EVERY == 0]
    require(validation_rows == expected_validation,
            f"validation iterations {validation_rows} != expected {expected_validation}")

    # -- the metric files bo.py's tooling reads -------------------------------
    best_rows = read_csv(run_dir / "BestObjectivePerEvaluation.csv")
    require(best_rows[0] == ["BestObjective", "Run"], f"unexpected header: {best_rows[0]}")
    require(len(best_rows) - 1 == N_OPTIMIZATION + 1,
            f"expected {N_OPTIMIZATION + 1} best-objective rows, got {len(best_rows) - 1}")
    require([int(r[1]) for r in best_rows[1:]] == list(range(0, N_OPTIMIZATION + 1)),
            "BestObjectivePerEvaluation Run column is not 0..N")

    legacy_rows = read_csv(run_dir / "HypervolumePerEvaluation.csv")
    require(legacy_rows[0] == ["Hypervolume", "Run"], f"unexpected header: {legacy_rows[0]}")
    require(len(legacy_rows) == len(best_rows), "legacy mirror row count differs")

    exec_rows = read_csv(run_dir / "ExecutionTimes.csv")
    require(exec_rows[0] == ["Optimization", "Execution_Time"],
            f"unexpected header: {exec_rows[0]}")
    require(len(exec_rows) - 1 == N_OPTIMIZATION,
            f"expected {N_OPTIMIZATION} execution-time rows, got {len(exec_rows) - 1}")

    return alphas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=default_backend_path(),
                        help="path to dbo.py inside the BOforUnity project")
    args = parser.parse_args()

    backend = args.backend.resolve()
    if not backend.exists():
        print(f"FAIL: backend not found at {backend}")
        print("      Pass --backend <path to dbo.py> or set BOFORUNITY_BODATA.")
        return 1

    print(f"Backend : {backend}")
    print(f"Launch  : {sys.executable} \"{backend}\"  (cwd={backend.parent.parent})")

    proc = None
    output = []
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="dbo_protocol_test_") as tmp:
        log_root = Path(tmp) / "LogData"
        try:
            assert_port_free()
            proc, output = start_backend(backend, log_root)
            sock = connect_with_retry(proc)
            print(f"Connected to {HOST}:{PORT}")
            try:
                result = run_session(MockUnity(sock))
            finally:
                sock.close()

            check_protocol(result)
            alphas = check_logs(log_root, result)

            code = proc.wait(timeout=60)
            require(code == 0, f"backend exited with code {code}")

        except Exception as exc:  # noqa: BLE001 - the test reports, it does not raise
            print()
            print("---- backend stdout (last 40 lines) ----")
            for line in output[-40:]:
                print("  " + line)
            print("----------------------------------------")
            print(f"FAIL: {type(exc).__name__}: {exc}")
            return 1
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

    elapsed = time.time() - started
    print()
    print(f"parameters messages : {result['parameters']} "
          f"({N_SAMPLING} sampling + {N_OPTIMIZATION} optimization)")
    print(f"tempCoverage        : {len(result['temp_coverage'])}")
    print(f"coverage            : {[round(v, 4) for v in result['coverage']]}")
    print(f"fitted alpha        : {[round(a, 4) for a in alphas]}")
    print("split messages      : init (2 writes), objectives #3 (2 writes)")
    print(f"elapsed             : {elapsed:.1f}s")
    print()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
