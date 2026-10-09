"""DBO backend (dbo_runtime.py) tests against the real torch/botorch stack.

The DBO backend imports the vendored dbo_torch package, which needs real torch,
botorch and gpytorch, so these tests are skipped automatically on the
lightweight CI environment and run locally in a full dev environment.

Other test modules in this suite replace torch/botorch in ``sys.modules`` with
stubs. The real module objects are captured here at import time (before any
test runs) and restored around each test, as in test_contextual_integration.

The end-to-end wire protocol (real TCP socket, split messages, float32
narrowing) is covered separately by ``tests/dbo_protocol_check.py``.
"""

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
import uuid
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization"

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import FakeServerSocket, json_line  # noqa: E402

_REAL_MODULE_ROOTS = ("torch", "botorch", "gpytorch", "linear_operator", "pandas")

try:
    import torch  # noqa: F401
    import botorch  # noqa: F401
    import botorch.acquisition  # noqa: F401
    import botorch.acquisition.analytic  # noqa: F401
    import botorch.fit  # noqa: F401
    import botorch.models  # noqa: F401
    import botorch.models.transforms  # noqa: F401
    import botorch.optim  # noqa: F401
    import botorch.utils.sampling  # noqa: F401
    import botorch.utils.transforms  # noqa: F401
    import gpytorch.constraints  # noqa: F401
    import gpytorch.kernels  # noqa: F401
    import gpytorch.mlls  # noqa: F401
    import pandas  # noqa: F401

    HAS_REAL_STACK = True
    _REAL_MODULES = {
        name: mod
        for name, mod in sys.modules.items()
        if name.split(".")[0] in _REAL_MODULE_ROOTS and mod is not None
    }
except ImportError:
    HAS_REAL_STACK = False
    _REAL_MODULES = {}


def restore_real_modules():
    """Replace stub torch/botorch/... entries with the captured real modules."""
    sys.modules.update(_REAL_MODULES)
    for name in list(sys.modules):
        if name.split(".")[0] not in _REAL_MODULE_ROOTS or name in _REAL_MODULES:
            continue
        if getattr(sys.modules[name], "__spec__", None) is None:
            del sys.modules[name]


def load_dbo_runtime():
    restore_real_modules()
    name = f"dbo_runtime_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BACKEND_DIR / "dbo_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


N_SAMPLING = 3
N_OPTIMIZATION = 4
VALIDATION_EVERY = 2  # global iterations 4 and 6 (both after sampling)


def init_message(**overrides):
    config = {
        "numSamplingIterations": N_SAMPLING,
        "numOptimizationIterations": N_OPTIMIZATION,
        "batchSize": 1,
        "numRestarts": 2,
        "rawSamples": 32,
        "mcSamples": 8,
        "seed": 7,
        "nParameters": 2,
        "nObjectives": 1,
        "warmStart": False,
        "initialParametersDataPath": "",
        "initialObjectivesDataPath": "",
        "dboValidationEvery": VALIDATION_EVERY,
    }
    config.update(overrides)
    return {
        "type": "init",
        "config": config,
        "parameters": [
            {"key": "p0", "init": {"low": 0.0, "high": 10.0}},
            {"key": "p1", "init": {"low": -1.0, "high": 1.0}},
        ],
        "objectives": [{"key": "cost", "init": {"low": 0.0, "high": 100.0, "minimize": 1}}],
        "user": {"userId": "u", "conditionId": "c", "groupId": "g"},
    }


class ScriptedUnity:
    """Answers every 'parameters' message with a drifting cost, like a participant."""

    def __init__(self, init_msg):
        self._outbox = [json_line(init_msg)]
        self.sent = []

    def recv(self, n):
        return self._outbox.pop(0) if self._outbox else b""

    def sendall(self, data):
        for line in data.decode("utf-8").splitlines():
            msg = json.loads(line)
            self.sent.append(msg)
            if msg.get("type") == "parameters":
                iteration = sum(1 for m in self.sent if m.get("type") == "parameters")
                u0 = msg["values"]["p0"] / 10.0
                u1 = (msg["values"]["p1"] + 1.0) / 2.0
                centre = 0.2 + 0.08 * iteration  # the optimum walks as the study runs
                cost = 100.0 * min(1.0, ((u0 - centre) ** 2 + (u1 - 0.5) ** 2))
                self._outbox.append(json_line({"type": "objectives", "values": {"cost": cost}}))

    def settimeout(self, timeout):
        pass

    def shutdown(self, how):
        pass

    def close(self):
        pass

    def parameters_sent(self):
        return [m["values"] for m in self.sent if m.get("type") == "parameters"]


@unittest.skipUnless(HAS_REAL_STACK, "requires torch, botorch and gpytorch")
class DboRuntimeTests(unittest.TestCase):
    def _run_main(self, module, init_msg, log_root, execute=None):
        conn = ScriptedUnity(init_msg)
        captured = {}
        real_execute = module.dbo_execute

        def capture(*args):
            captured["result"] = (execute or real_execute)(*args)
            return captured["result"]

        threads = torch.get_num_threads()
        try:
            with mock.patch.object(module.socket, "socket", lambda *a, **k: FakeServerSocket(conn)), \
                    mock.patch.object(module, "dbo_execute", capture), \
                    mock.patch.dict(os.environ, {"BO_LOG_ROOT": str(log_root)}):
                module.main()
        finally:
            torch.set_num_threads(threads)  # main() pins it for the whole process
        return conn, captured.get("result")

    def test_initial_alpha_of_one_is_rejected_before_any_evaluation(self):
        module = load_dbo_runtime()
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "dboInitialAlpha must lie in \\(0, 1\\)"):
                self._run_main(module, init_message(dboInitialAlpha=1.0), tmp,
                               execute=lambda *a: calls.append(a))
        self.assertEqual(calls, [], "the study must not start with an unfittable alpha")

    def test_initial_alpha_of_one_is_accepted_for_the_stationary_baseline(self):
        module = load_dbo_runtime()
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            self._run_main(module, init_message(dboInitialAlpha=1.0, dboStationaryBaseline=True),
                           tmp, execute=lambda *a: calls.append(a))
        self.assertEqual(len(calls), 1)

    def test_run_is_reproducible_validated_and_recorded(self):
        runs = []
        with tempfile.TemporaryDirectory() as tmp:
            for attempt in range(2):
                # Disturb torch's global generator between runs: the optimizer owns its
                # RNG, so this must not change a single suggestion.
                torch.manual_seed(1234 + attempt)
                torch.rand(attempt * 17 + 3)
                module = load_dbo_runtime()
                conn, result = self._run_main(module, init_message(), pathlib.Path(tmp) / str(attempt))
                runs.append((module, conn, result))

            (_, conn_a, _), (module, conn_b, (_, dbo)) = runs
            self.assertEqual(conn_a.parameters_sent(), conn_b.parameters_sent())
            self.assertEqual(len(conn_b.parameters_sent()), N_SAMPLING + N_OPTIMIZATION)

            run_dir = pathlib.Path(tmp) / "1" / "u" / "c" / "run"

            # Validation iterations: scheduled on the global index, never during sampling.
            with open(run_dir / "DboDiagnosticsPerEvaluation.csv", newline="") as f:
                rows = [line.rstrip("\r\n").split(";") for line in f][1:]
            flagged = [int(r[0]) for r in rows if r[2] == "TRUE"]
            self.assertEqual(flagged, [4, 6])
            # Every model-driven suggestion carries the model's prediction.
            self.assertTrue(all(r[4] != "" for r in rows[N_SAMPLING:]))
            self.assertTrue(all(r[4] == "" for r in rows[:N_SAMPLING]))

            # Best-so-far per evaluation on the global index, as bo.py logs it.
            with open(run_dir / "BestObjectivePerEvaluation.csv", newline="") as f:
                best_rows = [line.rstrip("\r\n").split(";") for line in f][1:]
            self.assertEqual([int(r[1]) for r in best_rows],
                             list(range(1, N_SAMPLING + N_OPTIMIZATION + 1)))

            # DboRunState.json: provenance, and an exact record of the optimizer.
            state_path = run_dir / module.DBO_STATE_FILENAME
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["versions"]["dbo_torch"], module.dbo_torch.__version__)
            self.assertEqual(state["config"]["validation_every"], VALIDATION_EVERY)
            self.assertEqual(state["config"]["seed"], 7)
            self.assertEqual(len(state["observations"]), N_SAMPLING + N_OPTIMIZATION)

            # Resuming from the record continues exactly as the live optimizer would.
            resumed = module.DynamicBO.load(state_path)
            self.assertEqual(resumed.suggest(), dbo.suggest())

    def test_warm_start_rows_lead_the_time_axis_and_the_metric_log(self):
        module = load_dbo_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            init_root = pathlib.Path(tmp) / "InitData"
            init_root.mkdir()
            (init_root / "x.csv").write_text("p0;p1\n2.0;-0.5\n8.0;0.5\n5.0;0.0\n", encoding="utf-8")
            (init_root / "y.csv").write_text("cost\n40.0\n60.0\n20.0\n", encoding="utf-8")
            msg = init_message(
                numSamplingIterations=0, numOptimizationIterations=2, warmStart=True,
                initialParametersDataPath="x.csv", initialObjectivesDataPath="y.csv",
                dboValidationEvery=0,
            )
            with mock.patch.dict(os.environ, {"BO_INIT_ROOT": str(init_root)}):
                conn, (_, dbo) = self._run_main(module, msg, pathlib.Path(tmp) / "logs")
            run_dir = pathlib.Path(tmp) / "logs" / "u" / "c" / "run"

            self.assertEqual(len(conn.parameters_sent()), 2)
            self.assertEqual([o.time for o in dbo.observations], [1.0, 2.0, 3.0, 4.0, 5.0])

            with open(run_dir / "ObservationsPerEvaluation.csv", newline="", encoding="utf-8") as f:
                obs_rows = [line.rstrip("\r\n").split(";") for line in f][1:]
            self.assertEqual([int(r[4]) for r in obs_rows], [4, 5])

            # Baseline over the warm-start rows at the last of them, then the same
            # Iteration as each evaluation's observation row.
            with open(run_dir / "BestObjectivePerEvaluation.csv", newline="", encoding="utf-8") as f:
                best_rows = [line.rstrip("\r\n").split(";") for line in f][1:]
            self.assertEqual([int(r[1]) for r in best_rows], [3, 4, 5])
            # cost 20 on [0, 100] with minimize=1 is +0.6 in the canonical frame.
            self.assertAlmostEqual(float(best_rows[0][0]), 0.6)


if __name__ == "__main__":
    unittest.main()
