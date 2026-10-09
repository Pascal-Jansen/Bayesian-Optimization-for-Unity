"""CABOP runtime tests that need only numpy + pandas (the CI environment).

cabop_runtime.py is loaded with a stand-in for the vendored cabop.bayesopt optimizer, so the
runtime's own logic -- warm start and the iteration axis, the IsBest/IsPareto flags, log
precision and init validation -- is covered where scipy/scikit-learn/loguru are missing.
tests/test_cabop_runtime.py repeats the end-to-end checks against the real optimizer.
"""

import contextlib
import csv
import importlib.util
import io
import itertools
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
import uuid
from unittest import mock

import numpy as np
import pandas as pd

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import BACKEND_DIR, FakeConn, json_line, reset_protocol_state  # noqa: E402

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class StubBayesOpt:
    """Proposes a fixed walk through the box and realizes every proposal unchanged."""

    instances = []

    def __init__(self, space, ifCost=True, random_state=None):
        self.space = space
        self.current_best = {"x": None, "y": float("inf")}
        self.n_init_seen = []
        self.told = []
        StubBayesOpt.instances.append(self)

    def ask(self, n_init=5):
        self.n_init_seen.append(n_init)
        lo, hi = self.space.bounds[:, 0], self.space.bounds[:, 1]
        u = (0.137 + 0.29 * len(self.n_init_seen)) % 1.0
        return lo + u * (hi - lo), None

    def select_sample(self, x, prefab=None):
        return (1.0,), np.asarray(x, dtype=float)

    def tell(self, x, y, x_intended=None, update_rule="actual"):
        self.told.append((np.asarray(x, dtype=float), float(y), update_rule))
        if y < self.current_best["y"]:
            self.current_best = {"x": x, "y": float(y)}


class StubBOSpace:
    def __init__(self, parameters):
        self.parameters = parameters
        self.bounds = np.array([p["bound"] for p in parameters["parameters"].values()], dtype=float)


@contextlib.contextmanager
def cabop_stub_modules():
    package = types.ModuleType("cabop")
    package.__path__ = []
    bayesopt = types.ModuleType("cabop.bayesopt")
    bayesopt.BayesOpt = StubBayesOpt
    bayesopt.BOSpace = StubBOSpace
    package.bayesopt = bayesopt
    saved = {name: sys.modules.get(name) for name in ("cabop", "cabop.bayesopt")}
    sys.modules.update({"cabop": package, "cabop.bayesopt": bayesopt})
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_runtime():
    name = f"cabop_runtime_stub_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BACKEND_DIR / "cabop_runtime.py")
    module = importlib.util.module_from_spec(spec)
    with cabop_stub_modules():
        spec.loader.exec_module(module)
    reset_protocol_state()
    StubBayesOpt.instances = []
    return module


def init_msg(mode="single", sampling=2, optimization=2, param_range=(0.0, 10.0), objectives=None, **config):
    if objectives is None:
        objectives = [("o0", 0.0, 10.0, 0)]  # maximized
    msg = {
        "type": "init",
        "config": {
            "numSamplingIterations": sampling, "numOptimizationIterations": optimization, "seed": 3,
            "nParameters": 1, "nObjectives": len(objectives), "warmStart": False,
            "optimizerBackend": "cabop", "cabopObjectiveMode": mode,
        },
        "parameters": [{"key": "p0", "init": {"low": param_range[0], "high": param_range[1]},
                        "group": "g", "tolerance": 0.05, "prefabValues": []}],
        "objectives": [{"key": key, "init": {"low": lo, "high": hi, "minimize": minimize}, "weight": 1.0}
                       for key, lo, hi, minimize in objectives],
        "user": {"userId": "u", "conditionId": "c", "groupId": "g"},
    }
    msg["config"].update(config)
    return msg


class CabopStubTestCase(unittest.TestCase):
    def start(self, msg, mode=None, warm_params=None, warm_objectives=None):
        """Load the runtime and parse ``msg``; optional warm-start CSVs in a temp init root."""
        self.init_root = None
        if warm_params is not None:
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            self.init_root = tmp.name
            pd.DataFrame(warm_params).to_csv(os.path.join(tmp.name, "x.csv"), sep=";", index=False)
            pd.DataFrame(warm_objectives).to_csv(os.path.join(tmp.name, "y.csv"), sep=";", index=False)
            msg["config"].update(warmStart=True, initialParametersDataPath="x.csv",
                                 initialObjectivesDataPath="y.csv")
        runtime = load_runtime()
        runtime.parse_init_and_validate(msg, forced_mode=mode or msg["config"]["cabopObjectiveMode"])
        return runtime

    def run_session(self, runtime, replies):
        conn = FakeConn([json_line({"type": "objectives", "values": v}) for v in replies])
        logs = {}
        with tempfile.TemporaryDirectory() as tmp:
            env = {"BO_LOG_ROOT": tmp}
            if self.init_root is not None:
                env["BO_INIT_ROOT"] = self.init_root
            with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
                runtime.run_cabop(conn)
            for path in pathlib.Path(runtime.PROJECT_PATH).glob("*.csv"):
                with open(path, newline="", encoding="utf-8") as f:
                    logs[path.name] = list(csv.DictReader(f, delimiter=";"))
        sent = [json.loads(line) for chunk in conn.sent for line in chunk.decode("utf-8").splitlines() if line]
        return sent, logs


class WarmStartTests(CabopStubTestCase):
    def test_without_warm_start_the_sampling_phase_comes_first(self):
        runtime = self.start(init_msg(sampling=2, optimization=2))
        sent, logs = self.run_session(runtime, [{"o0": v} for v in (1.0, 2.0, 3.0, 4.0)])
        self.assertEqual([m["iteration"] for m in sent if m["type"] == "parameters"], [1, 2, 3, 4])
        rows = logs["ObservationsPerEvaluation.csv"]
        self.assertEqual([r["Phase"] for r in rows], ["sampling", "sampling", "optimization", "optimization"])
        self.assertEqual(StubBayesOpt.instances[0].n_init_seen, [2, 2, 2, 2])
        self.assertEqual([int(r["Iteration"]) for r in logs["BestObjectivePerEvaluation.csv"]], [1, 2, 3, 4])

    def test_warm_start_replaces_sampling_and_continues_the_iteration_axis(self):
        runtime = self.start(init_msg(sampling=3, optimization=2),
                             warm_params={"p0": [2.0, 7.0]}, warm_objectives={"o0": [9.0, 3.0]})
        sent, logs = self.run_session(runtime, [{"o0": 5.0}, {"o0": 6.0}])
        optimizer = StubBayesOpt.instances[0]
        # The model takes over at once (n_init 0): no Sobol points from the optimization budget.
        self.assertEqual(optimizer.n_init_seen, [0, 0])
        self.assertEqual(len(optimizer.told), 4)
        self.assertEqual([m["iteration"] for m in sent if m["type"] == "parameters"], [3, 4])
        rows = logs["ObservationsPerEvaluation.csv"]
        self.assertEqual([int(r["Iteration"]) for r in rows], [3, 4])
        self.assertEqual([r["Phase"] for r in rows], ["optimization", "optimization"])
        self.assertEqual([int(r["Optimization"]) for r in logs["ExecutionTimes.csv"]], [3, 4])
        self.assertEqual([int(r["Iteration"]) for r in logs["CABOPMetricsPerEvaluation.csv"]], [3, 4])
        # Baseline row at the last warm-start row (as bo.py): best o0 = 9 of [0, 10] -> coverage 0.9.
        metric = logs["BestObjectivePerEvaluation.csv"]
        self.assertEqual([int(r["Iteration"]) for r in metric], [2, 3, 4])
        self.assertAlmostEqual(float(metric[0]["BestObjective"]), 0.9)
        coverage = [m["value"] for m in sent if m["type"] == "coverage"]
        self.assertAlmostEqual(coverage[0], 0.9)
        self.assertEqual(len(coverage), 3)
        # Progress counts this run's evaluations only.
        self.assertEqual([m["value"] for m in sent if m["type"] == "tempCoverage"], [0.5, 1.0])
        # The warm-start best beats every live row: no live row is IsBest (as in bo.py).
        self.assertEqual([r["IsBest"] for r in rows], ["FALSE", "FALSE"])

    def test_a_live_row_better_than_the_warm_start_is_best(self):
        runtime = self.start(init_msg(sampling=3, optimization=2),
                             warm_params={"p0": [2.0, 7.0]}, warm_objectives={"o0": [4.0, 3.0]})
        _, logs = self.run_session(runtime, [{"o0": 6.0}, {"o0": 5.0}])
        self.assertEqual([r["IsBest"] for r in logs["ObservationsPerEvaluation.csv"]], ["TRUE", "FALSE"])


class ParetoTests(CabopStubTestCase):
    OBJECTIVES = [("o0", 0.0, 10.0, 0), ("o1", 0.0, 10.0, 1)]  # o0 maximized, o1 minimized
    REPLIES = [{"o0": 5.0, "o1": 5.0}, {"o0": 8.0, "o1": 2.0}, {"o0": 9.0, "o1": 6.0},
               {"o0": 8.0, "o1": 2.0}, {"o0": 1.0, "o1": 0.0}]

    def test_is_pareto_is_the_non_dominated_set_with_one_copy_of_duplicates(self):
        runtime = self.start(init_msg(mode="multi", sampling=5, optimization=0, objectives=self.OBJECTIVES))
        _, logs = self.run_session(runtime, self.REPLIES)
        # (5,5) is dominated by (8,2); (8,2), (9,6) and (1,0) trade off; the second (8,2) is a copy.
        # Was: only the rows with the lowest weighted score, i.e. both copies of (8,2).
        self.assertEqual([r["IsPareto"] for r in logs["ObservationsPerEvaluation.csv"]],
                         ["FALSE", "TRUE", "TRUE", "FALSE", "TRUE"])

    def test_flags_follow_each_objective_direction(self):
        flipped = [("o0", 0.0, 10.0, 1), ("o1", 0.0, 10.0, 0)]  # now o0 minimized, o1 maximized
        runtime = self.start(init_msg(mode="multi", sampling=5, optimization=0, objectives=flipped))
        _, logs = self.run_session(runtime, self.REPLIES)
        # (5,5) and (1,0) trade off; (8,2) is dominated by (5,5); (9,6) trades off.
        self.assertEqual([r["IsPareto"] for r in logs["ObservationsPerEvaluation.csv"]],
                         ["TRUE", "FALSE", "TRUE", "FALSE", "TRUE"])

    def test_warm_start_rows_compete_for_is_pareto(self):
        runtime = self.start(init_msg(mode="multi", sampling=2, optimization=3, objectives=self.OBJECTIVES),
                             warm_params={"p0": [4.0]}, warm_objectives={"o0": [8.5], "o1": [2.0]})
        _, logs = self.run_session(runtime, self.REPLIES[:3])
        # (8.5, 2) from the warm start dominates the live (5,5) and (8,2).
        self.assertEqual([r["IsPareto"] for r in logs["ObservationsPerEvaluation.csv"]],
                         ["FALSE", "FALSE", "TRUE"])

    def test_non_dominated_mask_properties(self):
        runtime = load_runtime()
        rng = np.random.default_rng(1)
        for _ in range(200):
            values = rng.integers(1, 5, size=(int(rng.integers(1, 12)), int(rng.integers(2, 4)))).astype(float)
            keep = runtime.non_dominated_mask(values)

            def dominates(a, b):
                return bool(np.all(a >= b) and np.any(a > b))

            kept = values[keep]
            # Kept rows are distinct and do not dominate each other ...
            self.assertEqual(len({tuple(r) for r in kept}), len(kept))
            for a, b in itertools.permutations(kept, 2):
                self.assertFalse(dominates(a, b))
            # ... every other row is dominated, or a later copy of a kept row ...
            for i in np.flatnonzero(~keep):
                row = values[i]
                self.assertTrue(any(dominates(k, row) for k in kept) or
                                any(np.array_equal(values[j], row) for j in range(i)))
            # ... and the first copy is the one kept.
            for i in np.flatnonzero(keep):
                self.assertFalse(any(np.array_equal(values[j], values[i]) for j in range(i)))


class LogAndInitTests(CabopStubTestCase):
    def test_observation_log_keeps_ten_significant_digits(self):
        runtime = self.start(init_msg(sampling=2, optimization=0, param_range=(0.0, 0.004),
                                      objectives=[("o0", 0.0, 0.004, 1)]))
        sent, logs = self.run_session(runtime, [{"o0": 0.0035}, {"o0": 0.000274}])
        rows = logs["ObservationsPerEvaluation.csv"]
        designs = [m["values"]["p0"] for m in sent if m["type"] == "parameters"]
        # Was: 3 decimals (0.0035 -> 0.004, 0.000274 -> 0.0, the designs -> 0.001/0.002).
        self.assertEqual([r["p0"] for r in rows], [repr(float(f"{v:.10g}")) for v in designs])
        self.assertEqual([r["o0"] for r in rows], ["0.0035", "0.000274"])

    def test_tolerance_must_be_a_fraction_of_the_range(self):
        for tolerance in (-0.01, 1.01, 5.0, None, "abc"):
            msg = init_msg()
            if tolerance is None:
                del msg["parameters"][0]["tolerance"]
            else:
                msg["parameters"][0]["tolerance"] = tolerance
            with self.subTest(tolerance=tolerance):
                if tolerance is None:  # missing: the default, 5% of the range
                    runtime = self.start(msg)
                    self.assertEqual(runtime.parameters_info[0][3], 0.05)
                else:
                    with self.assertRaisesRegex(ValueError, r"must lie in \[0, 1\]"):
                        self.start(msg)
        for tolerance in (0.0, 0.05, 1.0):
            msg = init_msg()
            msg["parameters"][0]["tolerance"] = tolerance
            self.assertEqual(self.start(msg).parameters_info[0][3], tolerance)

    def test_prefabricated_values_must_lie_within_the_bounds(self):
        msg = init_msg(param_range=(0.0, 10.0))
        msg["parameters"][0]["prefabValues"] = [0.0, 5.0, 10.0]
        self.assertEqual(self.start(msg).parameters_info[0][4], [0.0, 5.0, 10.0])
        msg["parameters"][0]["prefabValues"] = [5.0, 10.5]
        with self.assertRaisesRegex(ValueError, r"\[10.5\] of parameter 'p0' lie outside its bounds"):
            self.start(msg)


if __name__ == "__main__":
    unittest.main()
