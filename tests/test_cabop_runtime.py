"""CABOP backend tests against the real scipy/sklearn stack.

These cover the cost-aware backend (cabop_runtime.py + cabop/bayesopt.py),
most importantly the regression for the parameter-ordering bug: with multiple
CABOP groups whose parameters interleave in declaration order, vector
positions and parameter names must stay aligned end-to-end.

Skipped automatically when scipy/scikit-learn/loguru are not installed (the
lightweight CI environment); they run in a full dev environment and in the
full-stack CI job.
"""

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
import uuid

import numpy as np

# Support both `discover tests` (tests/ on sys.path) and direct module runs.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import (  # noqa: E402
    FakeConn,
    assert_hardened_listener,
    json_line,
    run_main_recording_listener,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization"

# The cabop package lives next to the runtime scripts.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import loguru  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401

    HAS_CABOP_DEPS = True
except ImportError:
    HAS_CABOP_DEPS = False


def load_cabop_runtime():
    name = f"cabop_runtime_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BACKEND_DIR / "cabop_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interleaved_group_init_msg():
    """Three parameters whose groups interleave in declaration order.

    The disjoint bounds make any vector/name misalignment immediately visible:
    a value from one parameter cannot fall inside another parameter's range.
    """
    return {
        "type": "init",
        "config": {
            "numSamplingIterations": 2,
            "numOptimizationIterations": 1,
            "seed": 3,
            "nParameters": 3,
            "nObjectives": 1,
            "warmStart": False,
            "optimizerBackend": "cabop",
            "cabopObjectiveMode": "single",
            "cabopUseCostAwareAcquisition": True,
            "cabopUpdateRule": "actual",
            "cabopEnableCostBudget": False,
            "cabopMaxCumulativeCost": -1.0,
        },
        "parameters": [
            {"key": "p0", "init": {"low": 0.0, "high": 1.0}, "group": "gA", "tolerance": 0.01, "prefabValues": []},
            {"key": "p1", "init": {"low": 10.0, "high": 20.0}, "group": "gB", "tolerance": 0.01, "prefabValues": []},
            {"key": "p2", "init": {"low": 100.0, "high": 200.0}, "group": "gA", "tolerance": 0.01, "prefabValues": []},
        ],
        "objectives": [
            {"key": "o0", "init": {"low": 0.0, "high": 10.0, "minimize": 0}, "weight": 1.0},
        ],
        "cabopGroupCosts": [
            {"group": "gA", "cost": {"unchanged": 1, "swapped": 5, "acquired": 20},
             "actualCost": {"unchanged": 1, "swapped": 5, "acquired": 20}},
            {"group": "gB", "cost": {"unchanged": 1, "swapped": 5, "acquired": 20},
             "actualCost": {"unchanged": 1, "swapped": 5, "acquired": 20}},
        ],
        "user": {"userId": "u", "conditionId": "c", "groupId": "g"},
    }


PARAM_BOUNDS = {"p0": (0.0, 1.0), "p1": (10.0, 20.0), "p2": (100.0, 200.0)}


@unittest.skipUnless(HAS_CABOP_DEPS, "scipy/scikit-learn/loguru not installed")
class BayesOptOrderingTests(unittest.TestCase):
    def _make_optimizer(self):
        from cabop.bayesopt import BayesOpt, BOSpace

        runtime = load_cabop_runtime()
        runtime.parse_init_and_validate(interleaved_group_init_msg(), forced_mode="single")
        space = BOSpace(parameters=runtime.build_cabop_space_dict())
        return BayesOpt(space, ifCost=True, random_state=3), space

    def test_bounds_follow_declaration_order(self):
        _, space = self._make_optimizer()
        np.testing.assert_allclose(
            space.bounds,
            np.array([[0.0, 1.0], [10.0, 20.0], [100.0, 200.0]]),
        )

    def test_numpy_design_round_trip_with_interleaved_groups(self):
        optimizer, _ = self._make_optimizer()
        x = np.array([0.5, 15.0, 150.0])
        design = optimizer._numpy_to_design(x)
        self.assertEqual(design, {"gA": {"p0": 0.5, "p2": 150.0}, "gB": {"p1": 15.0}})
        np.testing.assert_allclose(optimizer._design_to_numpy(design), x)

    def test_optimize_acquisition_clamps_to_bounds(self):
        from cabop.bayesopt import BayesOpt, BOSpace

        # lo + u * (hi - lo) can float-overshoot hi (e.g. -0.1 + 0.4 > 0.3);
        # the optimizer must clamp instead of crashing on a bounds assert.
        space = BOSpace(parameters={
            "groups": ["g"],
            "cost": {"g": {"unchanged": 1.0, "swapped": 5.0, "acquired": 20.0}},
            "actual_cost": {"g": {"unchanged": 1.0, "swapped": 5.0, "acquired": 20.0}},
            "parameters": {
                "p0": {"bound": np.asarray([-0.1, 0.3]), "tolerance": 0.01, "group": "g"},
            },
        })
        optimizer = BayesOpt(space, ifCost=False, random_state=3)

        # Acquisition maximized at the upper unit bound drives u -> 1.0 exactly.
        upper_seeking = lambda X, Xs, Ys, gp: X.sum(axis=1)  # noqa: E731
        x_sample = np.array([[0.5]])
        y_sample = np.array([0.0])
        min_x, _ = optimizer._optimize_acquisition(
            acquisition=upper_seeking,
            X_sample=x_sample,
            Y_sample=y_sample,
            gp=optimizer.gp,
            dim=1,
        )
        self.assertLessEqual(float(min_x[0]), 0.3)
        self.assertGreaterEqual(float(min_x[0]), -0.1)
        self.assertAlmostEqual(float(min_x[0]), 0.3, places=9)

    def test_zero_costs_keep_acquisition_finite(self):
        from cabop.bayesopt import BayesOpt, BOSpace

        space = BOSpace(parameters={
            "groups": ["g"],
            "cost": {"g": {"unchanged": 0.0, "swapped": 0.0, "acquired": 0.0}},
            "actual_cost": {"g": {"unchanged": 0.0, "swapped": 0.0, "acquired": 0.0}},
            "parameters": {
                "p0": {"bound": np.asarray([0.0, 1.0]), "tolerance": 0.01, "group": "g"},
            },
        })
        optimizer = BayesOpt(space, ifCost=True, random_state=3)
        optimizer.tell(np.array([0.5]), 1.0, np.array([0.5]), update_rule="actual")

        values = optimizer._expected_improvement_per_cost(
            np.array([[0.25], [0.75]]),
            optimizer.X_sample,
            optimizer.Y_sample,
            optimizer.gp,
        )
        self.assertTrue(np.all(np.isfinite(values)))

    @staticmethod
    def _one_param_space():
        from cabop.bayesopt import BOSpace

        return BOSpace(parameters={
            "groups": ["g"],
            "cost": {"g": {"unchanged": 1.0, "swapped": 5.0, "acquired": 20.0}},
            "actual_cost": {"g": {"unchanged": 1.0, "swapped": 5.0, "acquired": 20.0}},
            "parameters": {
                "p0": {"bound": np.asarray([0.0, 1.0]), "tolerance": 0.01, "group": "g"},
                "p1": {"bound": np.asarray([0.0, 1.0]), "tolerance": 0.01, "group": "g"},
            },
        })

    def test_initial_design_is_one_sobol_sequence(self):
        from cabop.bayesopt import BayesOpt

        one_at_a_time = BayesOpt(self._one_param_space(), ifCost=False, random_state=3)
        all_at_once = BayesOpt(self._one_param_space(), ifCost=False, random_state=3)
        drawn = np.vstack([one_at_a_time._sobol_sample(1) for _ in range(4)])
        np.testing.assert_allclose(drawn, all_at_once._sobol_sample(4))

    def test_fixed_seed_reproduces_suggestions_whatever_the_global_rng(self):
        from cabop.bayesopt import BayesOpt

        def run(global_seed):
            np.random.seed(global_seed)  # the GP restarts must not draw from here
            optimizer = BayesOpt(self._one_param_space(), ifCost=False, random_state=3)
            suggestions = []
            for _ in range(6):
                x, _ = optimizer.ask(n_init=3)
                y = float((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2)
                optimizer.tell(x, y, x, update_rule="actual")
                suggestions.append(np.asarray(x, dtype=float))
            return np.vstack(suggestions)

        np.testing.assert_array_equal(run(1), run(2))

    def test_both_rule_enters_an_unmoved_design_once(self):
        from cabop.bayesopt import BayesOpt

        optimizer = BayesOpt(self._one_param_space(), ifCost=False, random_state=3)
        optimizer.tell(np.array([0.2, 0.2]), 1.0, np.array([0.2, 0.2]), update_rule="both")
        optimizer.tell(np.array([0.5, 0.5]), 2.0, np.array([0.45, 0.5]), update_rule="both")
        # Two realized rows plus the one intended design that snapping actually moved.
        self.assertEqual(optimizer.X_fit.shape[0], 3)
        self.assertEqual(optimizer.Y_fit.shape[0], 3)


@unittest.skipUnless(HAS_CABOP_DEPS, "scipy/scikit-learn/loguru not installed")
class CabopInitTests(unittest.TestCase):
    def test_group_costs_match_parameter_groups_case_insensitively(self):
        msg = interleaved_group_init_msg()
        msg["parameters"][1]["group"] = "GB"  # Unity treats "GB" and "gB" as one group
        runtime = load_cabop_runtime()
        runtime.parse_init_and_validate(msg, forced_mode="single")
        msg["cabopGroupCosts"][1]["cost"] = {"unchanged": 2, "swapped": 3, "acquired": 4}
        runtime.init_group_costs(msg)
        space = runtime.build_cabop_space_dict()
        self.assertEqual(space["parameters"]["p1"]["group"], "GB")
        self.assertEqual(space["cost"]["GB"], {"unchanged": 2.0, "swapped": 3.0, "acquired": 4.0})

    def test_main_listens_on_loopback_only_and_stops_after_connect(self):
        runtime = load_cabop_runtime()
        server, listening = run_main_recording_listener(
            runtime, interleaved_group_init_msg(), "run_cabop", main_args=("single",)
        )
        assert_hardened_listener(self, runtime.socket, server, listening)

    def test_degenerate_parameter_range_is_rejected_at_init(self):
        msg = interleaved_group_init_msg()
        msg["parameters"][0]["init"] = {"low": 0.5, "high": 0.5}
        runtime = load_cabop_runtime()
        with self.assertRaisesRegex(ValueError, "degenerate range"):
            runtime.parse_init_and_validate(msg, forced_mode="single")


@unittest.skipUnless(HAS_CABOP_DEPS, "scipy/scikit-learn/loguru not installed")
class CabopRuntimeLoopTests(unittest.TestCase):
    def test_multi_group_loop_keeps_parameter_names_aligned(self):
        runtime = load_cabop_runtime()
        runtime.parse_init_and_validate(interleaved_group_init_msg(), forced_mode="single")

        # o0 is maximized on [0, 10]; 7.0 is the best of the three evaluations.
        conn = FakeConn([
            json_line({"type": "objectives", "values": {"o0": 4.0}}),
            json_line({"type": "objectives", "values": {"o0": 7.0}}),
            json_line({"type": "objectives", "values": {"o0": 5.0}}),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            prev_cwd = os.getcwd()
            os.chdir(tmp)
            prev_log_root = os.environ.pop("BO_LOG_ROOT", None)
            try:
                runtime.run_cabop(conn)

                sent = []
                for chunk in conn.sent:
                    for line in chunk.decode("utf-8").splitlines():
                        if line.strip():
                            sent.append(json.loads(line))

                param_msgs = [m for m in sent if m.get("type") == "parameters"]
                self.assertEqual(len(param_msgs), 3)
                for msg in param_msgs:
                    self.assertEqual(sorted(msg["values"].keys()), ["p0", "p1", "p2"])
                    for name, (lo, hi) in PARAM_BOUNDS.items():
                        value = msg["values"][name]
                        self.assertGreaterEqual(
                            value, lo, f"{name}={value} below its own bounds -> misaligned ordering"
                        )
                        self.assertLessEqual(
                            value, hi, f"{name}={value} above its own bounds -> misaligned ordering"
                        )

                self.assertTrue(any(m.get("type") == "optimization_finished" for m in sent))

                import pandas as pd

                obs_csv = pathlib.Path(runtime.PROJECT_PATH) / "ObservationsPerEvaluation.csv"
                df = pd.read_csv(obs_csv, delimiter=";")
                self.assertEqual(len(df), 3)
                for name, (lo, hi) in PARAM_BOUNDS.items():
                    self.assertTrue(
                        df[name].between(lo, hi).all(),
                        f"CSV column {name} out of bounds -> misaligned ordering",
                    )

                # Marker flags use full-precision scalarized values: exactly the
                # o0=7.0 row (best of a maximized objective) is marked TRUE.
                # (pandas parses the TRUE/FALSE strings as booleans.)
                flags = [str(v).strip().upper() for v in df["IsBest"]]
                self.assertEqual(flags, ["FALSE", "TRUE", "FALSE"])
                self.assertEqual(float(df.loc[[f == "TRUE" for f in flags], "o0"].iloc[0]), 7.0)
            finally:
                if prev_log_root is not None:
                    os.environ["BO_LOG_ROOT"] = prev_log_root
                os.chdir(prev_cwd)

    def test_rewritten_log_keeps_ids_as_sent(self):
        msg = interleaved_group_init_msg()
        msg["user"] = {"userId": "007", "conditionId": "01", "groupId": "NA"}
        runtime = load_cabop_runtime()
        runtime.parse_init_and_validate(msg, forced_mode="single")
        conn = FakeConn([
            json_line({"type": "objectives", "values": {"o0": 4.0}}),
            json_line({"type": "objectives", "values": {"o0": 7.0}}),
            json_line({"type": "objectives", "values": {"o0": 5.0}}),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            prev_log_root = os.environ.get("BO_LOG_ROOT")
            os.environ["BO_LOG_ROOT"] = tmp
            try:
                runtime.run_cabop(conn)
                obs_csv = pathlib.Path(runtime.PROJECT_PATH) / "ObservationsPerEvaluation.csv"
                with open(obs_csv, newline="", encoding="utf-8") as f:
                    rows = [line.rstrip("\r\n").split(";") for line in f][1:]
            finally:
                if prev_log_root is None:
                    os.environ.pop("BO_LOG_ROOT", None)
                else:
                    os.environ["BO_LOG_ROOT"] = prev_log_root
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r[:3] == ["007", "01", "NA"] for r in rows), rows)


if __name__ == "__main__":
    unittest.main()
