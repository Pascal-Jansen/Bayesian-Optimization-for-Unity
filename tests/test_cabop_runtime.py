"""CABOP backend tests against the real scipy/sklearn stack.

These cover the cost-aware backend (cabop_runtime.py + cabop/bayesopt.py),
most importantly the regression for the parameter-ordering bug: with multiple
CABOP groups whose parameters interleave in declaration order, vector
positions and parameter names must stay aligned end-to-end.

Skipped automatically when scipy/scikit-learn/loguru are not installed (the
lightweight CI environment); they run in a full dev environment and in the
full-stack CI job.
"""

import contextlib
import csv
import importlib
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
import uuid
from unittest import mock

import numpy as np

# Support both `discover tests` (tests/ on sys.path) and direct module runs.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import (  # noqa: E402
    FakeConn,
    assert_hardened_listener,
    json_line,
    reset_protocol_state,
    run_main_recording_listener,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization"

# The cabop package lives next to the runtime scripts.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    for _dependency in ("loguru", "scipy", "sklearn"):
        importlib.import_module(_dependency)

    HAS_CABOP_DEPS = True
except ImportError:
    HAS_CABOP_DEPS = False


def load_cabop_runtime():
    name = f"cabop_runtime_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BACKEND_DIR / "cabop_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reset_protocol_state()
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


def simple_init_msg(mode="single", sampling=2, optimization=1, param_range=(0.0, 1.0), n_params=1,
                    objectives=None, **config):
    """One CABOP group; objectives default to one maximized objective on [0, 10]."""
    if objectives is None:
        objectives = [("o0", 0.0, 10.0, 0)]
    msg = {
        "type": "init",
        "config": {
            "numSamplingIterations": sampling,
            "numOptimizationIterations": optimization,
            "seed": 3,
            "nParameters": n_params,
            "nObjectives": len(objectives),
            "warmStart": False,
            "optimizerBackend": "cabop",
            "cabopObjectiveMode": mode,
            "cabopUseCostAwareAcquisition": True,
            "cabopUpdateRule": "actual",
        },
        "parameters": [
            {"key": f"p{i}", "init": {"low": param_range[0], "high": param_range[1]}, "group": "g",
             "tolerance": 0.05, "prefabValues": []}
            for i in range(n_params)
        ],
        "objectives": [
            {"key": key, "init": {"low": lo, "high": hi, "minimize": minimize}, "weight": 1.0}
            for key, lo, hi, minimize in objectives
        ],
        "user": {"userId": "u", "conditionId": "c", "groupId": "g"},
    }
    msg["config"].update(config)
    return msg


def run_session(runtime, replies, init_root=None):
    """run_cabop against scripted objective replies; returns (sent messages, {log name: rows})."""
    conn = FakeConn([json_line({"type": "objectives", "values": v}) for v in replies])
    logs = {}
    with tempfile.TemporaryDirectory() as tmp:
        env = {"BO_LOG_ROOT": tmp}
        if init_root is not None:
            env["BO_INIT_ROOT"] = init_root
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            runtime.run_cabop(conn)
        for path in pathlib.Path(runtime.PROJECT_PATH).glob("*.csv"):
            with open(path, newline="", encoding="utf-8") as f:
                logs[path.name] = list(csv.DictReader(f, delimiter=";"))
    sent = [json.loads(line) for chunk in conn.sent for line in chunk.decode("utf-8").splitlines() if line]
    return sent, logs


def write_warm_start(folder, params, objectives):
    """Warm-start CSVs x.csv / y.csv from {column: values} dicts."""
    import pandas as pd

    pd.DataFrame(params).to_csv(os.path.join(folder, "x.csv"), sep=";", index=False)
    pd.DataFrame(objectives).to_csv(os.path.join(folder, "y.csv"), sep=";", index=False)
    return {"warmStart": True, "initialParametersDataPath": "x.csv", "initialObjectivesDataPath": "y.csv"}


def one_param_space(lo, hi, tolerance=0.05):
    from cabop.bayesopt import BOSpace

    costs = {"g": {"unchanged": 1.0, "swapped": 10.0, "acquired": 100.0}}
    return BOSpace(parameters={
        "groups": ["g"], "cost": costs, "actual_cost": costs,
        "parameters": {"p": {"bound": np.asarray([lo, hi]), "tolerance": tolerance, "group": "g"}},
    })


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


@unittest.skipUnless(HAS_CABOP_DEPS, "scipy/scikit-learn/loguru not installed")
class CabopUnitSpaceTests(unittest.TestCase):
    """Reuse tolerance and soft costs are fractions of each range, not parameter units."""

    def test_soft_cost_is_the_same_on_every_parameter_range(self):
        from cabop.bayesopt import BayesOpt

        expected = None
        for lo, hi in [(0.0, 1.0), (0.0, 100.0), (0.0, 0.01), (-3.0, 7.0)]:
            with self.subTest(range=(lo, hi)):
                optimizer = BayesOpt(one_param_space(lo, hi), ifCost=True, random_state=0)
                middle = np.array([lo + 0.5 * (hi - lo)])
                optimizer.tell(middle, 0.5, middle)
                # Unit-space candidates: the last design, moves of 1%, 5% and 40% of the range.
                costs = optimizer._compute_costs(np.array([[0.5], [0.51], [0.55], [0.9]]))
                if expected is None:
                    expected = costs
                    # [0, 1] keeps its previous values: cheap near the last design, full price far away.
                    np.testing.assert_allclose(costs, [10.0, 10.09, 12.7, 100.0], atol=0.01)
                # Was: [0, 100] charged the full 100 for a 1% move, [0, 0.01] charged ~10 everywhere.
                np.testing.assert_allclose(costs, expected, rtol=1e-9)

    def test_tolerance_is_a_fraction_of_the_range(self):
        from cabop.bayesopt import BayesOpt

        # (range, last design, proposal, reused?) with tolerance 0.05 = 5% of the range.
        cases = [
            ((0.0, 100.0), 50.0, 52.0, True),    # 2% of the range: same prototype (was: acquired)
            ((0.0, 100.0), 50.0, 60.0, False),   # 10%: a new one
            ((0.0, 0.1), 0.05, 0.06, False),     # 10% (was: reused, as 0.01 < 0.05 units)
            ((0.0, 0.1), 0.05, 0.052, True),     # 2%
            ((0.0, 1.0), 0.5, 0.54, True),       # [0, 1] unchanged
            ((0.0, 1.0), 0.5, 0.56, False),
        ]
        for (lo, hi), last, proposal, reused in cases:
            with self.subTest(range=(lo, hi), proposal=proposal):
                optimizer = BayesOpt(one_param_space(lo, hi), ifCost=True, random_state=0)
                optimizer.tell(np.array([last]), 0.5, np.array([last]))
                costs, realized = optimizer.select_sample(np.array([proposal]))
                self.assertEqual(costs, (1.0,) if reused else (100.0,))
                self.assertEqual(float(realized[0]), last if reused else proposal)

    def test_runs_on_any_parameter_range_show_the_same_designs(self):
        from cabop.bayesopt import BayesOpt, BOSpace

        def run(lo, hi):
            costs = {"g": {"unchanged": 1.0, "swapped": 10.0, "acquired": 100.0}}
            params = {name: {"bound": np.asarray([lo, hi]), "tolerance": 0.05, "group": "g"} for name in "ab"}
            optimizer = BayesOpt(
                BOSpace(parameters={"groups": ["g"], "cost": costs, "actual_cost": costs, "parameters": params}),
                ifCost=True, random_state=3,
            )
            shown = []
            for i in range(10):
                x, _ = optimizer.ask(n_init=4)
                _, realized = optimizer.select_sample(x)
                u = (realized - lo) / (hi - lo)
                optimizer.tell(realized, float(np.sum((u - 0.7) ** 2) + 0.01 * np.sin(7 * i)), x)
                shown.append(u)
            return np.array(shown)

        reference = run(0.0, 1.0)
        for lo, hi in [(0.0, 0.1), (0.0, 100.0)]:
            with self.subTest(range=(lo, hi)):
                # Was: on [0, 0.1] every proposal was "within tolerance" and only 2 distinct
                # designs were ever shown.
                np.testing.assert_allclose(run(lo, hi), reference, atol=1e-5)

    def test_reported_expected_cost_is_the_cost_at_the_proposal(self):
        from cabop.bayesopt import BayesOpt

        optimizer = BayesOpt(one_param_space(0.0, 100.0), ifCost=True, random_state=0)
        optimizer.tell(np.array([50.0]), 0.5, np.array([50.0]))
        x, result = optimizer.ask(n_init=1)
        direct = optimizer.cost_model.smooth_cost(x, optimizer._numpy_to_design, optimizer._design_to_numpy)[0]
        # Was: smooth_cost received the unit-space coordinate (e.g. 0.54) as if it were 0.54 units.
        self.assertAlmostEqual(result.expected_cost, direct, places=9)
        self.assertLess(result.expected_cost, 100.0)

    def test_tolerance_outside_zero_one_is_rejected_at_init(self):
        for tolerance in (-0.1, 1.5, 5.0, "abc"):
            with self.subTest(tolerance=tolerance):
                msg = simple_init_msg(param_range=(0.0, 100.0))
                msg["parameters"][0]["tolerance"] = tolerance
                with self.assertRaisesRegex(ValueError, r"must lie in \[0, 1\] \(a fraction of its range"):
                    load_cabop_runtime().parse_init_and_validate(msg, forced_mode="single")

        msg = simple_init_msg()
        del msg["parameters"][0]["tolerance"]  # older senders: the default 5% of the range
        runtime = load_cabop_runtime()
        runtime.parse_init_and_validate(msg, forced_mode="single")
        self.assertEqual(runtime.build_cabop_space_dict()["parameters"]["p0"]["tolerance"], 0.05)

    def test_prefabricated_values_outside_the_bounds_are_rejected_at_init(self):
        msg = simple_init_msg()
        msg["parameters"][0]["prefabValues"] = [-0.1, 0.5, 1.0]
        with self.assertRaisesRegex(ValueError, r"prefabricated value\(s\) \[-0.1\] of parameter 'p0'"):
            load_cabop_runtime().parse_init_and_validate(msg, forced_mode="single")


@unittest.skipUnless(HAS_CABOP_DEPS, "scipy/scikit-learn/loguru not installed")
class CabopLogTests(unittest.TestCase):
    def test_observation_log_keeps_ten_significant_digits(self):
        from bo_normalize import round_for_log

        runtime = load_cabop_runtime()
        runtime.parse_init_and_validate(
            simple_init_msg(param_range=(0.0, 0.004), objectives=[("o0", 0.0, 0.004, 1)], optimization=0),
            forced_mode="single",
        )
        sent, logs = run_session(runtime, [{"o0": 0.0035}, {"o0": 0.000274}])
        rows = logs["ObservationsPerEvaluation.csv"]
        designs = [m["values"]["p0"] for m in sent if m["type"] == "parameters"]
        # Was: 3 decimals, so the designs and 0.000274 were logged as 0.0 or 0.004.
        self.assertEqual([float(r["p0"]) for r in rows], [round_for_log(v) for v in designs])
        self.assertEqual([float(r["o0"]) for r in rows], [0.0035, 0.000274])
        self.assertEqual([r["IsBest"] for r in rows], ["FALSE", "TRUE"])


@unittest.skipUnless(HAS_CABOP_DEPS, "scipy/scikit-learn/loguru not installed")
class CabopWarmStartTests(unittest.TestCase):
    def _run(self, warm_objectives, replies, sampling=3, mode="single", objectives=None):
        from cabop.bayesopt import BayesOpt

        with tempfile.TemporaryDirectory() as init_root:
            n_warm = len(next(iter(warm_objectives.values())))
            config = write_warm_start(init_root, {"p0": np.linspace(0.2, 0.8, n_warm)}, warm_objectives)
            runtime = load_cabop_runtime()
            runtime.parse_init_and_validate(
                simple_init_msg(mode=mode, sampling=sampling, optimization=len(replies),
                                objectives=objectives, **config),
                forced_mode=mode,
            )
            # Warm start replaces the sampling phase: no Sobol point may be drawn.
            with mock.patch.object(BayesOpt, "_sobol_sample", side_effect=AssertionError("Sobol draw")):
                return run_session(runtime, replies, init_root=init_root)

    def test_warm_start_skips_sampling_and_continues_the_iteration_axis(self):
        sent, logs = self._run({"o0": [9.0, 3.0]}, [{"o0": 5.0}, {"o0": 6.0}])
        iterations = [m["iteration"] for m in sent if m["type"] == "parameters"]
        # Was: iterations 1, 2 and the remaining sampling rounds drawn from the optimization budget.
        self.assertEqual(iterations, [3, 4])
        observations = logs["ObservationsPerEvaluation.csv"]
        self.assertEqual([int(r["Iteration"]) for r in observations], [3, 4])
        self.assertEqual([r["Phase"] for r in observations], ["optimization", "optimization"])
        self.assertEqual([int(r["Optimization"]) for r in logs["ExecutionTimes.csv"]], [3, 4])
        self.assertEqual([int(r["Iteration"]) for r in logs["CABOPMetricsPerEvaluation.csv"]], [3, 4])
        # Baseline at the last warm-start row, as bo.py writes it: best warm-start o0 = 9 of
        # [0, 10] maximized, i.e. scalarized 0.1 and coverage 0.9.
        metric = logs["BestObjectivePerEvaluation.csv"]
        self.assertEqual([int(r["Iteration"]) for r in metric], [2, 3, 4])
        self.assertAlmostEqual(float(metric[0]["BestObjective"]), 0.9)
        self.assertAlmostEqual([m["value"] for m in sent if m["type"] == "coverage"][0], 0.9)
        # The warm-start best (9.0) beats both live rows, so neither is IsBest (as in bo.py).
        self.assertEqual([r["IsBest"] for r in observations], ["FALSE", "FALSE"])

    def test_a_live_row_better_than_the_warm_start_is_best(self):
        _, logs = self._run({"o0": [4.0, 3.0]}, [{"o0": 5.0}, {"o0": 6.0}])
        self.assertEqual([r["IsBest"] for r in logs["ObservationsPerEvaluation.csv"]], ["FALSE", "TRUE"])

    def test_warm_start_rows_compete_for_is_pareto(self):
        # o0 maximized, o1 minimized. The warm-start row (8.5, 2) dominates the live (8, 2).
        objectives = [("o0", 0.0, 10.0, 0), ("o1", 0.0, 10.0, 1)]
        replies = [{"o0": 5.0, "o1": 5.0}, {"o0": 8.0, "o1": 2.0}, {"o0": 9.0, "o1": 6.0}]
        _, logs = self._run({"o0": [8.5], "o1": [2.0]}, replies, mode="multi", objectives=objectives)
        self.assertEqual([r["IsPareto"] for r in logs["ObservationsPerEvaluation.csv"]],
                         ["FALSE", "FALSE", "TRUE"])


@unittest.skipUnless(HAS_CABOP_DEPS, "scipy/scikit-learn/loguru not installed")
class CabopParetoTests(unittest.TestCase):
    def test_is_pareto_marks_every_non_dominated_row_once(self):
        runtime = load_cabop_runtime()
        runtime.parse_init_and_validate(
            simple_init_msg(mode="multi", sampling=4, optimization=0,
                            objectives=[("o0", 0.0, 10.0, 0), ("o1", 0.0, 10.0, 1)]),
            forced_mode="multi",
        )
        # o0 maximized, o1 minimized: (8, 2) dominates (5, 5); (9, 6) trades off against (8, 2);
        # the second (8, 2) is a duplicate.
        replies = [{"o0": 5.0, "o1": 5.0}, {"o0": 8.0, "o1": 2.0}, {"o0": 9.0, "o1": 6.0}, {"o0": 8.0, "o1": 2.0}]
        _, logs = run_session(runtime, replies)
        # Was: TRUE only for the rows with the lowest weighted score -- both copies of (8, 2) --
        # so FinalDesignSelector never saw the (9, 6) trade-off.
        self.assertEqual([r["IsPareto"] for r in logs["ObservationsPerEvaluation.csv"]],
                         ["FALSE", "TRUE", "TRUE", "FALSE"])

    def test_non_dominated_mask_matches_mobo_semantics(self):
        # The installed moocore, not the stub other test modules leave in sys.modules.
        saved = sys.modules.pop("moocore", None)
        try:
            moocore = importlib.import_module("moocore")
        except ImportError:
            self.skipTest("moocore not installed")
        finally:
            if saved is not None:
                sys.modules["moocore"] = saved
        runtime = load_cabop_runtime()
        rng = np.random.default_rng(0)
        for _ in range(300):
            # Likert-like ratings: many ties and duplicates.
            values = rng.integers(1, 6, size=(int(rng.integers(1, 15)), int(rng.integers(2, 4)))).astype(float)
            first_copy = np.zeros(len(values), dtype=bool)
            seen = set()
            for i, row in enumerate(values):
                first_copy[i] = tuple(row) not in seen
                seen.add(tuple(row))
            expected = moocore.is_nondominated(values, maximise=True, keep_weakly=True) & first_copy
            np.testing.assert_array_equal(runtime.non_dominated_mask(values), expected, err_msg=str(values))


if __name__ == "__main__":
    unittest.main()
