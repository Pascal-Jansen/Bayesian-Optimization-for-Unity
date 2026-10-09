import csv
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
import uuid
from unittest import mock

import numpy as np
import pandas as pd

import sys

# Support both `discover tests` (tests/ on sys.path) and direct module runs.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import (  # noqa: E402
    FakeConn as _FakeConn,
    FakeServerSocket as _FakeServerSocket,
    FakeTensor,
    assert_hardened_listener,
    install_stub_modules,
    json_line as _json_line,
    reset_protocol_state,
    run_main_recording_listener,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BO_PATH = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization/bo.py"


def load_bo_module():
    install_stub_modules()
    name = f"bo_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reset_protocol_state()
    return module


def legacy_inline_objective(name, raw, lo, hi, minflag):
    """bo.py/mobo.py's inline objective transform before they called bo_normalize (reference)."""
    try:
        val = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Objective '{name}' must be numeric, got {raw!r}") from e
    if not np.isfinite(val):
        raise ValueError(f"Objective '{name}' is non-finite: {val}")
    eps = 1e-9
    if hi == lo:
        if not np.isclose(val, lo, rtol=0.0, atol=eps):
            raise ValueError(f"Objective '{name}' value {val} is out of bounds for degenerate interval [{lo}, {hi}]")
        f = 0.0
    else:
        if val < (lo - eps) or val > (hi + eps):
            raise ValueError(f"Objective '{name}' value {val} is out of bounds [{lo}, {hi}]")
        f = (val - lo) / (hi - lo) * 2 - 1
    if int(minflag) == 1:
        f *= -1
    return float(np.clip(f, -1.0, 1.0))


# (raw value, lo, hi): bounds, interior, a hair outside (within 1e-9), out of bounds, degenerate,
# non-numeric and non-finite payloads.
OBJECTIVE_TRANSFORM_CASES = [
    (0.0, 0.0, 10.0), (10.0, 0.0, 10.0), (2.5, 0.0, 10.0), (7.0, 1.0, 9.0), (-3.0, -5.0, 5.0),
    (10.0 + 5e-10, 0.0, 10.0), (-5e-10, 0.0, 10.0), (0.7, 0.1, 0.7), (11.0, 0.0, 10.0),
    (-0.1, 0.0, 10.0), (3.0, 3.0, 3.0), (3.0 + 5e-10, 3.0, 3.0), (3.5, 3.0, 3.0),
    ("abc", 0.0, 1.0), (None, 0.0, 1.0), ("inf", 0.0, 1.0), (float("nan"), 0.0, 1.0),
]


def legacy_outcome(name, raw, lo, hi, minflag):
    try:
        return ("value", legacy_inline_objective(name, raw, lo, hi, minflag))
    except ValueError as e:
        return ("error", str(e))


class BoTests(unittest.TestCase):
    def _base_init(self):
        return {
            "type": "init",
            "config": {
                "numSamplingIterations": 2,
                "numOptimizationIterations": 1,
                "batchSize": 1,
                "numRestarts": 3,
                "rawSamples": 16,
                "mcSamples": 8,
                "seed": 5,
                "nParameters": 1,
                "nObjectives": 1,
                "warmStart": False,
                "initialParametersDataPath": "",
                "initialObjectivesDataPath": "",
            },
            "parameters": [{"key": "p0", "init": {"low": 0.0, "high": 1.0}}],
            "objectives": [{"key": "o0", "init": {"low": 0.0, "high": 1.0, "minimize": 0}}],
            "user": {"userId": "u", "conditionId": "c", "groupId": "g"},
        }

    def _run_main_with_init(self, bo, init_msg, execute_stub=None, accept_error=None):
        conn = _FakeConn([_json_line(init_msg)])
        fake_server = _FakeServerSocket(conn, accept_error=accept_error)
        called = {}

        def _default_execute(conn_arg, seed, iterations, initial_samples):
            called["args"] = (conn_arg, seed, iterations, initial_samples)
            return [], FakeTensor([[0.0]]), FakeTensor([[0.0]])

        original_socket_ctor = bo.socket.socket
        original_execute = bo.bo_execute
        try:
            bo.socket.socket = lambda *args, **kwargs: fake_server
            bo.bo_execute = execute_stub if execute_stub is not None else _default_execute
            bo.main()
        finally:
            bo.socket.socket = original_socket_ctor
            bo.bo_execute = original_execute
        return conn, fake_server, called

    def test_parse_init_missing_keys_raises(self):
        bo = load_bo_module()
        with self.assertRaises(ValueError):
            bo.parse_param_init({"low": 0.0})
        with self.assertRaises(ValueError):
            bo.parse_obj_init({"low": 0.0, "high": 1.0})

    def test_send_json_line_disconnect_raises_connection_error(self):
        bo = load_bo_module()
        conn = _FakeConn([], send_error=BrokenPipeError("pipe closed"))
        with self.assertRaises(ConnectionError):
            bo.send_json_line(conn, {"type": "coverage", "value": 1.0})

    def test_recv_objectives_blocking_preserves_buffer_across_calls(self):
        bo = load_bo_module()
        conn = _FakeConn(
            [
                (
                    json.dumps({"type": "log", "message": "a"})
                    + "\n"
                    + json.dumps({"type": "objectives", "values": {"o0": 0.1}})
                    + "\n"
                    + json.dumps({"type": "objectives", "values": {"o0": 0.2}})
                    + "\n"
                ).encode("utf-8")
            ]
        )

        first = bo.recv_objectives_blocking(conn)
        second = bo.recv_objectives_blocking(conn)
        self.assertEqual(first, {"o0": 0.1})
        self.assertEqual(second, {"o0": 0.2})

    def test_objective_function_missing_objective_key_raises(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"wrong": 0.2}})])

        with self.assertRaises(KeyError):
            bo.objective_function(conn, FakeTensor([0.2]), 1)

    def test_objective_function_non_numeric_objective_raises(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": "abc"}})])

        with self.assertRaises(ValueError):
            bo.objective_function(conn, FakeTensor([0.2]), 1)

    def test_objective_function_non_finite_objective_raises(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": "inf"}})])

        with self.assertRaises(ValueError):
            bo.objective_function(conn, FakeTensor([0.2]), 1)

    def test_objective_function_out_of_bounds_raises(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": 2.0}})])

        with self.assertRaises(ValueError):
            bo.objective_function(conn, FakeTensor([0.2]), 1)

    def test_objective_function_preserves_parameter_precision(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": 0.5}})])

        bo.objective_function(conn, FakeTensor([0.123456789]), 1)
        sent = json.loads(conn.sent[0].decode("utf-8"))
        self.assertAlmostEqual(sent["values"]["p0"], 0.123456789, places=9)

    def test_objective_function_invalid_values_payload_type_raises(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        conn = _FakeConn([_json_line({"type": "objectives", "values": [1.0]})])

        with self.assertRaisesRegex(RuntimeError, "non-dict 'values'"):
            bo.objective_function(conn, FakeTensor([0.2]), 1)

    def test_objective_function_matches_former_inline_transform(self):
        # objective_function now calls bo_normalize.normalize_objective_value (like DBO and
        # MetaTAF) instead of its own copy: same values, same errors.
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        for raw, lo, hi in OBJECTIVE_TRANSFORM_CASES:
            for minflag in (0, 1):
                with self.subTest(raw=raw, lo=lo, hi=hi, minflag=minflag):
                    bo.objectives_info = [(lo, hi, minflag)]
                    reset_protocol_state()
                    conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": raw}})])
                    try:
                        got = ("value", bo.objective_function(conn, FakeTensor([0.5]), 1).item())
                    except ValueError as e:
                        got = ("error", str(e))
                    self.assertEqual(got, legacy_outcome("o0", raw, lo, hi, minflag))

        bo.objectives_info = [(0.0, 10.0, 1)]
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": 2.5}})])
        with mock.patch.object(bo.bo_normalize, "normalize_objective_value",
                               wraps=bo.bo_normalize.normalize_objective_value) as shared:
            bo.objective_function(conn, FakeTensor([0.5]), 1)
        shared.assert_called_once_with(2.5, 0.0, 10.0, 1, name="o0")

    def test_acquisition_keeps_batch_limit_five(self):
        bo = load_bo_module()
        captured = {}

        def fake_optimize_acqf(acq_function, bounds, q, num_restarts, raw_samples, options, sequential):
            captured.update(options=options, num_restarts=num_restarts)
            return FakeTensor([[0.5]]), None

        bo.optimize_acqf = fake_optimize_acqf
        bo.qLogNoisyExpectedImprovement = lambda **kwargs: object()
        bo.NUM_RESTARTS, bo.RAW_SAMPLES, bo.BATCH_SIZE = 7, 64, 1
        bo.problem_bounds = FakeTensor([[0.0], [1.0]])
        bo.optimize_candidates(model=None, sampler=None, X_baseline=FakeTensor([[0.1]]))
        self.assertEqual(captured["num_restarts"], 7)
        # batch_limit = restarts changed the local optimum in ~1% of problems for < 0.1 s; keep 5.
        self.assertEqual(captured["options"], {"batch_limit": 5, "init_batch_limit": 128, "maxiter": 200})

    def test_mc_samples_fallback_matches_unity_default(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        del init_msg["config"]["mcSamples"]
        self._run_main_with_init(bo, init_msg)
        self.assertEqual(bo.MC_SAMPLES, 128)

    def test_load_data_normalizes_and_validates(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)

            import pandas as pd
            pd.DataFrame({"p0": [5.0]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o0": [20.0]}).to_csv(init_dir / "objs.csv", sep=";", index=False)

            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                bo.CSV_PATH_PARAMETERS = "params.csv"
                bo.CSV_PATH_OBJECTIVES = "objs.csv"
                bo.parameter_names = ["p0"]
                bo.objective_names = ["o0"]
                bo.parameters_info = [(0.0, 10.0)]
                bo.objectives_info = [(0.0, 100.0, 0)]
                bo.PROBLEM_DIM = 1
                bo.NUM_OBJS = 1

                x, y = bo.load_data()
            finally:
                os.chdir(prev_cwd)

        np.testing.assert_allclose(x.numpy(), np.array([[0.5]]), atol=1e-12)
        np.testing.assert_allclose(y.numpy(), np.array([[-0.6]]), atol=1e-12)

    def test_load_data_normalized_native_flips_min_objective(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({"p0": [5.0]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o_min": [0.5]}).to_csv(init_dir / "objs.csv", sep=";", index=False)

            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                bo.CSV_PATH_PARAMETERS = "params.csv"
                bo.CSV_PATH_OBJECTIVES = "objs.csv"
                bo.parameter_names = ["p0"]
                bo.objective_names = ["o_min"]
                bo.parameters_info = [(0.0, 10.0)]
                bo.objectives_info = [(0.0, 100.0, 1)]
                bo.PROBLEM_DIM = 1
                bo.NUM_OBJS = 1
                bo.WARM_START_OBJECTIVE_FORMAT = "normalized_native"

                x, y = bo.load_data()
            finally:
                os.chdir(prev_cwd)

        np.testing.assert_allclose(x.numpy(), np.array([[0.5]]), atol=1e-12)
        np.testing.assert_allclose(y.numpy(), np.array([[-0.5]]), atol=1e-12)

    def test_normalize_obj_column_modes(self):
        bo = load_bo_module()
        col = np.array([0.5, -0.5])
        lo, hi, minflag = (0.0, 10.0, 1)

        bo.WARM_START_OBJECTIVE_FORMAT = "normalized_max"
        y_max = bo.normalize_obj_column(col, lo, hi, minflag)
        np.testing.assert_allclose(y_max, np.array([0.5, -0.5]), atol=1e-12)

        bo.WARM_START_OBJECTIVE_FORMAT = "normalized_native"
        y_native = bo.normalize_obj_column(col, lo, hi, minflag)
        np.testing.assert_allclose(y_native, np.array([-0.5, 0.5]), atol=1e-12)

        bo.WARM_START_OBJECTIVE_FORMAT = "raw"
        y_raw = bo.normalize_obj_column(np.array([2.0]), lo, hi, minflag)
        np.testing.assert_allclose(y_raw, np.array([0.6]), atol=1e-12)

    def test_load_data_missing_columns_raises(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)

            import pandas as pd
            pd.DataFrame({"wrong_param": [5.0]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o0": [20.0]}).to_csv(init_dir / "objs.csv", sep=";", index=False)

            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                bo.CSV_PATH_PARAMETERS = "params.csv"
                bo.CSV_PATH_OBJECTIVES = "objs.csv"
                bo.parameter_names = ["p0"]
                bo.objective_names = ["o0"]
                bo.parameters_info = [(0.0, 10.0)]
                bo.objectives_info = [(0.0, 100.0, 0)]
                bo.PROBLEM_DIM = 1
                bo.NUM_OBJS = 1

                with self.assertRaises(ValueError):
                    bo.load_data()
            finally:
                os.chdir(prev_cwd)

    def test_load_data_out_of_bounds_values_raises(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)

            import pandas as pd
            pd.DataFrame({"p0": [50.0]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o0": [20.0]}).to_csv(init_dir / "objs.csv", sep=";", index=False)

            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                bo.CSV_PATH_PARAMETERS = "params.csv"
                bo.CSV_PATH_OBJECTIVES = "objs.csv"
                bo.parameter_names = ["p0"]
                bo.objective_names = ["o0"]
                bo.parameters_info = [(0.0, 10.0)]
                bo.objectives_info = [(0.0, 100.0, 0)]
                bo.PROBLEM_DIM = 1
                bo.NUM_OBJS = 1

                with self.assertRaises(ValueError):
                    bo.load_data()
            finally:
                os.chdir(prev_cwd)

    def test_generate_initial_data_rejects_non_positive_n(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            with self.assertRaises(ValueError):
                bo.generate_initial_data(conn=None, n_samples=0)

    def test_save_xy_rejects_corrupt_observation_schema(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            bo.USER_ID = "u"
            bo.CONDITION_ID = "c"
            bo.GROUP_ID = "g"
            bo.N_INITIAL = 2
            bo.PROBLEM_DIM = 1
            bo.parameter_names = ["p0"]
            bo.objective_names = ["o0"]
            bo.parameters_info = [(0.0, 10.0)]
            bo.objectives_info = [(0.0, 100.0, 0)]

            obs = pathlib.Path(tmp) / "ObservationsPerEvaluation.csv"
            obs.write_text(
                "wrong;columns\nx;y\n",
                encoding="utf-8",
            )
            x_sample = FakeTensor([[0.1], [0.2]])
            y_sample = FakeTensor([[-0.5], [0.8]])

            with self.assertRaises(ValueError):
                bo.save_xy(x_sample, y_sample, iteration=1)

    def test_save_xy_uses_row_count_for_iteration(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            bo.USER_ID = "u"
            bo.CONDITION_ID = "c"
            bo.GROUP_ID = "g"
            bo.N_INITIAL = 99
            bo.PROBLEM_DIM = 1
            bo.parameter_names = ["p0"]
            bo.objective_names = ["o0"]
            bo.parameters_info = [(0.0, 10.0)]
            bo.objectives_info = [(0.0, 100.0, 0)]

            x_sample = FakeTensor([[0.1], [0.2]])
            y_sample = FakeTensor([[-0.5], [0.8]])
            bo.save_xy(x_sample, y_sample, iteration=1)

            df = pd.read_csv(pathlib.Path(tmp) / "ObservationsPerEvaluation.csv", delimiter=";")
            self.assertEqual(int(df.iloc[-1]["Iteration"]), 2)

    def test_save_metric_to_file_writes_bestobjective_header_once(self):
        bo = load_bo_module()
        bo.objectives_info = [(0.0, 1.0, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            bo.save_metric_to_file([0.1], iteration=1)
            bo.save_metric_to_file([0.2], iteration=2)
            p_best = pathlib.Path(tmp) / "BestObjectivePerEvaluation.csv"
            lines_best = p_best.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines_best[0], "BestObjective;Iteration;Scale;BestObjectiveRaw")
            self.assertEqual(len(lines_best), 3)

            p_legacy = pathlib.Path(tmp) / "HypervolumePerEvaluation.csv"
            lines_legacy = p_legacy.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines_legacy[0], "Hypervolume;Iteration;Scale;BestObjectiveRaw")
            self.assertEqual(len(lines_legacy), 3)

    def _metric_rows(self, tmp, filename):
        with open(pathlib.Path(tmp) / filename, newline="", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter=";"))

    def test_metric_logs_carry_raw_best_objective_and_direction(self):
        # BestObjective stays the normalized maximize-space value (sign-flipped for a minimized
        # objective); BestObjectiveRaw is the same observation in the objective's own units.
        bo = load_bo_module()
        bo.objectives_info = [(0.0, 10.0, 1)]  # task time in s, smaller is better
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            best_norm = bo.bo_normalize.normalize_objective_value(2.5, 0.0, 10.0, 1)
            bo.save_metric_to_file([best_norm], iteration=4)
            for filename in ("BestObjectivePerEvaluation.csv", "HypervolumePerEvaluation.csv"):
                header, row = self._metric_rows(tmp, filename)
                self.assertEqual(header[2:], ["Scale", "BestObjectiveRaw"])
                self.assertAlmostEqual(float(row[0]), 0.5)  # (2.5 on [0,10] -> -0.5) sign-flipped
                self.assertEqual(row[1], "4")
                self.assertIn("minimized", row[2])
                self.assertEqual(float(row[3]), 2.5)

        bo.objectives_info = [(0.0, 10.0, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            bo.save_metric_to_file([0.5], iteration=1)
            _, row = self._metric_rows(tmp, "BestObjectivePerEvaluation.csv")
            self.assertIn("maximized", row[2])
            self.assertEqual(float(row[3]), 7.5)

    def test_metric_raw_column_is_empty_without_current_context_observations(self):
        bo = load_bo_module()
        bo.objectives_info = [(0.0, 10.0, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            bo.save_metric_to_file([-1.0], iteration=0, observed=False)
            _, row = self._metric_rows(tmp, "BestObjectivePerEvaluation.csv")
        self.assertEqual(row[:2], ["-1.0", "0"])
        self.assertEqual(row[3], "")

    def test_bo_execute_metric_raw_column_tracks_best_observation(self):
        bo = load_bo_module()
        bo.objectives_info = [(0.0, 1.0, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            # Stub values 0.2, 0.8, then 0.5, 0.5: minimized, so the best raw value stays 0.2.
            self._run_bo_execute(bo, tmp, iterations=2, keep_objectives_info=True)
            rows = self._metric_rows(pathlib.Path(tmp) / "LogData" / "ids" / "ids" / "run",
                                     "BestObjectivePerEvaluation.csv")
        self.assertEqual([r[1] for r in rows[1:]], ["1", "2", "3", "4"])
        self.assertEqual([float(r[3]) for r in rows[1:]], [0.2, 0.2, 0.2, 0.2])
        self.assertTrue(all(abs(float(r[0]) - 0.6) < 1e-12 for r in rows[1:]))

    def test_bo_execute_logs_metric_for_every_evaluation(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                bo.USER_ID = bo.USER_LOG_ID = "u"
                bo.CONDITION_ID = bo.CONDITION_LOG_ID = "c"
                bo.GROUP_ID = "g"
                bo.WARM_START = False
                bo.SEED = 3
                bo.PROBLEM_DIM = 1
                bo.NUM_OBJS = 1
                bo.BATCH_SIZE = 1
                bo.NUM_RESTARTS = 2
                bo.RAW_SAMPLES = 16
                bo.MC_SAMPLES = 8
                bo.parameter_names = ["p0"]
                bo.objective_names = ["o0"]
                bo.parameters_info = [(0.0, 1.0)]
                bo.objectives_info = [(0.0, 1.0, 0)]
                bo.problem_bounds = FakeTensor([[0.0], [1.0]])
                bo.SobolQMCNormalSampler = lambda sample_shape, seed: {
                    "shape": sample_shape, "seed": seed
                }
                # Initial samples (n=2, q=1, d=1): x=0.2 then x=0.8
                bo.draw_sobol_samples = lambda bounds, n, q, seed: FakeTensor(
                    [[[0.2]], [[0.8]]]
                )
                bo.optimize_candidates = lambda model, sampler, X_baseline: FakeTensor([[0.4]])

                conn = _FakeConn([
                    _json_line({"type": "objectives", "values": {"o0": 0.2}}),
                    _json_line({"type": "objectives", "values": {"o0": 0.8}}),
                    _json_line({"type": "objectives", "values": {"o0": 0.5}}),
                ])
                original_send = bo.send_json_line
                bo.send_json_line = lambda c, payload: None
                try:
                    metric_values, _, _ = bo.bo_execute(
                        conn=conn, seed=3, iterations=1, initial_samples=2
                    )
                finally:
                    bo.send_json_line = original_send
            finally:
                os.chdir(prev_cwd)

            with open(
                pathlib.Path(tmp) / "LogData" / "u" / "c" / "run"
                / "HypervolumePerEvaluation.csv",
                newline="",
            ) as f:
                rows = list(csv.reader(f, delimiter=";"))

        self.assertEqual(len(metric_values), 3)
        self.assertEqual([row[1] for row in rows[1:]], ["1", "2", "3"])

    def _run_bo_execute(self, bo, tmp, iterations, user_ids=("u", "c", "g"), keep_objectives_info=False):
        """bo_execute against stubs: Sobol draws x=0.2, 0.8; candidates x=0.4."""
        sobol_calls = []
        bo.USER_ID, bo.CONDITION_ID, bo.GROUP_ID = user_ids
        bo.USER_LOG_ID = bo.CONDITION_LOG_ID = "ids"
        bo.WARM_START = False
        bo.SEED = 3
        bo.PROBLEM_DIM = 1
        bo.NUM_OBJS = 1
        bo.BATCH_SIZE = 1
        bo.NUM_RESTARTS = 2
        bo.RAW_SAMPLES = 16
        bo.MC_SAMPLES = 8
        bo.parameter_names = ["p0"]
        bo.objective_names = ["o0"]
        bo.parameters_info = [(0.0, 1.0)]
        if not keep_objectives_info:
            bo.objectives_info = [(0.0, 1.0, 0)]
        bo.problem_bounds = FakeTensor([[0.0], [1.0]])
        bo.SobolQMCNormalSampler = lambda sample_shape, seed: {"shape": sample_shape, "seed": seed}

        def sobol(bounds, n, q, seed):
            sobol_calls.append((n, q))
            return FakeTensor([[[0.2]], [[0.8]]])

        bo.draw_sobol_samples = sobol
        bo.optimize_candidates = lambda model, sampler, X_baseline: FakeTensor([[0.4]])
        values = [0.2, 0.8] + [0.5] * iterations
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": v}}) for v in values])
        prev_cwd = os.getcwd()
        original_send = bo.send_json_line
        try:
            os.chdir(tmp)
            bo.send_json_line = lambda c, payload: None
            bo.bo_execute(conn=conn, seed=3, iterations=iterations, initial_samples=2)
        finally:
            bo.send_json_line = original_send
            os.chdir(prev_cwd)
        return sobol_calls

    def test_stop_during_sampling_finalizes_isbest_flags(self):
        # A perfect-rating stop in the sampling phase used to leave the provisional best-so-far
        # flags (2, 5, 8 -> TRUE, TRUE, TRUE) in the log, which FinalDesignSelector reads.
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            bo.USER_ID, bo.CONDITION_ID, bo.GROUP_ID = "u", "c", "g"
            bo.SEED = 3
            bo.PROBLEM_DIM = 1
            bo.parameter_names = ["p0"]
            bo.objective_names = ["o0"]
            bo.parameters_info = [(0.0, 1.0)]
            bo.objectives_info = [(0.0, 10.0, 0)]
            bo.problem_bounds = FakeTensor([[0.0], [1.0]])
            bo.draw_sobol_samples = lambda bounds, n, q, seed: FakeTensor([[[0.1]], [[0.2]], [[0.3]], [[0.4]]])
            conn = _FakeConn(
                [_json_line({"type": "objectives", "values": {"o0": v}}) for v in (2.0, 5.0, 8.0)]
                + [_json_line({"type": "stop", "reason": "perfect_rating"})]
            )
            with mock.patch.object(bo, "send_json_line", lambda c, payload: None):
                with self.assertRaises(bo.StopRequested):
                    bo.generate_initial_data(conn, n_samples=4, metric_values=[])
            with open(pathlib.Path(tmp) / "ObservationsPerEvaluation.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter=";"))
        self.assertEqual([(r["o0"], r["IsBest"]) for r in rows],
                         [("2.0", "FALSE"), ("5.0", "FALSE"), ("8.0", "TRUE")])

    def test_initial_design_is_n_points_of_one_sobol_sequence(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            calls = self._run_bo_execute(bo, tmp, iterations=1)
        # n=N, q=1: N points of a d-dimensional sequence, not one point of an N*d one.
        self.assertEqual(calls, [(2, 1)])

    def test_rewritten_log_keeps_ids_as_sent(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            self._run_bo_execute(bo, tmp, iterations=2, user_ids=("007", "01", "NA"))
            obs = pathlib.Path(tmp) / "LogData" / "ids" / "ids" / "run" / "ObservationsPerEvaluation.csv"
            with open(obs, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f, delimiter=";"))[1:]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r[:3] == ["007", "01", "NA"] for r in rows), rows)

    def test_objectives_decode_utf8_split_across_chunks(self):
        bo = load_bo_module()
        payload = '{"type":"objectives","values":{"Größe":1.0}}\n'.encode("utf-8")
        cut = payload.index("ö".encode("utf-8")) + 1  # inside the two-byte character
        conn = _FakeConn([payload[:cut], payload[cut:]])
        values = bo.recv_objectives_blocking(conn)
        self.assertEqual(list(values), ["Größe"])

    def test_main_pins_torch_threads_from_environment(self):
        bo = load_bo_module()
        calls = []
        bo.torch.set_num_threads = calls.append
        self.assertEqual(bo.TORCH_THREADS, 1)  # single-threaded unless BO_TORCH_THREADS says otherwise
        bo.TORCH_THREADS = 3
        self._run_main_with_init(bo, self._base_init(), execute_stub=lambda *args, **kwargs: None)
        self.assertEqual(calls, [3])

        bo.TORCH_THREADS = 0
        with self.assertRaisesRegex(ValueError, "BO_TORCH_THREADS"):
            self._run_main_with_init(bo, self._base_init(), execute_stub=lambda *args, **kwargs: None)

    def test_main_listens_on_loopback_only_and_stops_after_connect(self):
        bo = load_bo_module()
        server, listening = run_main_recording_listener(bo, self._base_init(), "bo_execute")
        assert_hardened_listener(self, bo.socket, server, listening)

    def test_main_explains_a_taken_port(self):
        bo = load_bo_module()

        class _TakenPortServer(_FakeServerSocket):
            def bind(self, addr):
                raise OSError(10048, "Only one usage of each socket address is normally permitted")

        server = _TakenPortServer(_FakeConn([]))
        original_socket_ctor = bo.socket.socket
        try:
            bo.socket.socket = lambda *args, **kwargs: server
            with self.assertRaisesRegex(OSError, "already in use.*left over"):
                bo.main()
        finally:
            bo.socket.socket = original_socket_ctor
        self.assertTrue(server.closed)

    def test_main_rejects_keys_colliding_with_log_columns(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["parameters"][0]["key"] = "Phase"
        with self.assertRaisesRegex(ValueError, "collide with log columns"):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_csv_helpers_survive_a_locked_log_and_catch_up(self):
        bo = load_bo_module()
        protocol = reset_protocol_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ExecutionTimes.csv")
            with mock.patch.object(protocol, "LOG_WRITE_RETRY_SEC", 0.0), \
                    mock.patch("builtins.open", side_effect=PermissionError(13, "locked by Excel")), \
                    mock.patch("builtins.print"):
                bo.create_csv_file(path, ["A"])  # no exception: the study goes on
                bo.write_data_to_csv(path, ["A"], [{"A": 1}])
            self.assertFalse(os.path.exists(path))
            with mock.patch("builtins.print"):
                bo.write_data_to_csv(path, ["A"], [{"A": 2}])  # lock gone: everything is written
            with open(path, newline="", encoding="utf-8") as f:
                self.assertEqual(list(csv.reader(f, delimiter=";")), [["A"], ["1"], ["2"]])
        self.assertEqual(protocol.unsaved_logs(), [])

    def test_main_rejects_missing_required_nparameters(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        del init_msg["config"]["nParameters"]
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_closes_socket_and_conn_on_init_validation_error(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        del init_msg["config"]["nParameters"]
        conn = _FakeConn([_json_line(init_msg)])
        fake_server = _FakeServerSocket(conn)

        original_socket_ctor = bo.socket.socket
        original_execute = bo.bo_execute
        try:
            bo.socket.socket = lambda *args, **kwargs: fake_server
            bo.bo_execute = lambda *args, **kwargs: None
            with self.assertRaises(ValueError):
                bo.main()
        finally:
            bo.socket.socket = original_socket_ctor
            bo.bo_execute = original_execute

        self.assertTrue(conn.shutdown_called)
        self.assertTrue(conn.closed)
        self.assertTrue(fake_server.closed)

    def test_main_rejects_missing_required_nobjectives(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        del init_msg["config"]["nObjectives"]
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_duplicate_parameter_keys(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["config"]["nParameters"] = 2
        init_msg["parameters"] = [
            {"key": "p0", "init": {"low": 0.0, "high": 1.0}},
            {"key": "p0", "init": {"low": 0.0, "high": 2.0}},
        ]
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_duplicate_objective_keys(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["config"]["nObjectives"] = 1
        init_msg["objectives"] = [
            {"key": "o0", "init": {"low": 0.0, "high": 1.0, "minimize": 0}},
            {"key": "o0", "init": {"low": 0.0, "high": 2.0, "minimize": 0}},
        ]
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_invalid_parameter_bounds(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["parameters"][0]["init"]["low"] = 2.0
        init_msg["parameters"][0]["init"]["high"] = 1.0
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_non_finite_parameter_bounds(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["parameters"][0]["init"]["low"] = "nan"
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_invalid_objective_bounds(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["objectives"][0]["init"]["low"] = 2.0
        init_msg["objectives"][0]["init"]["high"] = 1.0
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_non_finite_objective_bounds(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["objectives"][0]["init"]["high"] = "inf"
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_invalid_minimize_flag(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["objectives"][0]["init"]["minimize"] = 3
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_missing_objective_minimize_key(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        del init_msg["objectives"][0]["init"]["minimize"]
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_negative_iteration_counts(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["config"]["numSamplingIterations"] = -1
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_non_positive_optimizer_hyperparams(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["config"]["rawSamples"] = 0
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_invalid_warm_start_objective_format(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        init_msg["config"]["warmStartObjectiveFormat"] = "invalid-mode"
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_rejects_non_positive_socket_timeout(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        bo.SOCKET_TIMEOUT_SEC = 0
        with self.assertRaises(ValueError):
            self._run_main_with_init(bo, init_msg, execute_stub=lambda *args, **kwargs: None)

    def test_main_accept_timeout_raises(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        with self.assertRaises(TimeoutError):
            self._run_main_with_init(
                bo,
                init_msg,
                execute_stub=lambda *args, **kwargs: None,
                accept_error=bo.socket.timeout("accept timeout"),
            )

    def test_main_sets_conn_timeout_and_calls_execute(self):
        bo = load_bo_module()
        init_msg = self._base_init()
        conn, fake_server, called = self._run_main_with_init(bo, init_msg)

        self.assertIn("args", called)
        self.assertEqual(called["args"][1:], (5, 1, 2))
        self.assertEqual(conn.timeout, bo.SOCKET_TIMEOUT_SEC)
        self.assertTrue(conn.shutdown_called)
        self.assertTrue(conn.closed)
        self.assertTrue(fake_server.closed)

    def test_save_xy_updates_tail_isbest_on_mismatch(self):
        bo = load_bo_module()
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            bo.USER_ID = "u"
            bo.CONDITION_ID = "c"
            bo.GROUP_ID = "g"
            bo.N_INITIAL = 2
            bo.PROBLEM_DIM = 1
            bo.parameter_names = ["p0"]
            bo.objective_names = ["o0"]
            bo.parameters_info = [(0.0, 10.0)]
            bo.objectives_info = [(0.0, 100.0, 0)]

            obs = pathlib.Path(tmp) / "ObservationsPerEvaluation.csv"
            obs.write_text(
                "UserID;ConditionID;GroupID;Timestamp;Iteration;Phase;IsBest;o0;p0\n"
                "u;c;g;2026-01-01 00:00:00;1;sampling;FALSE;10.0;1.0\n",
                encoding="utf-8",
            )

            x_sample = FakeTensor([[0.1], [0.2]])
            y_sample = FakeTensor([[-0.5], [0.8]])
            bo.save_xy(x_sample, y_sample, iteration=1)

            lines = obs.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            # row0 remains untouched due mismatch fallback, row1 updated for current run tail.
            self.assertIn(";FALSE;", lines[1])
            self.assertIn(";TRUE;", lines[2])


if __name__ == "__main__":
    unittest.main()
