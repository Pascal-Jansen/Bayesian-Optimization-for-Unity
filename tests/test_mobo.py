import csv
import json
import importlib.util
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
from test_bo import OBJECTIVE_TRANSFORM_CASES, legacy_outcome  # noqa: E402


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MOBO_PATH = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization/mobo.py"


def load_mobo_module():
    install_stub_modules()
    name = f"mobo_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, MOBO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reset_protocol_state()
    return module


class MoboTests(unittest.TestCase):
    def _base_init_message(self):
        return {
            "type": "init",
            "config": {
                "numSamplingIterations": 2,
                "numOptimizationIterations": 1,
                "batchSize": 1,
                "numRestarts": 3,
                "rawSamples": 16,
                "mcSamples": 8,
                "seed": 11,
                "nParameters": 1,
                "nObjectives": 2,
                "warmStart": False,
                "warmStartObjectiveFormat": "auto",
                "initialParametersDataPath": "",
                "initialObjectivesDataPath": "",
            },
            "parameters": [{"key": "p0", "init": {"low": 0.0, "high": 1.0}}],
            "objectives": [
                {"key": "o0", "init": {"low": 0.0, "high": 10.0, "minimize": 1}},
                {"key": "o1", "init": {"low": 0.0, "high": 10.0, "minimize": 0}},
            ],
            "user": {"userId": "u", "conditionId": "c", "groupId": "g"},
        }

    def _run_main_with_init(self, mobo, init_msg, execute_stub=None, accept_error=None):
        conn = _FakeConn([_json_line(init_msg)])
        fake_server = _FakeServerSocket(conn, accept_error=accept_error)
        original_socket_ctor = mobo.socket.socket
        original_execute = mobo.mobo_execute
        try:
            mobo.socket.socket = lambda *args, **kwargs: fake_server
            if execute_stub is None:
                mobo.mobo_execute = lambda *args, **kwargs: None
            else:
                mobo.mobo_execute = execute_stub
            mobo.main()
        finally:
            mobo.socket.socket = original_socket_ctor
            mobo.mobo_execute = original_execute
        return conn, fake_server

    def test_parse_param_and_obj_init(self):
        mobo = load_mobo_module()
        self.assertEqual(mobo.parse_param_init({"low": 0, "high": 1}), (0.0, 1.0))
        self.assertEqual(mobo.parse_param_init("2.5, 3.5"), (2.5, 3.5))
        self.assertEqual(mobo.parse_obj_init({"low": 1, "high": 9, "minimize": 1}), (1.0, 9.0, 1))
        self.assertEqual(mobo.parse_obj_init("0,10,0"), (0.0, 10.0, 0))

    def test_parse_init_invalid_raises(self):
        mobo = load_mobo_module()
        with self.assertRaises(ValueError):
            mobo.parse_param_init("0")
        with self.assertRaises(ValueError):
            mobo.parse_obj_init("0,1")

    def test_parse_init_missing_keys_raises(self):
        mobo = load_mobo_module()
        with self.assertRaises(ValueError):
            mobo.parse_param_init({"low": 0.0})
        with self.assertRaises(ValueError):
            mobo.parse_obj_init({"low": 0.0, "high": 1.0})

    def test_send_json_line_disconnect_raises_connection_error(self):
        mobo = load_mobo_module()
        conn = _FakeConn([], send_error=BrokenPipeError("pipe closed"))
        with self.assertRaises(ConnectionError):
            mobo.send_json_line(conn, {"type": "coverage", "value": 1.0})

    def test_objective_function_mixed_minimize_maximize(self):
        mobo = load_mobo_module()
        conn = _FakeConn(
            [
                _json_line({"type": "log", "message": "starting"}),
                _json_line({"type": "tempCoverage", "value": 0.5}),
                _json_line(
                    {
                        "type": "objectives",
                        "values": {"obj_min_1": 20.0, "obj_max_1": 8.0, "obj_min_2": 0.9},
                    }
                ),
            ]
        )
        mobo.parameter_names = ["p0"]
        mobo.parameters_info = [(0.0, 1.0)]
        mobo.objective_names = ["obj_min_1", "obj_max_1", "obj_min_2"]
        mobo.objectives_info = [
            (0.0, 100.0, 1),
            (0.0, 10.0, 0),
            (0.0, 1.0, 1),
        ]

        x = FakeTensor([0.123456789])
        y = mobo.objective_function(conn=conn, x_tensor=x, iteration=1)

        np.testing.assert_allclose(y.numpy(), np.array([0.6, 0.6, -0.8]), atol=1e-12)
        self.assertGreaterEqual(len(conn.sent), 1)
        sent_obj = json.loads(conn.sent[0].decode("utf-8"))
        self.assertEqual(sent_obj["type"], "parameters")
        self.assertEqual(sent_obj["iteration"], 1)
        # Precision-preserving payload (not rounded to 3 decimals)
        self.assertAlmostEqual(sent_obj["values"]["p0"], 0.123456789, places=9)

    def test_objective_function_matches_former_inline_transform(self):
        # objective_function now calls bo_normalize.normalize_objective_value per objective
        # (like DBO and MetaTAF) instead of its own copy: same values, same errors.
        mobo = load_mobo_module()
        mobo.parameter_names = ["p0"]
        mobo.parameters_info = [(0.0, 1.0)]
        mobo.objective_names = ["o0", "o1"]
        for raw, lo, hi in OBJECTIVE_TRANSFORM_CASES:
            for minflag in (0, 1):
                with self.subTest(raw=raw, lo=lo, hi=hi, minflag=minflag):
                    mobo.objectives_info = [(0.0, 10.0, 1), (lo, hi, minflag)]
                    reset_protocol_state()
                    conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": 2.5, "o1": raw}})])
                    try:
                        got = ("value", mobo.objective_function(conn, FakeTensor([0.5]), 1).tolist())
                    except ValueError as e:
                        got = ("error", str(e))
                    kind, want = legacy_outcome("o1", raw, lo, hi, minflag)
                    self.assertEqual(got, (kind, [0.5, want] if kind == "value" else want))

    def test_acquisition_uses_shared_reference_point_and_batch_limit_five(self):
        mobo = load_mobo_module()
        captured = {}

        def fake_acq(**kwargs):
            captured["ref_point"] = kwargs["ref_point"]
            return object()

        def fake_optimize_acqf(acq_function, bounds, q, num_restarts, raw_samples, options, sequential):
            captured.update(options=options, num_restarts=num_restarts)
            return FakeTensor([[0.5]]), None

        mobo.qLogNoisyExpectedHypervolumeImprovement = fake_acq
        mobo.optimize_acqf = fake_optimize_acqf
        mobo.NUM_RESTARTS, mobo.RAW_SAMPLES, mobo.BATCH_SIZE = 6, 64, 1
        mobo.problem_bounds = FakeTensor([[0.0], [1.0]])
        mobo.ref_point = mobo.reference_point(3)
        mobo.optimize_qnehvi(model=None, sampler=None, X_baseline=FakeTensor([[0.1]]))
        self.assertEqual(captured["ref_point"], [-1.1, -1.1, -1.1])
        self.assertEqual(captured["options"], {"batch_limit": 5, "init_batch_limit": 128, "maxiter": 200})

    def test_main_sets_reference_point_and_mc_samples_fallback(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        del init_msg["config"]["mcSamples"]
        self._run_main_with_init(mobo, init_msg)
        self.assertEqual(mobo.MC_SAMPLES, 128)
        self.assertEqual(mobo.ref_point.tolist(), [-1.1, -1.1])

    def test_objective_function_missing_objective_key_raises(self):
        mobo = load_mobo_module()
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"obj0": 0.2}})])
        mobo.parameter_names = ["p0"]
        mobo.parameters_info = [(0.0, 1.0)]
        mobo.objective_names = ["obj0", "obj1"]
        mobo.objectives_info = [(0.0, 1.0, 0), (0.0, 1.0, 0)]

        with self.assertRaises(KeyError):
            mobo.objective_function(conn=conn, x_tensor=FakeTensor([0.5]), iteration=1)

    def test_objective_function_non_finite_objective_raises(self):
        mobo = load_mobo_module()
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"obj0": float("nan")}})])
        mobo.parameter_names = ["p0"]
        mobo.parameters_info = [(0.0, 1.0)]
        mobo.objective_names = ["obj0"]
        mobo.objectives_info = [(0.0, 1.0, 0)]
        with self.assertRaises(ValueError):
            mobo.objective_function(conn=conn, x_tensor=FakeTensor([0.2]), iteration=1)

    def test_objective_function_non_numeric_objective_raises(self):
        mobo = load_mobo_module()
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"obj0": "not-a-number"}})])
        mobo.parameter_names = ["p0"]
        mobo.parameters_info = [(0.0, 1.0)]
        mobo.objective_names = ["obj0"]
        mobo.objectives_info = [(0.0, 1.0, 0)]
        with self.assertRaises(ValueError):
            mobo.objective_function(conn=conn, x_tensor=FakeTensor([0.2]), iteration=1)

    def test_objective_function_out_of_bounds_raises(self):
        mobo = load_mobo_module()
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"obj0": 2.0}})])
        mobo.parameter_names = ["p0"]
        mobo.parameters_info = [(0.0, 1.0)]
        mobo.objective_names = ["obj0"]
        mobo.objectives_info = [(0.0, 1.0, 0)]
        with self.assertRaises(ValueError):
            mobo.objective_function(conn=conn, x_tensor=FakeTensor([0.2]), iteration=1)

    def test_objective_function_invalid_values_payload_type_raises(self):
        mobo = load_mobo_module()
        conn = _FakeConn([_json_line({"type": "objectives", "values": ["bad"]})])
        mobo.parameter_names = ["p0"]
        mobo.parameters_info = [(0.0, 1.0)]
        mobo.objective_names = ["obj0"]
        mobo.objectives_info = [(0.0, 1.0, 0)]
        with self.assertRaisesRegex(RuntimeError, "non-dict 'values'"):
            mobo.objective_function(conn=conn, x_tensor=FakeTensor([0.2]), iteration=1)

    def test_objective_function_timeout_mid_eval_raises(self):
        mobo = load_mobo_module()
        mobo.SOCKET_TIMEOUT_SEC = 5
        conn = _FakeConn([mobo.socket.timeout("timeout")])
        mobo.parameter_names = ["p0"]
        mobo.parameters_info = [(0.0, 1.0)]
        mobo.objective_names = ["obj0"]
        mobo.objectives_info = [(0.0, 1.0, 0)]
        with self.assertRaises(TimeoutError):
            mobo.objective_function(conn=conn, x_tensor=FakeTensor([0.2]), iteration=1)

    def test_recv_objectives_blocking_skips_non_objective_messages(self):
        mobo = load_mobo_module()
        conn = _FakeConn(
            [
                _json_line({"type": "coverage", "value": 1.0}),
                _json_line({"type": "log", "message": "mid"}),
                _json_line({"type": "objectives", "values": {"a": 1.25}}),
            ]
        )
        msg = mobo.recv_objectives_blocking(conn)
        self.assertEqual(msg, {"a": 1.25})

    def test_recv_objectives_blocking_skips_non_dict_messages(self):
        mobo = load_mobo_module()
        conn = _FakeConn(
            [
                b'["not-a-dict"]\n',
                _json_line({"type": "objectives", "values": {"a": 2.5}}),
            ]
        )
        msg = mobo.recv_objectives_blocking(conn)
        self.assertEqual(msg, {"a": 2.5})

    def test_load_data_normalized_native_flips_min_objectives(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({"p0": [5.0]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o_min": [0.5], "o_max": [-0.25]}).to_csv(
                init_dir / "objs.csv", sep=";", index=False
            )

            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mobo.CSV_PATH_PARAMETERS = "params.csv"
                mobo.CSV_PATH_OBJECTIVES = "objs.csv"
                mobo.parameter_names = ["p0"]
                mobo.objective_names = ["o_min", "o_max"]
                mobo.parameters_info = [(0.0, 10.0)]
                mobo.objectives_info = [(0.0, 100.0, 1), (0.0, 100.0, 0)]
                mobo.PROBLEM_DIM = 1
                mobo.NUM_OBJS = 2
                mobo.WARM_START_OBJECTIVE_FORMAT = "normalized_native"

                x, y = mobo.load_data()
            finally:
                os.chdir(prev_cwd)

        np.testing.assert_allclose(x.numpy(), np.array([[0.5]]), atol=1e-12)
        np.testing.assert_allclose(y.numpy(), np.array([[-0.5, -0.25]]), atol=1e-12)

    def test_normalize_obj_column_modes(self):
        mobo = load_mobo_module()
        col = np.array([0.5, -0.5])
        lo, hi, minflag = (0.0, 10.0, 1)

        mobo.WARM_START_OBJECTIVE_FORMAT = "normalized_max"
        y_max = mobo.normalize_obj_column(col, lo, hi, minflag)
        np.testing.assert_allclose(y_max, np.array([0.5, -0.5]), atol=1e-12)

        mobo.WARM_START_OBJECTIVE_FORMAT = "normalized_native"
        y_native = mobo.normalize_obj_column(col, lo, hi, minflag)
        np.testing.assert_allclose(y_native, np.array([-0.5, 0.5]), atol=1e-12)

        mobo.WARM_START_OBJECTIVE_FORMAT = "raw"
        y_raw = mobo.normalize_obj_column(np.array([2.0]), lo, hi, minflag)
        np.testing.assert_allclose(y_raw, np.array([0.6]), atol=1e-12)

    def test_normalize_obj_column_forced_normalized_out_of_range_raises(self):
        mobo = load_mobo_module()
        mobo.WARM_START_OBJECTIVE_FORMAT = "normalized_max"
        with self.assertRaises(ValueError):
            mobo.normalize_obj_column(np.array([2.5]), 0.0, 10.0, 0)

    def test_normalize_param_column_auto_out_of_range_raises(self):
        mobo = load_mobo_module()
        with self.assertRaises(ValueError):
            mobo.normalize_param_column(np.array([50.0]), 0.0, 10.0)

    def test_normalize_obj_column_auto_out_of_range_raises(self):
        mobo = load_mobo_module()
        mobo.WARM_START_OBJECTIVE_FORMAT = "auto"
        with self.assertRaises(ValueError):
            mobo.normalize_obj_column(np.array([50.0]), 0.0, 10.0, 0)

    def test_load_data_auto_ambiguous_prefers_raw(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({"p0": [0.2]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o_min": [0.2]}).to_csv(init_dir / "objs.csv", sep=";", index=False)

            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mobo.CSV_PATH_PARAMETERS = "params.csv"
                mobo.CSV_PATH_OBJECTIVES = "objs.csv"
                mobo.parameter_names = ["p0"]
                mobo.objective_names = ["o_min"]
                mobo.parameters_info = [(0.0, 1.0)]
                mobo.objectives_info = [(0.0, 1.0, 1)]
                mobo.PROBLEM_DIM = 1
                mobo.NUM_OBJS = 1
                mobo.WARM_START_OBJECTIVE_FORMAT = "auto"

                x, y = mobo.load_data()
            finally:
                os.chdir(prev_cwd)

        np.testing.assert_allclose(x.numpy(), np.array([[0.2]]), atol=1e-12)
        # Ambiguous [0,1] values are treated as raw and then min-objective-flipped.
        np.testing.assert_allclose(y.numpy(), np.array([[0.6]]), atol=1e-12)

    def test_load_data_missing_paths_raises(self):
        mobo = load_mobo_module()
        mobo.CSV_PATH_PARAMETERS = ""
        mobo.CSV_PATH_OBJECTIVES = ""
        with self.assertRaises(ValueError):
            mobo.load_data()

    def test_load_data_missing_columns_raises(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"wrong": [1.0]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o0": [1.0]}).to_csv(init_dir / "objs.csv", sep=";", index=False)
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mobo.CSV_PATH_PARAMETERS = "params.csv"
                mobo.CSV_PATH_OBJECTIVES = "objs.csv"
                mobo.parameter_names = ["p0"]
                mobo.objective_names = ["o0"]
                mobo.parameters_info = [(0.0, 1.0)]
                mobo.objectives_info = [(0.0, 1.0, 0)]
                mobo.PROBLEM_DIM = 1
                mobo.NUM_OBJS = 1
                with self.assertRaises(ValueError):
                    mobo.load_data()
            finally:
                os.chdir(prev_cwd)

    def test_load_data_row_mismatch_raises(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"p0": [0.1, 0.2]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o0": [0.1]}).to_csv(init_dir / "objs.csv", sep=";", index=False)
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mobo.CSV_PATH_PARAMETERS = "params.csv"
                mobo.CSV_PATH_OBJECTIVES = "objs.csv"
                mobo.parameter_names = ["p0"]
                mobo.objective_names = ["o0"]
                mobo.parameters_info = [(0.0, 1.0)]
                mobo.objectives_info = [(0.0, 1.0, 0)]
                mobo.PROBLEM_DIM = 1
                mobo.NUM_OBJS = 1
                with self.assertRaises(ValueError):
                    mobo.load_data()
            finally:
                os.chdir(prev_cwd)

    def test_load_data_non_numeric_raises(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"p0": ["abc"]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o0": [0.1]}).to_csv(init_dir / "objs.csv", sep=";", index=False)
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mobo.CSV_PATH_PARAMETERS = "params.csv"
                mobo.CSV_PATH_OBJECTIVES = "objs.csv"
                mobo.parameter_names = ["p0"]
                mobo.objective_names = ["o0"]
                mobo.parameters_info = [(0.0, 1.0)]
                mobo.objectives_info = [(0.0, 1.0, 0)]
                mobo.PROBLEM_DIM = 1
                mobo.NUM_OBJS = 1
                with self.assertRaises(Exception):
                    mobo.load_data()
            finally:
                os.chdir(prev_cwd)

    def test_load_data_nan_inf_raises(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"p0": [np.nan]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o0": [np.inf]}).to_csv(init_dir / "objs.csv", sep=";", index=False)
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mobo.CSV_PATH_PARAMETERS = "params.csv"
                mobo.CSV_PATH_OBJECTIVES = "objs.csv"
                mobo.parameter_names = ["p0"]
                mobo.objective_names = ["o0"]
                mobo.parameters_info = [(0.0, 1.0)]
                mobo.objectives_info = [(0.0, 1.0, 0)]
                mobo.PROBLEM_DIM = 1
                mobo.NUM_OBJS = 1
                with self.assertRaises(ValueError):
                    mobo.load_data()
            finally:
                os.chdir(prev_cwd)

    def test_load_data_out_of_bounds_raises(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            init_dir = pathlib.Path(tmp) / "InitData"
            init_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"p0": [50.0]}).to_csv(init_dir / "params.csv", sep=";", index=False)
            pd.DataFrame({"o0": [5.0], "o1": [6.0]}).to_csv(init_dir / "objs.csv", sep=";", index=False)
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mobo.CSV_PATH_PARAMETERS = "params.csv"
                mobo.CSV_PATH_OBJECTIVES = "objs.csv"
                mobo.parameter_names = ["p0"]
                mobo.objective_names = ["o0", "o1"]
                mobo.parameters_info = [(0.0, 10.0)]
                mobo.objectives_info = [(0.0, 10.0, 0), (0.0, 10.0, 0)]
                mobo.PROBLEM_DIM = 1
                mobo.NUM_OBJS = 2
                mobo.WARM_START_OBJECTIVE_FORMAT = "auto"
                with self.assertRaises(ValueError):
                    mobo.load_data()
            finally:
                os.chdir(prev_cwd)

    def test_generate_initial_data_rejects_non_positive_n(self):
        mobo = load_mobo_module()
        with self.assertRaises(ValueError):
            mobo.generate_initial_data(conn=None, n_samples=0)

    def test_csv_helpers_survive_a_locked_log_and_catch_up(self):
        mobo = load_mobo_module()
        protocol = reset_protocol_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ExecutionTimes.csv")
            with mock.patch.object(protocol, "LOG_WRITE_RETRY_SEC", 0.0), \
                    mock.patch("builtins.open", side_effect=PermissionError(13, "locked by Excel")), \
                    mock.patch("builtins.print"):
                mobo.create_csv_file(path, ["A"])  # no exception: the study goes on
                mobo.write_data_to_csv(path, ["A"], [{"A": 1}])
            self.assertFalse(os.path.exists(path))
            with mock.patch("builtins.print"):
                mobo.write_data_to_csv(path, ["A"], [{"A": 2}])  # lock gone: everything is written
            with open(path, newline="", encoding="utf-8") as f:
                self.assertEqual(list(csv.reader(f, delimiter=";")), [["A"], ["1"], ["2"]])
        self.assertEqual(protocol.unsaved_logs(), [])

    def test_moocore_metric_wrappers_use_maximized_objective_space(self):
        mobo = load_mobo_module()
        mobo.moocore.calls.clear()

        y_sample = FakeTensor([[0.2, 0.1], [0.1, 0.2]])
        ref = FakeTensor([-1.0, -1.0])

        mask = mobo.is_non_dominated(y_sample)
        volume = mobo.Hypervolume(ref).compute(y_sample)

        np.testing.assert_array_equal(mask, np.array([True, True]))
        self.assertAlmostEqual(volume, 0.6)
        self.assertEqual(mobo.moocore.calls[0][0], "is_nondominated")
        self.assertTrue(mobo.moocore.calls[0][2])
        self.assertTrue(mobo.moocore.calls[0][3])
        self.assertEqual(mobo.moocore.calls[1][0], "hypervolume")
        np.testing.assert_allclose(mobo.moocore.calls[1][2], np.array([-1.0, -1.0]))
        self.assertTrue(mobo.moocore.calls[1][3])

    def test_moocore_metric_wrapper_keeps_first_duplicate_front_point(self):
        mobo = load_mobo_module()

        y_sample = FakeTensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mask = mobo.is_non_dominated(y_sample)

        np.testing.assert_array_equal(mask, np.array([True, False, True]))

    def test_save_xy_updates_tail_ispareto_on_mismatch(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            mobo.PROJECT_PATH = tmp
            mobo.USER_ID = "u"
            mobo.CONDITION_ID = "c"
            mobo.GROUP_ID = "g"
            mobo.N_INITIAL = 0
            mobo.PROBLEM_DIM = 1
            mobo.NUM_OBJS = 2
            mobo.parameter_names = ["p0"]
            mobo.objective_names = ["o0", "o1"]
            mobo.parameters_info = [(0.0, 10.0)]
            mobo.objectives_info = [(0.0, 10.0, 0), (0.0, 10.0, 0)]

            csv_path = pathlib.Path(tmp) / "ObservationsPerEvaluation.csv"
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(
                    ["UserID", "ConditionID", "GroupID", "Timestamp", "Iteration", "Phase", "IsPareto", "o0", "o1", "p0"]
                )
                w.writerow(["old", "x", "x", "t", 1, "sampling", "OLD_A", 1, 1, 1])
                w.writerow(["old", "x", "x", "t", 2, "sampling", "OLD_B", 1, 1, 1])
                w.writerow(["run", "x", "x", "t", 3, "sampling", "FALSE", 1, 1, 1])
                w.writerow(["run", "x", "x", "t", 4, "sampling", "FALSE", 1, 1, 1])

            mobo.is_non_dominated = lambda y: np.array([True, False, True], dtype=bool)
            x_sample = FakeTensor([[0.1], [0.2], [0.3]])
            y_sample = FakeTensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

            mobo.save_xy(x_sample, y_sample, iteration=1)
            df = pd.read_csv(csv_path, delimiter=";")

        self.assertEqual(df.loc[0, "IsPareto"], "OLD_A")
        self.assertEqual(df.loc[1, "IsPareto"], "OLD_B")
        self.assertEqual(df.loc[2, "IsPareto"], "TRUE")
        self.assertEqual(df.loc[3, "IsPareto"], "FALSE")
        self.assertEqual(df.loc[4, "IsPareto"], "TRUE")

    def test_save_xy_uses_row_count_for_iteration(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            mobo.PROJECT_PATH = tmp
            mobo.USER_ID = "u"
            mobo.CONDITION_ID = "c"
            mobo.GROUP_ID = "g"
            mobo.N_INITIAL = 99
            mobo.PROBLEM_DIM = 1
            mobo.NUM_OBJS = 1
            mobo.parameter_names = ["p0"]
            mobo.objective_names = ["o0"]
            mobo.parameters_info = [(0.0, 10.0)]
            mobo.objectives_info = [(0.0, 10.0, 0)]

            mobo.is_non_dominated = lambda y: np.array([True, False], dtype=bool)
            x_sample = FakeTensor([[0.1], [0.2]])
            y_sample = FakeTensor([[0.1], [0.2]])
            mobo.save_xy(x_sample, y_sample, iteration=1)
            df = pd.read_csv(pathlib.Path(tmp) / "ObservationsPerEvaluation.csv", delimiter=";")

        self.assertEqual(int(df.iloc[-1]["Iteration"]), 2)

    def test_save_xy_rejects_corrupt_observation_schema(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            mobo.PROJECT_PATH = tmp
            mobo.USER_ID = "u"
            mobo.CONDITION_ID = "c"
            mobo.GROUP_ID = "g"
            mobo.N_INITIAL = 0
            mobo.PROBLEM_DIM = 1
            mobo.NUM_OBJS = 1
            mobo.parameter_names = ["p0"]
            mobo.objective_names = ["o0"]
            mobo.parameters_info = [(0.0, 10.0)]
            mobo.objectives_info = [(0.0, 10.0, 0)]

            csv_path = pathlib.Path(tmp) / "ObservationsPerEvaluation.csv"
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["Wrong", "Schema"])
                w.writerow(["x", "y"])

            x_sample = FakeTensor([[0.1]])
            y_sample = FakeTensor([[0.2]])
            with self.assertRaises(ValueError):
                mobo.save_xy(x_sample, y_sample, iteration=1)

    def test_save_hypervolume_to_file_writes_header_once(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            mobo.PROJECT_PATH = tmp
            mobo.ref_point = mobo.reference_point(1)
            mobo.save_hypervolume_to_file([0.1], iteration=0)
            mobo.save_hypervolume_to_file([0.2], iteration=1)
            hv_csv = pathlib.Path(tmp) / "HypervolumePerEvaluation.csv"
            with hv_csv.open() as f:
                rows = list(csv.reader(f, delimiter=";"))

        self.assertEqual(rows[0], ["Hypervolume", "Iteration", "Scale", "ReferencePoint"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1], ["0.1", "0", "normalized maximize-space [-1,1] per objective", "[-1.1]"])
        self.assertEqual(rows[2], ["0.2", "1", "normalized maximize-space [-1,1] per objective", "[-1.1]"])

    def test_mobo_execute_with_simulated_unity_objective_stream(self):
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mobo.USER_ID = "u"
                mobo.CONDITION_ID = "c"
                mobo.GROUP_ID = "g"
                mobo.WARM_START = False
                mobo.SEED = 3
                mobo.PROBLEM_DIM = 1
                mobo.NUM_OBJS = 2
                mobo.BATCH_SIZE = 1
                mobo.NUM_RESTARTS = 2
                mobo.RAW_SAMPLES = 16
                mobo.MC_SAMPLES = 8
                mobo.parameter_names = ["p0"]
                mobo.objective_names = ["o_min", "o_max"]
                mobo.parameters_info = [(0.0, 10.0)]
                mobo.objectives_info = [(0.0, 10.0, 1), (0.0, 10.0, 0)]
                mobo.problem_bounds = FakeTensor([[0.0], [1.0]])
                mobo.ref_point = mobo.reference_point(2)
                mobo.SobolQMCNormalSampler = lambda sample_shape, seed: {"shape": sample_shape, "seed": seed}

                # Initial samples (n=2, q=1, d=1): x=0.2 then x=0.8
                mobo.draw_sobol_samples = lambda bounds, n, q, seed: FakeTensor([[[0.2]], [[0.8]]])
                # Optimization candidate: x=0.4
                mobo.optimize_qnehvi = lambda model, sampler, X_baseline: FakeTensor([[0.4]])

                # Unity-like objective stream for 3 evaluations:
                # 1) o_min=2,o_max=8 -> [0.6,0.6]
                # 2) o_min=8,o_max=2 -> [-0.6,-0.6]
                # 3) o_min=5,o_max=5 -> [0.0,0.0]
                conn = _FakeConn(
                    [
                        _json_line({"type": "log", "message": "eval1"}),
                        _json_line({"type": "objectives", "values": {"o_min": 2.0, "o_max": 8.0}}),
                        _json_line({"type": "coverage", "value": 0.2}),
                        _json_line({"type": "objectives", "values": {"o_min": 8.0, "o_max": 2.0}}),
                        _json_line({"type": "objectives", "values": {"o_min": 5.0, "o_max": 5.0}}),
                    ]
                )

                out_msgs = []
                original_send = mobo.send_json_line
                mobo.send_json_line = lambda c, payload: out_msgs.append(payload)
                try:
                    hvs, train_x, train_y = mobo.mobo_execute(conn=conn, seed=3, iterations=1, initial_samples=2)
                finally:
                    mobo.send_json_line = original_send
            finally:
                os.chdir(prev_cwd)

        # Three evaluations should have happened: 2 initial + 1 optimization.
        self.assertEqual(train_x.shape, (3, 1))
        self.assertEqual(train_y.shape, (3, 2))
        np.testing.assert_allclose(
            train_y.numpy(),
            np.array([[0.6, 0.6], [-0.6, -0.6], [0.0, 0.0]]),
            atol=1e-12,
        )
        # Hypervolume logs cover both sampling evaluations and the optimization step.
        self.assertEqual(len(hvs), 3)
        # Ensure loop sent completion signal.
        self.assertTrue(any(m.get("type") == "optimization_finished" for m in out_msgs))

    def test_stop_during_sampling_finalizes_ispareto_flags(self):
        # A perfect-rating stop in the sampling phase used to leave every sampling row at its
        # provisional IsPareto = FALSE, so FinalDesignSelector found no Pareto candidates.
        mobo = load_mobo_module()
        with tempfile.TemporaryDirectory() as tmp:
            mobo.PROJECT_PATH = tmp
            mobo.USER_ID, mobo.CONDITION_ID, mobo.GROUP_ID = "u", "c", "g"
            mobo.SEED = 3
            mobo.PROBLEM_DIM = 1
            mobo.NUM_OBJS = 2
            mobo.parameter_names = ["p0"]
            mobo.objective_names = ["o0", "o1"]
            mobo.parameters_info = [(0.0, 1.0)]
            mobo.objectives_info = [(0.0, 10.0, 0), (0.0, 10.0, 1)]
            mobo.problem_bounds = FakeTensor([[0.0], [1.0]])
            mobo.draw_sobol_samples = lambda bounds, n, q, seed: FakeTensor([[[0.1]], [[0.2]], [[0.3]], [[0.4]]])
            # The second design is dominated by the third (worse on both objectives).
            mobo.is_non_dominated = lambda y: np.array([True, False, True][: len(np.asarray(y.tolist()))])
            conn = _FakeConn(
                [_json_line({"type": "objectives", "values": {"o0": a, "o1": b}})
                 for a, b in ((2.0, 2.0), (5.0, 8.0), (8.0, 5.0))]
                + [_json_line({"type": "stop", "reason": "perfect_rating"})]
            )
            with mock.patch.object(mobo, "send_json_line", lambda c, payload: None):
                with self.assertRaises(mobo.StopRequested):
                    mobo.generate_initial_data(conn, n_samples=4)
            with open(pathlib.Path(tmp) / "ObservationsPerEvaluation.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter=";"))
        self.assertEqual([r["IsPareto"] for r in rows], ["TRUE", "FALSE", "TRUE"])

    def test_initial_design_ids_and_one_iteration_axis(self):
        mobo = load_mobo_module()
        sobol_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mobo.USER_ID, mobo.CONDITION_ID, mobo.GROUP_ID = "007", "01", "NA"
                mobo.USER_LOG_ID = mobo.CONDITION_LOG_ID = "ids"
                mobo.WARM_START = False
                mobo.SEED = 3
                mobo.PROBLEM_DIM = 1
                mobo.NUM_OBJS = 2
                mobo.BATCH_SIZE = 1
                mobo.NUM_RESTARTS = 2
                mobo.RAW_SAMPLES = 16
                mobo.MC_SAMPLES = 8
                mobo.parameter_names = ["p0"]
                mobo.objective_names = ["o_min", "o_max"]
                mobo.parameters_info = [(0.0, 10.0)]
                mobo.objectives_info = [(0.0, 10.0, 1), (0.0, 10.0, 0)]
                mobo.problem_bounds = FakeTensor([[0.0], [1.0]])
                mobo.ref_point = mobo.reference_point(2)
                mobo.SobolQMCNormalSampler = lambda sample_shape, seed: {"shape": sample_shape, "seed": seed}

                def sobol(bounds, n, q, seed):
                    sobol_calls.append((n, q))
                    return FakeTensor([[[0.2]], [[0.8]]])

                mobo.draw_sobol_samples = sobol
                mobo.optimize_qnehvi = lambda model, sampler, X_baseline: FakeTensor([[0.4]])
                conn = _FakeConn([
                    _json_line({"type": "objectives", "values": {"o_min": 2.0, "o_max": 8.0}}),
                    _json_line({"type": "objectives", "values": {"o_min": 8.0, "o_max": 2.0}}),
                    _json_line({"type": "objectives", "values": {"o_min": 5.0, "o_max": 5.0}}),
                    _json_line({"type": "objectives", "values": {"o_min": 4.0, "o_max": 6.0}}),
                ])
                original_send = mobo.send_json_line
                mobo.send_json_line = lambda c, payload: None
                try:
                    mobo.mobo_execute(conn=conn, seed=3, iterations=2, initial_samples=2)
                finally:
                    mobo.send_json_line = original_send
                run_dir = pathlib.Path(tmp) / "LogData" / "ids" / "ids" / "run"
                with open(run_dir / "ObservationsPerEvaluation.csv", newline="", encoding="utf-8") as f:
                    obs_rows = list(csv.reader(f, delimiter=";"))[1:]
                with open(run_dir / "HypervolumePerEvaluation.csv", newline="", encoding="utf-8") as f:
                    hv_rows = list(csv.reader(f, delimiter=";"))[1:]
            finally:
                os.chdir(prev_cwd)

        self.assertEqual(sobol_calls, [(2, 1)])
        self.assertTrue(all(r[:3] == ["007", "01", "NA"] for r in obs_rows), obs_rows)
        # Every metric row carries the Iteration of its evaluation's observation row.
        self.assertEqual([r[4] for r in obs_rows], ["1", "2", "3", "4"])
        self.assertEqual([r[1] for r in hv_rows], ["1", "2", "3", "4"])

    def test_objectives_decode_utf8_split_across_chunks(self):
        mobo = load_mobo_module()
        payload = '{"type":"objectives","values":{"Übersicht":1.0}}\n'.encode("utf-8")
        cut = payload.index("Ü".encode("utf-8")) + 1  # inside the two-byte character
        conn = _FakeConn([payload[:cut], payload[cut:]])
        values = mobo.recv_objectives_blocking(conn)
        self.assertEqual(list(values), ["Übersicht"])

    def test_main_listens_on_loopback_only_and_stops_after_connect(self):
        mobo = load_mobo_module()
        server, listening = run_main_recording_listener(mobo, self._base_init_message(), "mobo_execute")
        assert_hardened_listener(self, mobo.socket, server, listening)

    def test_main_rejects_keys_colliding_with_log_columns(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["objectives"][1]["key"] = "IsPareto"
        with self.assertRaisesRegex(ValueError, "collide with log columns"):
            self._run_main_with_init(mobo, init_msg)

    def test_main_parses_unity_init_stream_and_calls_execute(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["config"]["batchSize"] = 2
        init_msg["config"]["warmStartObjectiveFormat"] = "normalized_native"

        init_bytes = _json_line(init_msg)
        conn = _FakeConn([init_bytes[:20], init_bytes[20:]])
        fake_server = _FakeServerSocket(conn)
        called = {}

        original_socket_ctor = mobo.socket.socket
        original_execute = mobo.mobo_execute
        try:
            mobo.socket.socket = lambda *args, **kwargs: fake_server

            def _exec_stub(conn_arg, seed, iterations, initial_samples):
                called["args"] = (conn_arg, seed, iterations, initial_samples)
                return [], FakeTensor([[0.0]]), FakeTensor([[0.0, 0.0]])

            mobo.mobo_execute = _exec_stub
            mobo.main()
        finally:
            mobo.socket.socket = original_socket_ctor
            mobo.mobo_execute = original_execute

        self.assertIn("args", called)
        self.assertIs(called["args"][0], conn)
        self.assertEqual(called["args"][1:], (11, 1, 2))
        self.assertEqual(mobo.BATCH_SIZE, 1)  # forced from 2 -> 1
        self.assertEqual(mobo.WARM_START_OBJECTIVE_FORMAT, "normalized_native")
        self.assertEqual(conn.timeout, mobo.SOCKET_TIMEOUT_SEC)
        self.assertTrue(conn.shutdown_called)
        self.assertTrue(conn.closed)
        self.assertTrue(fake_server.closed)

    def test_main_skips_non_dict_before_init_and_calls_execute(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        conn = _FakeConn([b'["noise"]\n' + _json_line(init_msg)])
        fake_server = _FakeServerSocket(conn)
        called = {}

        def _exec_stub(conn_arg, seed, iterations, initial_samples):
            called["args"] = (conn_arg, seed, iterations, initial_samples)
            return [], FakeTensor([[0.0]]), FakeTensor([[0.0, 0.0]])

        original_socket_ctor = mobo.socket.socket
        original_execute = mobo.mobo_execute
        try:
            mobo.socket.socket = lambda *args, **kwargs: fake_server
            mobo.mobo_execute = _exec_stub
            mobo.main()
        finally:
            mobo.socket.socket = original_socket_ctor
            mobo.mobo_execute = original_execute

        self.assertIn("args", called)
        self.assertEqual(called["args"][1:], (11, 1, 2))
        self.assertTrue(conn.shutdown_called)
        self.assertTrue(conn.closed)
        self.assertTrue(fake_server.closed)

    def test_main_rejects_invalid_warm_start_objective_format(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["config"]["warmStartObjectiveFormat"] = "invalid-mode"
        conn = _FakeConn([_json_line(init_msg)])
        fake_server = _FakeServerSocket(conn)

        original_socket_ctor = mobo.socket.socket
        original_execute = mobo.mobo_execute
        try:
            mobo.socket.socket = lambda *args, **kwargs: fake_server
            mobo.mobo_execute = lambda *args, **kwargs: None
            with self.assertRaises(ValueError):
                mobo.main()
        finally:
            mobo.socket.socket = original_socket_ctor
            mobo.mobo_execute = original_execute

    def test_main_rejects_duplicate_parameter_keys(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["parameters"] = [
            {"key": "p0", "init": {"low": 0.0, "high": 1.0}},
            {"key": "p0", "init": {"low": 0.0, "high": 1.0}},
        ]
        init_msg["config"]["nParameters"] = 2
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_duplicate_objective_keys(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["objectives"] = [
            {"key": "o0", "init": {"low": 0.0, "high": 10.0, "minimize": 1}},
            {"key": "o0", "init": {"low": 0.0, "high": 10.0, "minimize": 0}},
        ]
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_invalid_parameter_bounds(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["parameters"][0]["init"]["low"] = 2.0
        init_msg["parameters"][0]["init"]["high"] = 1.0
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_missing_param_bounds_keys(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["parameters"][0]["init"] = {"low": 0.0}
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_non_finite_parameter_bounds(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["parameters"][0]["init"]["low"] = "nan"
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_invalid_objective_bounds(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["objectives"][0]["init"]["low"] = 10.0
        init_msg["objectives"][0]["init"]["high"] = 0.0
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_non_finite_objective_bounds(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["objectives"][0]["init"]["high"] = "inf"
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_invalid_minimize_flag(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["objectives"][0]["init"]["minimize"] = 3
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_missing_objective_minimize_key(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        del init_msg["objectives"][0]["init"]["minimize"]
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_parameter_count_mismatch(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["config"]["nParameters"] = 2
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_objective_count_mismatch(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["config"]["nObjectives"] = 3
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_negative_iteration_counts(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["config"]["numSamplingIterations"] = -1
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_non_positive_optimizer_hyperparams(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["config"]["numRestarts"] = 0
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_num_objs_below_two(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["config"]["nObjectives"] = 1
        init_msg["objectives"] = [init_msg["objectives"][0]]
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_non_positive_socket_timeout(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        mobo.SOCKET_TIMEOUT_SEC = 0
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_rejects_missing_required_nparameters(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        del init_msg["config"]["nParameters"]
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_closes_socket_and_conn_on_init_validation_error(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        del init_msg["config"]["nParameters"]
        conn = _FakeConn([_json_line(init_msg)])
        fake_server = _FakeServerSocket(conn)
        original_socket_ctor = mobo.socket.socket
        original_execute = mobo.mobo_execute
        try:
            mobo.socket.socket = lambda *args, **kwargs: fake_server
            mobo.mobo_execute = lambda *args, **kwargs: None
            with self.assertRaises(ValueError):
                mobo.main()
        finally:
            mobo.socket.socket = original_socket_ctor
            mobo.mobo_execute = original_execute

        self.assertTrue(conn.shutdown_called)
        self.assertTrue(conn.closed)
        self.assertTrue(fake_server.closed)

    def test_main_rejects_missing_required_nobjectives(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        del init_msg["config"]["nObjectives"]
        with self.assertRaises(ValueError):
            self._run_main_with_init(mobo, init_msg)

    def test_main_accept_timeout_raises(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        with self.assertRaises(TimeoutError):
            self._run_main_with_init(mobo, init_msg, accept_error=mobo.socket.timeout("accept timeout"))

    def test_main_fails_if_no_init_message_received(self):
        mobo = load_mobo_module()
        conn = _FakeConn([])
        fake_server = _FakeServerSocket(conn)
        original_socket_ctor = mobo.socket.socket
        original_execute = mobo.mobo_execute
        try:
            mobo.socket.socket = lambda *args, **kwargs: fake_server
            mobo.mobo_execute = lambda *args, **kwargs: None
            with self.assertRaises(RuntimeError):
                mobo.main()
        finally:
            mobo.socket.socket = original_socket_ctor
            mobo.mobo_execute = original_execute

    def test_main_warm_start_missing_files_raises_from_execute(self):
        mobo = load_mobo_module()
        init_msg = self._base_init_message()
        init_msg["config"]["warmStart"] = True
        init_msg["config"]["initialParametersDataPath"] = "missing_params.csv"
        init_msg["config"]["initialObjectivesDataPath"] = "missing_objs.csv"

        with tempfile.TemporaryDirectory() as tmp:
            init_data = pathlib.Path(tmp) / "InitData"
            init_data.mkdir(parents=True, exist_ok=True)
            prev_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                conn = _FakeConn([_json_line(init_msg)])
                fake_server = _FakeServerSocket(conn)
                original_socket_ctor = mobo.socket.socket
                original_execute = mobo.mobo_execute
                try:
                    mobo.socket.socket = lambda *args, **kwargs: fake_server
                    # exercise real execute path so load_data checks file existence
                    with self.assertRaises(FileNotFoundError):
                        mobo.main()
                finally:
                    mobo.socket.socket = original_socket_ctor
                    mobo.mobo_execute = original_execute
            finally:
                os.chdir(prev_cwd)


if __name__ == "__main__":
    unittest.main()
