import importlib.util
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


REPO_ROOT = pathlib.Path("/Users/pjansen/Desktop/Bayesian-Optimization-for-Unity")
BO_PATH = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization/bo.py"


class FakeTensor:
    def __init__(self, data):
        self.arr = np.asarray(data, dtype=np.float64)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.arr, dtype=np.float64)

    def clone(self):
        return FakeTensor(self.arr.copy())

    def to(self, dtype=None):
        return self

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.arr, axis=dim))

    def squeeze(self, dim=None):
        if dim is None:
            return FakeTensor(np.squeeze(self.arr))
        return FakeTensor(np.squeeze(self.arr, axis=dim))

    def detach(self):
        return self

    def tolist(self):
        return self.arr.tolist()

    def item(self):
        return float(np.asarray(self.arr).reshape(-1)[0])

    def dim(self):
        return self.arr.ndim

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, idx):
        out = self.arr[idx]
        if isinstance(out, np.ndarray):
            return FakeTensor(out)
        return float(out)


def _to_array(x):
    if isinstance(x, FakeTensor):
        return x.arr
    return np.asarray(x, dtype=np.float64)


def install_stub_modules():
    torch_mod = types.ModuleType("torch")
    torch_mod.double = np.float64

    class _Device:
        def __init__(self, name):
            self.name = name

    torch_mod.device = _Device
    torch_mod.Size = tuple
    torch_mod.tensor = lambda data, dtype=None: FakeTensor(data)
    torch_mod.stack = lambda seq, dim=0: FakeTensor(np.stack([_to_array(x) for x in seq], axis=dim))
    torch_mod.cat = lambda seq, dim=0: FakeTensor(np.concatenate([_to_array(x) for x in seq], axis=dim))
    torch_mod.manual_seed = lambda seed: None
    torch_mod.zeros = lambda n, dtype=None: FakeTensor(np.zeros(n, dtype=np.float64))
    torch_mod.ones = lambda n, dtype=None: FakeTensor(np.ones(n, dtype=np.float64))
    sys.modules["torch"] = torch_mod

    botorch_mod = types.ModuleType("botorch")
    sys.modules["botorch"] = botorch_mod
    sys.modules["botorch.acquisition"] = types.ModuleType("botorch.acquisition")
    acq_logei_mod = types.ModuleType("botorch.acquisition.logei")
    acq_logei_mod.qLogNoisyExpectedImprovement = object
    sys.modules["botorch.acquisition.logei"] = acq_logei_mod

    models_mod = types.ModuleType("botorch.models")

    class _SingleTaskGP:
        def __init__(self, train_x, train_obj):
            self.train_inputs = (train_x,)
            self.likelihood = object()

    models_mod.SingleTaskGP = _SingleTaskGP
    sys.modules["botorch.models"] = models_mod

    fit_mod = types.ModuleType("botorch.fit")
    fit_mod.fit_gpytorch_mll = lambda mll: None
    sys.modules["botorch.fit"] = fit_mod

    sys.modules["botorch.optim"] = types.ModuleType("botorch.optim")
    optim_opt_mod = types.ModuleType("botorch.optim.optimize")
    optim_opt_mod.optimize_acqf = (
        lambda acq_function, bounds, q, num_restarts, raw_samples, options, sequential: (
            FakeTensor(np.zeros((q, _to_array(bounds).shape[-1]))),
            None,
        )
    )
    sys.modules["botorch.optim.optimize"] = optim_opt_mod

    sys.modules["botorch.sampling"] = types.ModuleType("botorch.sampling")
    sampling_normal_mod = types.ModuleType("botorch.sampling.normal")
    sampling_normal_mod.SobolQMCNormalSampler = object
    sys.modules["botorch.sampling.normal"] = sampling_normal_mod

    sys.modules["botorch.utils"] = types.ModuleType("botorch.utils")
    utils_sampling_mod = types.ModuleType("botorch.utils.sampling")
    utils_sampling_mod.draw_sobol_samples = (
        lambda bounds, n, q, seed: FakeTensor(np.zeros((n, q, _to_array(bounds).shape[-1])))
    )
    sys.modules["botorch.utils.sampling"] = utils_sampling_mod

    gpytorch_mod = types.ModuleType("gpytorch")
    sys.modules["gpytorch"] = gpytorch_mod
    gpytorch_mlls_mod = types.ModuleType("gpytorch.mlls")

    class _ExactMarginalLogLikelihood:
        def __init__(self, likelihood, model):
            self.likelihood = likelihood
            self.model = model

    gpytorch_mlls_mod.ExactMarginalLogLikelihood = _ExactMarginalLogLikelihood
    sys.modules["gpytorch.mlls"] = gpytorch_mlls_mod


def load_bo_module():
    install_stub_modules()
    name = f"bo_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeConn:
    def __init__(self, chunks, send_error=None):
        self._chunks = list(chunks)
        self.timeout = None
        self.sent = []
        self.send_error = send_error
        self.shutdown_called = False
        self.closed = False

    def recv(self, n):
        if not self._chunks:
            return b""
        item = self._chunks.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def sendall(self, data):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(data)

    def settimeout(self, timeout):
        self.timeout = timeout

    def shutdown(self, how):
        self.shutdown_called = True

    def close(self):
        self.closed = True


class _FakeServerSocket:
    def __init__(self, conn, accept_error=None):
        self.conn = conn
        self.accept_error = accept_error
        self.closed = False
        self.timeout = None
        self.sockopt_calls = []

    def setsockopt(self, level, optname, value):
        self.sockopt_calls.append((level, optname, value))

    def bind(self, addr):
        pass

    def listen(self, backlog):
        pass

    def settimeout(self, timeout):
        self.timeout = timeout

    def accept(self):
        if self.accept_error is not None:
            raise self.accept_error
        return self.conn, ("127.0.0.1", 12345)

    def close(self):
        self.closed = True


def _json_line(obj):
    return (json.dumps(obj) + "\n").encode("utf-8")


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

    def test_recv_json_message_multiple_lines_single_chunk(self):
        bo = load_bo_module()
        bo.SOCKET_RECV_BUF = ""
        conn = _FakeConn([b'{"type":"a"}\n{"type":"b"}\n'])

        msg1 = bo.recv_json_message(conn)
        msg2 = bo.recv_json_message(conn)

        self.assertEqual(msg1["type"], "a")
        self.assertEqual(msg2["type"], "b")

    def test_recv_json_message_discards_unterminated_tail(self):
        bo = load_bo_module()
        bo.SOCKET_RECV_BUF = ""
        conn = _FakeConn([b'{"type":"init"}\n{"type":"partial"', b""])

        msg1 = bo.recv_json_message(conn)
        self.assertEqual(msg1["type"], "init")

        msg2 = bo.recv_json_message(conn)
        self.assertIsNone(msg2)
        self.assertEqual(bo.SOCKET_RECV_BUF, "")

    def test_recv_json_message_timeout_raises(self):
        bo = load_bo_module()
        bo.SOCKET_RECV_BUF = ""
        bo.SOCKET_TIMEOUT_SEC = 7
        conn = _FakeConn([bo.socket.timeout("timeout")])

        with self.assertRaises(TimeoutError) as ctx:
            bo.recv_json_message(conn)
        self.assertIn("7", str(ctx.exception))

    def test_recv_json_message_buffer_overflow_raises_and_resets(self):
        bo = load_bo_module()
        bo.SOCKET_RECV_BUF = ""
        bo.SOCKET_MAX_RECV_BUF_BYTES = 8
        conn = _FakeConn([b"123456789"])

        with self.assertRaises(RuntimeError) as ctx:
            bo.recv_json_message(conn)
        self.assertIn("exceeded 8 bytes", str(ctx.exception))
        self.assertEqual(bo.SOCKET_RECV_BUF, "")

    def test_recv_objectives_blocking_preserves_buffer_across_calls(self):
        bo = load_bo_module()
        bo.SOCKET_RECV_BUF = ""
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
        bo.SOCKET_RECV_BUF = ""
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"wrong": 0.2}})])

        with self.assertRaises(KeyError):
            bo.objective_function(conn, FakeTensor([0.2]))

    def test_objective_function_non_numeric_objective_raises(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        bo.SOCKET_RECV_BUF = ""
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": "abc"}})])

        with self.assertRaises(ValueError):
            bo.objective_function(conn, FakeTensor([0.2]))

    def test_objective_function_non_finite_objective_raises(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        bo.SOCKET_RECV_BUF = ""
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": "inf"}})])

        with self.assertRaises(ValueError):
            bo.objective_function(conn, FakeTensor([0.2]))

    def test_objective_function_out_of_bounds_raises(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        bo.SOCKET_RECV_BUF = ""
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": 2.0}})])

        with self.assertRaises(ValueError):
            bo.objective_function(conn, FakeTensor([0.2]))

    def test_objective_function_preserves_parameter_precision(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        bo.SOCKET_RECV_BUF = ""
        conn = _FakeConn([_json_line({"type": "objectives", "values": {"o0": 0.5}})])

        bo.objective_function(conn, FakeTensor([0.123456789]))
        sent = json.loads(conn.sent[0].decode("utf-8"))
        self.assertAlmostEqual(sent["values"]["p0"], 0.123456789, places=9)

    def test_objective_function_invalid_values_payload_type_raises(self):
        bo = load_bo_module()
        bo.parameter_names = ["p0"]
        bo.parameters_info = [(0.0, 1.0)]
        bo.objective_names = ["o0"]
        bo.objectives_info = [(0.0, 1.0, 0)]
        bo.SOCKET_RECV_BUF = ""
        conn = _FakeConn([_json_line({"type": "objectives", "values": [1.0]})])

        with self.assertRaises(TypeError):
            bo.objective_function(conn, FakeTensor([0.2]))

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
        with tempfile.TemporaryDirectory() as tmp:
            bo.PROJECT_PATH = tmp
            bo.save_metric_to_file([0.1], iteration=0)
            bo.save_metric_to_file([0.2], iteration=1)
            p_best = pathlib.Path(tmp) / "BestObjectivePerEvaluation.csv"
            lines_best = p_best.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines_best[0], "BestObjective;Run")
            self.assertEqual(len(lines_best), 3)

            p_legacy = pathlib.Path(tmp) / "HypervolumePerEvaluation.csv"
            lines_legacy = p_legacy.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines_legacy[0], "Hypervolume;Run")
            self.assertEqual(len(lines_legacy), 3)

    def test_create_and_write_csv_helpers_propagate_errors(self):
        bo = load_bo_module()
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                bo.create_csv_file("/tmp/a.csv", ["A"])
            with self.assertRaises(OSError):
                bo.write_data_to_csv("/tmp/a.csv", ["A"], [{"A": 1}])

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
