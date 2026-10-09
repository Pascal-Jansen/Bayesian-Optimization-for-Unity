"""Shared stub infrastructure for backend tests that run without torch/botorch.

The CI environment installs only numpy+pandas, so the backend test suites
(test_bo.py, test_mobo.py, test_context_support.py) replace torch/botorch/
gpytorch/moocore in ``sys.modules`` with the lightweight stand-ins defined
here. Keeping a single implementation prevents the previously duplicated
copies from drifting apart.

NOTE: installing stubs mutates global ``sys.modules`` state. Test modules that
need the real stack afterwards must restore it (see
test_contextual_integration.restore_real_modules and the save/restore pattern
in test_context_support.TaskColumnTests).
"""

import json
import pathlib
import sys
import types

import numpy as np

BACKEND_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "Assets/StreamingAssets/BOData/BayesianOptimization"
)


def protocol_module():
    """The shared bo_protocol module that every backend imports its plumbing from."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    import bo_protocol

    return bo_protocol


def reset_protocol_state():
    """Empty NDJSON reader and no unsaved logs.

    Each test loads its own copy of a backend module, but all copies share bo_protocol's
    module state (the receive buffer, logs kept in memory while locked).
    """
    protocol = protocol_module()
    protocol.reset_receive_state()
    protocol.discard_unsaved_logs()
    return protocol


class FakeTensor:
    """Minimal torch.Tensor stand-in backed by a numpy array."""

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

    def __iter__(self):
        for item in self.arr:
            if isinstance(item, np.ndarray):
                yield FakeTensor(item)
            else:
                yield float(item)

    def __repr__(self):
        return f"FakeTensor({self.arr!r})"


def _to_array(x):
    if isinstance(x, FakeTensor):
        return x.arr
    return np.asarray(x, dtype=np.float64)


def install_torch_stub():
    """Install (and return) a stub 'torch' module into sys.modules."""
    torch_mod = types.ModuleType("torch")
    torch_mod.double = np.float64

    class _Device:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f"device({self.name})"

    torch_mod.device = _Device
    torch_mod.Size = tuple
    torch_mod.Tensor = FakeTensor
    torch_mod.tensor = lambda data, dtype=None: FakeTensor(data)
    torch_mod.stack = lambda seq, dim=0: FakeTensor(np.stack([_to_array(x) for x in seq], axis=dim))
    torch_mod.cat = lambda seq, dim=0: FakeTensor(np.concatenate([_to_array(x) for x in seq], axis=dim))
    torch_mod.manual_seed = lambda seed: None
    torch_mod.set_num_threads = lambda n: None
    torch_mod.get_num_threads = lambda: 1
    torch_mod.zeros = lambda n, dtype=None: FakeTensor(np.zeros(n, dtype=np.float64))
    torch_mod.ones = lambda n, dtype=None: FakeTensor(np.ones(n, dtype=np.float64))
    torch_mod.full = lambda shape, fill_value, dtype=None: FakeTensor(
        np.full(shape, fill_value, dtype=np.float64)
    )
    sys.modules["torch"] = torch_mod
    return torch_mod


def install_stub_modules():
    """Install stub torch/botorch/gpytorch/moocore modules into sys.modules."""
    install_torch_stub()

    botorch_mod = types.ModuleType("botorch")
    sys.modules["botorch"] = botorch_mod

    sys.modules["botorch.acquisition"] = types.ModuleType("botorch.acquisition")
    acq_logei_mod = types.ModuleType("botorch.acquisition.logei")
    acq_logei_mod.qLogNoisyExpectedImprovement = object
    sys.modules["botorch.acquisition.logei"] = acq_logei_mod

    sys.modules["botorch.acquisition.multi_objective"] = types.ModuleType(
        "botorch.acquisition.multi_objective"
    )
    acq_mo_logei_mod = types.ModuleType("botorch.acquisition.multi_objective.logei")
    acq_mo_logei_mod.qLogNoisyExpectedHypervolumeImprovement = object
    sys.modules["botorch.acquisition.multi_objective.logei"] = acq_mo_logei_mod

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

    moocore_mod = types.ModuleType("moocore")
    moocore_mod.calls = []

    def _is_nondominated(data, maximise=False, keep_weakly=False):
        arr = _to_array(data)
        moocore_mod.calls.append(("is_nondominated", arr.copy(), maximise, keep_weakly))
        return np.all(np.isfinite(arr), axis=1)

    def _hypervolume(data, ref, maximise=False):
        arr = _to_array(data)
        moocore_mod.calls.append(("hypervolume", arr.copy(), _to_array(ref).copy(), maximise))
        return float(np.sum(arr))

    moocore_mod.is_nondominated = _is_nondominated
    moocore_mod.hypervolume = _hypervolume
    sys.modules["moocore"] = moocore_mod


def install_openbo_stub():
    """Install a stub 'openbo' package (mobo_taf + mobo_botorch) into sys.modules.

    Lets test_meta_mobo_runtime exercise the full NDJSON protocol and CSV contract of the
    Meta-TAF backend in the numpy+pandas-only CI environment. The stub optimizer is
    deterministic: suggest() walks a fixed sequence of unit-cube points, observe() only
    records, and the source list is derived from the staged gp_states directory with the
    real loader's warn-and-skip contract (same "MO TAF source '<name>': ..." warnings) for
    unreadable trajectories and fronts that never dominate the reference point.
    """
    import dataclasses
    import os
    import warnings

    openbo_mod = types.ModuleType("openbo")
    optimizers_mod = types.ModuleType("openbo.optimizers")
    mobo_taf_mod = types.ModuleType("openbo.optimizers.mobo_taf")
    mobo_botorch_mod = types.ModuleType("openbo.optimizers.mobo_botorch")
    acquisition_mod = types.ModuleType("openbo.acquisition")
    taf_mo_ehvi_mod = types.ModuleType("openbo.acquisition.taf_mo_ehvi")

    class _StubSource:
        def __init__(self, name):
            self.name = name

    @dataclasses.dataclass
    class MOTAFConfig:
        """Field-for-field mirror of openbo's MOTAFConfig (MOBoTorchConfig + transfer
        settings, same defaults): a misspelled field fails like the real one, and the
        runtime's dataclasses.asdict() record is exercised."""

        bounds: list
        ref_point: list
        n_init: int = 5
        num_restarts: int = 5
        raw_samples: int = 64
        mc_samples: int = 128
        seed: object = 0
        taf_run_dir: object = ""
        n_iter: int = 25
        rho: float = 1.0
        taf_weight_mode: str = "taf_r"
        target_weight: float = 1.0
        source_meta_features: object = None
        target_meta_features: object = None
        source_reference_mode: str = "front"
        source_reference_quantile: float = 0.9
        source_only_warmup_iters: int = 0
        min_informative_pairs: int = 1
        decay_start_iter: int = 0
        decay_rate: float = 0.0

    class MOTAFSequentialOptimizer:
        instances = []

        def __init__(self, config):
            self.config = config
            # As in openbo's MOBoTorchSequentialOptimizer (rejects negative seeds).
            self.rng = np.random.default_rng(config.seed)
            self.d = len(config.bounds)
            self.n_suggestions = 0
            self.observed_x = []
            self.observed_y = []
            # openbo's MOBoTorchSequentialOptimizer: one hypervolume entry per observed row.
            self.hypervolume_history = []
            gp_dir = os.path.join(str(config.taf_run_dir), "gp_states")
            traj_dir = os.path.join(str(config.taf_run_dir), "trajectories")
            names = []
            if os.path.isdir(gp_dir):
                names = sorted(
                    f[:-5] for f in os.listdir(gp_dir) if f.endswith(".json")
                )
            ref = np.asarray(config.ref_point, dtype=np.float64)
            self.source_surrogates = []
            for name in names:
                try:
                    with open(os.path.join(traj_dir, f"{name}.json"), encoding="utf-8") as f:
                        trajectory = json.load(f)
                    front = np.asarray(
                        trajectory.get("pareto_front", trajectory["y_values"]), dtype=np.float64
                    )
                except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
                    warnings.warn(
                        f"MO TAF source '{name}': malformed artifacts ({exc!r}); "
                        "skipping this source."
                    )
                    continue
                if (front.ndim != 2 or front.shape[1] != ref.shape[0]
                        or not np.any(np.all(front > ref, axis=1))):
                    warnings.warn(
                        f"MO TAF source '{name}': cannot build its hypervolume term (source "
                        f"'{name}' has no front point dominating the reference point.); "
                        "skipping this source."
                    )
                    continue
                self.source_surrogates.append(_StubSource(name))
            n = len(self.source_surrogates)
            self.last_source_weights = (
                np.ones(n, dtype=np.float64) / n if n else np.zeros(0, dtype=np.float64)
            )
            self.last_target_weight = float(getattr(config, "target_weight", 1.0))
            MOTAFSequentialOptimizer.instances.append(self)

        def _decay_factor(self):
            return 1.0

        def suggest(self):
            self.n_suggestions += 1
            value = min(0.9, 0.25 + 0.1 * self.n_suggestions)
            return np.full((1, self.d), value, dtype=np.float64)

        def observe(self, x_new, y_new):
            x_new = np.asarray(x_new, dtype=np.float64)
            y_new = np.asarray(y_new, dtype=np.float64)
            if x_new.ndim != 2 or y_new.ndim != 2 or x_new.shape[0] != y_new.shape[0]:
                raise ValueError("stub observe(): bad shapes")
            self.observed_x.append(x_new)
            self.observed_y.append(y_new)
            all_y = np.vstack(self.observed_y)
            first_new = all_y.shape[0] - y_new.shape[0]
            for k in range(y_new.shape[0]):
                self.hypervolume_history.append(
                    compute_hypervolume(all_y[: first_new + k + 1], self.config.ref_point)
                )

    def compute_hypervolume(y, ref_point):
        arr = np.asarray(y, dtype=np.float64)
        ref = np.asarray(ref_point, dtype=np.float64)
        if arr.shape[0] == 0:
            return 0.0
        return float(np.sum(np.clip(arr - ref[None, :], 0.0, None)))

    # The runtime probes this symbol to detect an openbo predating the 2026-08 TAF-R
    # rework (which changed what mode token "taf_r" computes); the stub only needs it
    # to exist, tests exercise absence by deleting it.
    def compute_taf_r_ranking_weights(*args, **kwargs):
        raise NotImplementedError("stub: weight math lives in the openbo test suite")

    mobo_taf_mod.MOTAFConfig = MOTAFConfig
    mobo_taf_mod.MOTAFSequentialOptimizer = MOTAFSequentialOptimizer
    mobo_botorch_mod.compute_hypervolume = compute_hypervolume
    taf_mo_ehvi_mod.compute_taf_r_ranking_weights = compute_taf_r_ranking_weights

    openbo_mod.optimizers = optimizers_mod
    optimizers_mod.mobo_taf = mobo_taf_mod
    optimizers_mod.mobo_botorch = mobo_botorch_mod
    openbo_mod.acquisition = acquisition_mod
    acquisition_mod.taf_mo_ehvi = taf_mo_ehvi_mod
    sys.modules["openbo"] = openbo_mod
    sys.modules["openbo.optimizers"] = optimizers_mod
    sys.modules["openbo.optimizers.mobo_taf"] = mobo_taf_mod
    sys.modules["openbo.optimizers.mobo_botorch"] = mobo_botorch_mod
    sys.modules["openbo.acquisition"] = acquisition_mod
    sys.modules["openbo.acquisition.taf_mo_ehvi"] = taf_mo_ehvi_mod
    return mobo_taf_mod


class FakeConn:
    """Scripted socket connection: yields queued chunks, records sent bytes."""

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


class FakeServerSocket:
    """Stand-in for the listening server socket used by the backend main()."""

    def __init__(self, conn, accept_error=None):
        self.conn = conn
        self.accept_error = accept_error
        self.bound = None
        self.listen_backlog = None
        self.timeout = None
        self.closed = False
        self.sockopt_calls = []

    def setsockopt(self, level, optname, value):
        self.sockopt_calls.append((level, optname, value))

    def bind(self, addr):
        self.bound = addr

    def listen(self, backlog):
        self.listen_backlog = backlog

    def settimeout(self, timeout):
        self.timeout = timeout

    def accept(self):
        if self.accept_error is not None:
            raise self.accept_error
        return self.conn, ("127.0.0.1", 12345)

    def close(self):
        self.closed = True


def json_line(obj):
    return (json.dumps(obj) + "\n").encode("utf-8")


def run_main_recording_listener(module, init_msg, execute_attr, main_args=()):
    """Run a backend's main() against a fake server; return (server, listening_during_run).

    The backend's execute function is replaced by a stub that records whether the listening
    socket was still open once the session started.
    """
    conn = FakeConn([json_line(init_msg)])
    server = FakeServerSocket(conn)
    listening = []
    original_ctor = module.socket.socket
    original_execute = getattr(module, execute_attr)
    try:
        module.socket.socket = lambda *args, **kwargs: server
        setattr(module, execute_attr, lambda *args, **kwargs: listening.append(not server.closed))
        module.main(*main_args)
    finally:
        module.socket.socket = original_ctor
        setattr(module, execute_attr, original_execute)
    return server, listening


def assert_hardened_listener(testcase, socket_module, server, listening_during_run):
    """Loopback only, exclusive port where the OS supports it, no listening once connected."""
    testcase.assertEqual(server.bound, ("127.0.0.1", 56001))
    testcase.assertEqual(listening_during_run, [False])
    options = {optname for _, optname, _ in server.sockopt_calls}
    if hasattr(socket_module, "SO_EXCLUSIVEADDRUSE"):
        # Windows: SO_REUSEADDR would let a second backend share the listening port.
        testcase.assertEqual(options, {socket_module.SO_EXCLUSIVEADDRUSE})
    else:
        testcase.assertEqual(options, {socket_module.SO_REUSEADDR})
