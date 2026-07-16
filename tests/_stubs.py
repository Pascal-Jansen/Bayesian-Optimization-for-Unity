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
import sys
import types

import numpy as np


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
