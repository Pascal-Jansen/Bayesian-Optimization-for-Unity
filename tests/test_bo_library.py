"""Tests of the model-library switch (bo_library.py) and its use by bo.py, mobo.py and MetaTAF.

Stub-only, like test_bo.py: torch/botorch come from tests/_stubs.py, and botorch_rs is either
absent (sys.modules["botorch_rs"] = None) or a stand-in package from fake_botorch_rs(), so the
checks run in the numpy+pandas-only CI job whether or not botorch-rust is installed. The real
loops on botorch-rust are in test_contextual_integration.BotorchRsLoopTests.
"""

import contextlib
import hashlib
import io
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import (  # noqa: E402
    BACKEND_DIR,
    FakeConn,
    FakeServerSocket,
    FakeTensor,
    json_line,
)
from test_bo import load_bo_module  # noqa: E402
from test_mobo import load_mobo_module  # noqa: E402

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import bo_library  # noqa: E402

NOT_INSTALLED = {"botorch_rs": None}  # `import botorch_rs` raises ImportError


def _named(name, module):
    """A callable whose __module__/__name__ say it comes from ``module``."""
    def function(*args, **kwargs):
        return (name, args, kwargs)

    function.__name__ = name
    function.__module__ = module
    return function


def fake_botorch_rs(version="9.9.9", lcemgp=False, provided_embeddings=False, posterior_mean=False,
                    extension_path=None):
    """sys.modules entries of a stand-in botorch_rs package and a record of the calls it got.

    Every other botorch_rs.* module maps to None, so a real botorch_rs imported by an earlier
    test (test_contextual_integration) cannot leak into the stand-in.
    """
    calls = {"set_num_threads": [], "samplers": []}
    threads = {"n": 22}
    modules = {}

    def add(name, **attrs):
        module = types.ModuleType(name)
        module.__path__ = []  # a package: unknown submodules raise ModuleNotFoundError
        module.__dict__.update(attrs)
        modules[name] = module
        return module

    def set_num_threads(n=None):
        calls["set_num_threads"].append(n)
        threads["n"] = n or 22

    class SobolQMCNormalSampler:
        def __init__(self, sample_shape, seed=None):
            calls["samplers"].append((sample_shape, seed))
            self.seed = seed

    root = add("botorch_rs", __version__=version, set_num_threads=set_num_threads,
               get_num_threads=lambda: threads["n"])
    root.__file__ = os.path.join(tempfile.gettempdir(), "botorch_rs", "__init__.py")
    add("botorch_rs._botorch_rs", __file__=extension_path)
    add("botorch_rs.acquisition")
    add("botorch_rs.acquisition.logei",
        qLogNoisyExpectedImprovement=_named("qLogNoisyExpectedImprovement", "botorch_rs.acquisition.logei"))
    add("botorch_rs.acquisition.multi_objective")
    add("botorch_rs.acquisition.multi_objective.logei",
        qLogNoisyExpectedHypervolumeImprovement=_named(
            "qLogNoisyExpectedHypervolumeImprovement", "botorch_rs.acquisition.multi_objective.logei"))
    add("botorch_rs.fit", fit_gpytorch_mll=_named("fit_gpytorch_mll", "botorch_rs.fit"))
    add("botorch_rs.optim")
    add("botorch_rs.optim.optimize", optimize_acqf=_named("optimize_acqf", "botorch_rs.optim.optimize"))
    add("botorch_rs.sampling")
    add("botorch_rs.sampling.normal", SobolQMCNormalSampler=SobolQMCNormalSampler)
    add("botorch_rs.mlls", ExactMarginalLogLikelihood=_named("ExactMarginalLogLikelihood", "botorch_rs.mlls"),
        SumMarginalLogLikelihood=_named("SumMarginalLogLikelihood", "botorch_rs.mlls"))
    models = add("botorch_rs.models", SingleTaskGP=_named("SingleTaskGP", "botorch_rs.models"),
                 ModelListGP=_named("ModelListGP", "botorch_rs.models"))
    if lcemgp:
        contextual = add("botorch_rs.models.contextual_multioutput",
                         LCEMGP=_named("LCEMGP", "botorch_rs.models.contextual_multioutput"))
        if provided_embeddings:
            contextual.ProvidedEmbeddingLCEMGP = _named(
                "ProvidedEmbeddingLCEMGP", "botorch_rs.models.contextual_multioutput")
    if posterior_mean:
        add("botorch_rs.models.deterministic",
            PosteriorMeanModel=_named("PosteriorMeanModel", "botorch_rs.models.deterministic"))
    root.models = models
    optional = ("botorch_rs.models.contextual_multioutput", "botorch_rs.models.deterministic")
    for name in (*optional, *sys.modules):
        if (name == "botorch_rs" or name.startswith("botorch_rs.")) and name not in modules:
            modules[name] = None
    return modules, calls


@contextlib.contextmanager
def environment(**values):
    """os.environ with BO_MODEL_LIBRARY / BO_TORCH_THREADS set (None: removed), restored after."""
    with mock.patch.dict(os.environ):
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class RequestTests(unittest.TestCase):
    def test_default_is_botorch(self):
        for raw in (None, "", "   "):
            self.assertEqual(bo_library.normalize_request(raw), "botorch")
        self.assertEqual(bo_library.requested_library({}), "botorch")

    def test_spellings(self):
        cases = {"botorch": "botorch", " BoTorch ": "botorch", "botorch_rs": "botorch_rs",
                 "BOTORCH_RS": "botorch_rs", "botorch-rust": "botorch_rs", "botorch_rust": "botorch_rs",
                 "botorch-rs": "botorch_rs", "auto": "auto", "Auto": "auto"}
        for raw, expected in cases.items():
            self.assertEqual(bo_library.requested_library({"BO_MODEL_LIBRARY": raw}), expected, raw)

    def test_unknown_value_is_refused_with_the_choices(self):
        with self.assertRaisesRegex(bo_library.ModelLibraryError, r"BO_MODEL_LIBRARY must be one of .*'auto'.*'torch'"):
            bo_library.requested_library({"BO_MODEL_LIBRARY": "torch"})


class ResolveTests(unittest.TestCase):
    def test_botorch_never_looks_for_botorch_rs(self):
        with mock.patch.object(bo_library, "botorch_rs_status", side_effect=AssertionError("probed")):
            self.assertEqual(bo_library.resolve("botorch", contextual="manual", meta=True)[0], "botorch")

    def test_botorch_rs_requested_but_missing_is_an_error(self):
        with mock.patch.dict(sys.modules, NOT_INSTALLED):
            with self.assertRaisesRegex(bo_library.ModelLibraryError, "not installed.*README 8.16"):
                bo_library.resolve("botorch_rs")

    def test_auto_without_botorch_rs_falls_back_silently(self):
        with mock.patch.dict(sys.modules, NOT_INSTALLED):
            library, reason = bo_library.resolve("auto")
            self.assertEqual(library, "botorch")
            self.assertIn("not installed", reason)
            self.assertIsNone(bo_library.fallback_message("auto", library, reason))

    def test_plain_runs_use_an_installed_botorch_rs(self):
        modules, _ = fake_botorch_rs(version="1.2.3")
        with mock.patch.dict(sys.modules, modules):
            for requested in ("botorch_rs", "auto"):
                library, reason = bo_library.resolve(requested)
                self.assertEqual(library, "botorch_rs")
                self.assertIn("botorch_rs 1.2.3", reason)
                self.assertIsNone(bo_library.fallback_message(requested, library, reason))

    def test_contextual_mode_needs_botorch_rs_lcemgp(self):
        for lcemgp, expected in ((False, "botorch"), (True, "botorch_rs")):
            modules, _ = fake_botorch_rs(lcemgp=lcemgp)
            with mock.patch.dict(sys.modules, modules):
                for contextual in ("learned", True):
                    library, reason = bo_library.resolve("botorch_rs", contextual=contextual)
                    self.assertEqual(library, expected, (lcemgp, contextual))
                    if not lcemgp:
                        self.assertIn("botorch_rs.models.contextual_multioutput.LCEMGP", reason)
                        message = bo_library.fallback_message("auto", library, reason)
                        self.assertTrue(message.startswith("Warning: the model library falls back to BoTorch"))

    def test_provided_embeddings_need_botorch_rs_provided_embedding_lcemgp(self):
        for provided, expected in ((False, "botorch"), (True, "botorch_rs")):
            modules, _ = fake_botorch_rs(lcemgp=True, provided_embeddings=provided)
            with mock.patch.dict(sys.modules, modules):
                for source in ("manual", "image"):
                    library, reason = bo_library.resolve("botorch_rs", contextual=source)
                    self.assertEqual(library, expected, (provided, source))
                    if not provided:
                        self.assertIn(bo_library.PROVIDED_EMBEDDINGS_REASON, reason)
                # learned embeddings only need LCEMGP
                self.assertEqual(bo_library.resolve("botorch_rs", contextual="learned")[0], "botorch_rs")

    def test_provided_embedding_model_has_no_botorch_name(self):
        with self.assertRaisesRegex(AttributeError, "no botorch counterpart"):
            bo_library.namespace("botorch").ProvidedEmbeddingLCEMGP

    def test_metataf_needs_botorch_rs_posterior_mean_model(self):
        for posterior_mean, expected in ((False, "botorch"), (True, "botorch_rs")):
            modules, _ = fake_botorch_rs(posterior_mean=posterior_mean)
            with mock.patch.dict(sys.modules, modules):
                library, reason = bo_library.resolve("auto", meta=True)
                self.assertEqual(library, expected)
                if not posterior_mean:
                    self.assertIn("PosteriorMeanModel", reason)

    def test_a_broken_botorch_rs_build_is_reported_not_raised(self):
        class _BrokenBuild:  # a meta-path finder failing like an extension that cannot load
            def find_spec(self, name, path=None, target=None):
                if name == "botorch_rs":
                    raise OSError("DLL load failed")
                return None

        with mock.patch.dict(sys.modules), mock.patch.object(sys, "meta_path", [_BrokenBuild(), *sys.meta_path]):
            sys.modules.pop("botorch_rs", None)
            installed, version, detail = bo_library.botorch_rs_status()
        self.assertFalse(installed)
        self.assertIn("failed to import (OSError: DLL load failed)", detail)


class NamespaceTests(unittest.TestCase):
    def test_botorch_names_come_from_botorch(self):
        fit = _named("fit_gpytorch_mll", "botorch.fit")
        mlls = _named("ExactMarginalLogLikelihood", "gpytorch.mlls")
        fakes = {"botorch.fit": types.SimpleNamespace(fit_gpytorch_mll=fit),
                 "gpytorch.mlls": types.SimpleNamespace(ExactMarginalLogLikelihood=mlls)}
        with mock.patch.dict(sys.modules, fakes):
            names = bo_library.namespace("botorch")
            self.assertIs(names.fit_gpytorch_mll, fit)
            self.assertIs(names.ExactMarginalLogLikelihood, mlls)

    def test_botorch_rs_names_come_from_botorch_rs(self):
        modules, _ = fake_botorch_rs()
        with mock.patch.dict(sys.modules, modules):
            names = bo_library.namespace("botorch_rs")
            for name in ("qLogNoisyExpectedImprovement", "qLogNoisyExpectedHypervolumeImprovement",
                         "fit_gpytorch_mll", "optimize_acqf", "SingleTaskGP", "ModelListGP",
                         "ExactMarginalLogLikelihood", "SumMarginalLogLikelihood"):
                self.assertTrue(getattr(names, name).__module__.startswith("botorch_rs."), name)

    def test_botorch_rs_design_is_botorchs_sobol_not_botorch_rs(self):
        modules, _ = fake_botorch_rs()
        modules["botorch_rs.utils"] = types.ModuleType("botorch_rs.utils")
        modules["botorch_rs.utils.sampling"] = types.SimpleNamespace(
            draw_sobol_samples=_named("draw_sobol_samples", "botorch_rs.utils.sampling"))
        with mock.patch.dict(sys.modules, modules):
            self.assertIs(bo_library.namespace("botorch_rs").draw_sobol_samples, bo_library.draw_sobol_samples)

    def test_botorch_rs_sampler_takes_negative_seeds(self):
        modules, calls = fake_botorch_rs()
        with mock.patch.dict(sys.modules, modules):
            make = bo_library.namespace("botorch_rs").SobolQMCNormalSampler
            make(sample_shape=(8,), seed=-1)
            make((8,), seed=3)
            make((8,))
        self.assertEqual(calls["samplers"], [((8,), 2**64 - 1), ((8,), 3), ((8,), None)])

    def test_unknown_names_and_libraries(self):
        with self.assertRaises(AttributeError):
            bo_library.namespace("botorch").SaasFullyBayesianSingleTaskGP
        with self.assertRaises(ValueError):
            bo_library.namespace("auto")


class OptionsAndThreadsTests(unittest.TestCase):
    BASE = {"batch_limit": 5, "init_batch_limit": 128, "maxiter": 200}

    def test_botorch_options_are_unchanged(self):
        base = dict(self.BASE)
        options = bo_library.acqf_options("botorch", base, seed=3, iteration=2)
        self.assertEqual(options, self.BASE)
        self.assertIsNot(options, base)

    def test_botorch_rs_options_carry_seed_plus_iteration(self):
        base = dict(self.BASE)
        self.assertEqual(bo_library.acqf_options("botorch_rs", base, seed=3, iteration=2), dict(self.BASE, seed=5))
        self.assertEqual(base, self.BASE)
        self.assertEqual(bo_library.acqf_options("botorch_rs", base, seed=-5, iteration=1)["seed"], 2**64 - 4)

    def test_seed_mapping(self):
        self.assertEqual([bo_library.to_rs_seed(s) for s in (0, 3, -1, 2**64 + 5)], [0, 3, 2**64 - 1, 5])

    def test_threads_from_environment(self):
        self.assertIsNone(bo_library.threads_from_env({}))
        self.assertIsNone(bo_library.threads_from_env({"BO_TORCH_THREADS": " "}))
        self.assertEqual(bo_library.threads_from_env({"BO_TORCH_THREADS": "3"}), 3)
        for bad in ("0", "-2", "two"):
            with self.assertRaisesRegex(ValueError, "BO_TORCH_THREADS"):
                bo_library.threads_from_env({"BO_TORCH_THREADS": bad})

    def test_default_threads_leave_cores_to_unity(self):
        self.assertEqual([bo_library.default_rs_threads(n) for n in (1, 2, 4, 6, 8, 22, 64)], [1, 1, 2, 3, 4, 4, 4])
        self.assertEqual(bo_library.default_rs_threads(), max(1, min(4, (os.cpu_count() or 2) // 2)))

    def test_configure_threads(self):
        with mock.patch.dict(sys.modules, NOT_INSTALLED):
            self.assertIsNone(bo_library.configure_rs_threads("botorch", 3))  # never imports botorch_rs
        modules, calls = fake_botorch_rs()
        with mock.patch.dict(sys.modules, modules):
            self.assertEqual(bo_library.configure_rs_threads("botorch_rs", 3), 3)
            self.assertEqual(bo_library.configure_rs_threads("botorch_rs"), bo_library.default_rs_threads())
        self.assertEqual(calls["set_num_threads"], [3, bo_library.default_rs_threads()])


class ProvenanceTests(unittest.TestCase):
    def test_botorch_record(self):
        record = bo_library.library_record("auto", "botorch", "why", None)
        self.assertEqual((record["requested"], record["library"], record["reason"], record["fallback"]),
                         ("auto", "botorch", "why", True))
        self.assertIsNone(record["botorch_rs"])
        self.assertFalse(bo_library.library_record("botorch", "botorch", "why")["fallback"])
        self.assertEqual(bo_library.describe(record), f"botorch {record['version'] or ''}".strip())

    def test_botorch_rs_record_hashes_the_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            extension = pathlib.Path(tmp) / "_botorch_rs.pyd"
            extension.write_bytes(b"\x00rust\x01")
            modules, _ = fake_botorch_rs(version="0.1.0-dev", extension_path=str(extension))
            with mock.patch.dict(sys.modules, modules):
                record = bo_library.library_record("botorch_rs", "botorch_rs", "why", 4)
        self.assertEqual(record["version"], "0.1.0-dev")
        self.assertEqual(record["botorch_rs"]["extension_sha256"], hashlib.sha256(b"\x00rust\x01").hexdigest())
        self.assertEqual(bo_library.describe(record), "botorch_rs 0.1.0-dev, 4 threads")

    def test_run_info(self):
        info = bo_library.run_info("bo.py", 3, {"library": "botorch"})
        self.assertEqual(info["schema_version"], 1)
        self.assertEqual((info["generator"], info["seed"], info["model_library"]), ("bo.py", 3, {"library": "botorch"}))
        self.assertIn("botorch-rust", info["library_versions"])
        for key in ("created_utc", "python", "platform"):
            self.assertIn(key, info)

    def test_bound_library(self):
        rs = _named("optimize_acqf", "botorch_rs.optim.optimize")
        bt = _named("optimize_acqf", "botorch.optim.optimize")
        gp = _named("ExactMarginalLogLikelihood", "gpytorch.mlls")
        self.assertEqual(bo_library.bound_library([rs]), "botorch_rs")
        self.assertEqual(bo_library.bound_library([bt, gp]), "botorch")
        self.assertIsNone(bo_library.bound_library([rs, bt]))
        self.assertIsNone(bo_library.bound_library([]))

    def test_openbo_library(self):
        mobo_botorch = types.ModuleType("openbo.optimizers.mobo_botorch")
        # Isolate every inspected module, including when real openbo is installed.
        modules = {name: None for name in bo_library.OPENBO_LIBRARY_MODULES}
        modules["openbo.optimizers.mobo_botorch"] = mobo_botorch
        with mock.patch.dict(sys.modules, modules):
            self.assertIsNone(bo_library.openbo_library())
            mobo_botorch.optimize_acqf = _named("optimize_acqf", "botorch.optim.optimize")
            self.assertEqual(bo_library.openbo_library(), "botorch")
            mobo_botorch.MODEL_LIBRARY = "botorch_rs"  # an explicit declaration wins
            self.assertEqual(bo_library.openbo_library(), "botorch_rs")


class _BackendMixin:
    """main() of a stub-loaded backend with BO_MODEL_LIBRARY set; prints are captured."""

    MODULE = None  # "bo" or "mobo"

    def _load(self):
        return load_bo_module() if self.MODULE == "bo" else load_mobo_module()

    def _init(self, context=None):
        msg = {
            "type": "init",
            "config": {"numSamplingIterations": 2, "numOptimizationIterations": 1, "batchSize": 1,
                       "numRestarts": 3, "rawSamples": 16, "mcSamples": 8, "seed": 5,
                       "nParameters": 1, "nObjectives": 1 if self.MODULE == "bo" else 2,
                       "warmStart": False},
            "parameters": [{"key": "p0", "init": {"low": 0.0, "high": 1.0}}],
            "objectives": [{"key": f"o{j}", "init": {"low": 0.0, "high": 1.0, "minimize": 0}}
                           for j in range(1 if self.MODULE == "bo" else 2)],
            "user": {"userId": "u", "conditionId": "c", "groupId": "g"},
        }
        if context is not None:
            msg["context"] = context
        return msg

    def _main(self, module, init_msg, modules=None, **env):
        server = FakeServerSocket(FakeConn([json_line(init_msg)]))
        execute = "bo_execute" if self.MODULE == "bo" else "mobo_execute"
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(environment(**env))
            if modules:
                stack.enter_context(mock.patch.dict(sys.modules, modules))
            stack.enter_context(mock.patch.object(module.socket, "socket", lambda *a, **k: server))
            stack.enter_context(mock.patch.object(module, execute, lambda *a, **k: None))
            stack.enter_context(contextlib.redirect_stdout(out))
            module.main()
        return out.getvalue()

    def test_default_keeps_botorch_names(self):
        module = self._load()
        before = {name: getattr(module, name) for name in module.MODEL_LIBRARY_NAMES}
        out = self._main(module, self._init(), BO_MODEL_LIBRARY=None)
        self.assertEqual(module.MODEL_LIBRARY, "botorch")
        self.assertEqual({name: getattr(module, name) for name in module.MODEL_LIBRARY_NAMES}, before)
        self.assertIn("Model library: botorch", out)
        self.assertIn("'MODEL_LIBRARY': 'botorch", out)
        self.assertEqual(module.MODEL_LIBRARY_INFO["requested"], "botorch")

    def test_missing_botorch_rs_is_a_startup_error(self):
        module = self._load()
        with self.assertRaisesRegex(bo_library.ModelLibraryError, "not installed"):
            self._main(module, self._init(), NOT_INSTALLED, BO_MODEL_LIBRARY="botorch_rs")

    def test_auto_without_botorch_rs_runs_botorch(self):
        module = self._load()
        out = self._main(module, self._init(), NOT_INSTALLED, BO_MODEL_LIBRARY="auto")
        self.assertEqual(module.MODEL_LIBRARY, "botorch")
        self.assertNotIn("Warning", out)
        self.assertIn("not installed", out)

    def test_botorch_rs_binds_its_names(self):
        module = self._load()
        modules, _ = fake_botorch_rs()
        out = self._main(module, self._init(), modules, BO_MODEL_LIBRARY="botorch_rs", BO_TORCH_THREADS=None)
        self.assertEqual(module.MODEL_LIBRARY, "botorch_rs")
        for name in module.MODEL_LIBRARY_NAMES:
            if name == "draw_sobol_samples":
                self.assertIs(module.draw_sobol_samples, bo_library.draw_sobol_samples)
            elif name != "SobolQMCNormalSampler":
                self.assertTrue(getattr(module, name).__module__.startswith("botorch_rs."), name)
        self.assertIn("'MODEL_LIBRARY': 'botorch_rs 9.9.9", out)

    def test_contextual_run_without_lcemgp_falls_back_with_a_warning(self):
        module = self._load()
        modules, _ = fake_botorch_rs(lcemgp=False)
        context = {"enabled": True, "currentContext": "a", "embeddingSource": "learned",
                   "contexts": [{"key": "a"}, {"key": "b"}]}
        out = self._main(module, self._init(context), modules, BO_MODEL_LIBRARY="botorch_rs")
        self.assertEqual(module.MODEL_LIBRARY, "botorch")
        self.assertIn("Warning: the model library falls back to BoTorch", out)
        self.assertTrue(module.MODEL_LIBRARY_INFO["fallback"])

    def test_options_carry_the_seed_on_botorch_rs_only(self):
        module = self._load()
        captured = []

        def fake_optimize_acqf(acq_function, bounds, q, num_restarts, raw_samples, options, sequential):
            captured.append(options)
            return FakeTensor([[0.5]]), None

        module.optimize_acqf = fake_optimize_acqf
        module.problem_bounds = FakeTensor([[0.0], [1.0]])
        module.SEED, module.OPTIMIZATION_ITERATION = 5, 2
        acq = "qLogNoisyExpectedImprovement" if self.MODULE == "bo" else "qLogNoisyExpectedHypervolumeImprovement"
        setattr(module, acq, lambda **kwargs: object())
        if self.MODULE == "mobo":
            module.ref_point = FakeTensor([-1.1, -1.1])
        optimize = module.optimize_candidates if self.MODULE == "bo" else module.optimize_qnehvi
        for library in ("botorch", "botorch_rs"):
            module.MODEL_LIBRARY = library
            optimize(model=None, sampler=None, X_baseline=FakeTensor([[0.1]]))
        self.assertNotIn("seed", captured[0])
        self.assertEqual(captured[1]["seed"], 7)
        self.assertEqual({k: v for k, v in captured[1].items() if k != "seed"}, captured[0])

    def test_run_info_is_written_to_the_run_folder(self):
        module = self._load()
        modules, _ = fake_botorch_rs(version="4.5.6")
        with mock.patch.dict(sys.modules, modules), quiet():
            module.select_model_library("botorch_rs")
        with tempfile.TemporaryDirectory() as tmp:
            module.PROJECT_PATH = tmp
            module.SEED = 11
            module.write_run_info()
            info = json.loads((pathlib.Path(tmp) / "RunInfo.json").read_text(encoding="utf-8"))
        self.assertEqual(info["generator"], f"Assets/StreamingAssets/BOData/BayesianOptimization/{self.MODULE}.py")
        self.assertEqual(info["seed"], 11)
        self.assertEqual((info["model_library"]["library"], info["model_library"]["version"]), ("botorch_rs", "4.5.6"))


class BoBackendTests(_BackendMixin, unittest.TestCase):
    MODULE = "bo"

    def test_botorch_rs_threads_follow_bo_torch_threads(self):
        module = self._load()
        modules, calls = fake_botorch_rs()
        module.TORCH_THREADS = 3
        self._main(module, self._init(), modules, BO_MODEL_LIBRARY="botorch_rs")
        self.assertEqual(calls["set_num_threads"], [3])
        self.assertEqual(module.MODEL_LIBRARY_INFO["threads"], 3)


class MoboBackendTests(_BackendMixin, unittest.TestCase):
    MODULE = "mobo"

    def test_botorch_rs_threads(self):
        for env, expected in ((None, bo_library.default_rs_threads()), ("2", 2)):
            module = self._load()
            modules, calls = fake_botorch_rs()
            self._main(module, self._init(), modules, BO_MODEL_LIBRARY="botorch_rs", BO_TORCH_THREADS=env)
            self.assertEqual(calls["set_num_threads"], [expected])

    def test_botorch_keeps_torch_default_threads(self):
        module = self._load()
        calls = []
        module.torch.set_num_threads = calls.append
        self._main(module, self._init(), BO_MODEL_LIBRARY=None, BO_TORCH_THREADS="2")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
