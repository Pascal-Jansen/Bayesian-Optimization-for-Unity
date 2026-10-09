"""End-to-end contextual BO/MOBO tests against the real botorch stack.

These tests exercise the LCE-M GP (LCEMGP) integration with definable context
embeddings, including a simulated Unity objective stream. They are skipped
automatically when torch/botorch/moocore are not installed (e.g. on the
lightweight CI environment) and run locally in a full dev environment.

Other test modules in this suite replace torch/botorch in ``sys.modules`` with
stubs. The real module objects are captured here at import time (before any
test runs) and restored around each test.
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

from _stubs import reset_protocol_state  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization"

_REAL_MODULE_ROOTS = ("torch", "botorch", "gpytorch", "linear_operator", "moocore", "pandas")

try:
    import torch  # noqa: F401
    import botorch  # noqa: F401
    import botorch.acquisition.logei  # noqa: F401
    import botorch.acquisition.multi_objective.logei  # noqa: F401
    import botorch.fit  # noqa: F401
    import botorch.models  # noqa: F401
    import botorch.models.contextual_multioutput  # noqa: F401
    import botorch.optim.optimize  # noqa: F401
    import botorch.sampling.normal  # noqa: F401
    import botorch.utils.sampling  # noqa: F401
    import gpytorch.mlls  # noqa: F401
    import moocore  # noqa: F401
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
    """Replace stub torch/botorch/... entries with the captured real modules.

    Stub modules (created via ``types.ModuleType`` by other test files) have no
    ``__spec__``; real modules imported lazily after the snapshot are left
    untouched so C extensions are never re-imported.
    """
    sys.modules.update(_REAL_MODULES)
    for name in list(sys.modules):
        if name.split(".")[0] not in _REAL_MODULE_ROOTS or name in _REAL_MODULES:
            continue
        if getattr(sys.modules[name], "__spec__", None) is None:
            del sys.modules[name]


def load_backend_module(filename):
    restore_real_modules()
    name = f"{pathlib.Path(filename).stem}_integration_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BACKEND_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reset_protocol_state()
    return module


class _FakeConn:
    """Feeds scripted 'objectives' NDJSON messages and records sent lines."""

    def __init__(self, objective_payloads):
        self._pending = [
            (json.dumps({"type": "objectives", "values": payload}) + "\n").encode("utf-8")
            for payload in objective_payloads
        ]
        self.sent = []

    def recv(self, n):
        if not self._pending:
            return b""
        return self._pending.pop(0)

    def sendall(self, data):
        self.sent.append(data.decode("utf-8"))

    def sent_messages(self):
        msgs = []
        for chunk in self.sent:
            for line in chunk.splitlines():
                if line.strip():
                    msgs.append(json.loads(line))
        return msgs


def _write_warmstart_csvs(init_dir, objective_names):
    """Two-parameter warm start covering contexts user_A / user_B (current)."""
    import pandas as pd

    x_rows = [
        # Context, p0, p1  (raw scale [0, 1])
        ("user_A", 0.10, 0.90),
        ("user_A", 0.80, 0.20),
        ("user_A", 0.45, 0.55),
        ("user_B", 0.25, 0.60),
        ("user_B", 0.70, 0.35),
        ("user_B", 0.50, 0.50),
    ]
    x_df = pd.DataFrame(x_rows, columns=["Context", "p0", "p1"])
    # raw objective scale [0, 10]
    y_data = np.array([
        [2.0, 8.0],
        [7.0, 3.0],
        [5.0, 5.0],
        [3.0, 7.0],
        [6.0, 4.0],
        [5.5, 5.5],
    ])
    y_df = pd.DataFrame(y_data[:, : len(objective_names)], columns=objective_names)
    x_df.to_csv(init_dir / "params.csv", sep=";", index=False)
    y_df.to_csv(init_dir / "objs.csv", sep=";", index=False)


def _context_init_msg():
    return {
        "context": {
            "enabled": True,
            "currentContext": "user_B",
            "embeddingSource": "manual",
            "normalizeEmbeddings": True,
            "contexts": [
                {"key": "user_A", "embedding": [0.9, 0.1, 0.3]},
                {"key": "user_B", "embedding": [0.85, 0.15, 0.35]},
            ],
        }
    }


def _new_participant_context_init_msg():
    """README 8.13's main case: warm start from user_A/user_B, current user_C has no data."""
    msg = _context_init_msg()
    msg["context"]["currentContext"] = "user_C"
    msg["context"]["contexts"].append({"key": "user_C", "embedding": [0.8, 0.2, 0.3]})
    return msg


def _configure_common(module, num_objs, context_msg=None):
    import torch

    module.USER_ID = "u"
    module.CONDITION_ID = "c"
    module.GROUP_ID = "g"
    module.USER_LOG_ID = "u"
    module.CONDITION_LOG_ID = "c"
    module.WARM_START = True
    module.CSV_PATH_PARAMETERS = "params.csv"
    module.CSV_PATH_OBJECTIVES = "objs.csv"
    module.WARM_START_OBJECTIVE_FORMAT = "auto"
    module.SEED = 3
    module.PROBLEM_DIM = 2
    module.NUM_OBJS = num_objs
    module.BATCH_SIZE = 1
    module.NUM_RESTARTS = 2
    module.RAW_SAMPLES = 16
    module.MC_SAMPLES = 16
    module.parameter_names = ["p0", "p1"]
    module.parameters_info = [(0.0, 1.0), (0.0, 1.0)]
    module.objective_names = ["o0", "o1"][:num_objs]
    module.objectives_info = [(0.0, 10.0, 0), (0.0, 10.0, 0)][:num_objs]
    module.problem_bounds = torch.stack(
        [torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double)], dim=0
    )
    module.ref_point = torch.full(
        (num_objs,), module.bo_normalize.HYPERVOLUME_REFERENCE_VALUE, dtype=torch.double
    )
    setup = module.context_support.parse_context_config(context_msg or _context_init_msg())
    module.context_support.resolve_embeddings(setup)
    module.CONTEXT_SETUP = setup
    return setup


@unittest.skipUnless(HAS_REAL_STACK, "torch/botorch/moocore not installed")
class ContextualModelTests(unittest.TestCase):
    def setUp(self):
        restore_real_modules()

    def test_lcemgp_manual_embeddings_fit_and_predict(self):
        import torch
        from botorch.fit import fit_gpytorch_mll

        cs = load_backend_module("context_support.py")
        setup = cs.parse_context_config(_context_init_msg())
        cs.resolve_embeddings(setup)

        torch.manual_seed(0)
        x = torch.rand(10, 2, dtype=torch.double)
        xt = cs.append_task_column(x, np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
        y = -((x - 0.4) ** 2).sum(dim=-1, keepdim=True)

        mll, model = cs.build_contextual_model(xt, y, setup)
        fit_gpytorch_mll(mll)
        posterior = model.posterior(torch.rand(4, 2, dtype=torch.double))
        self.assertEqual(tuple(posterior.mean.shape), (4, 1))

    def test_lcemgp_learned_embeddings_fit_and_predict(self):
        import torch
        from botorch.fit import fit_gpytorch_mll

        cs = load_backend_module("context_support.py")
        msg = _context_init_msg()
        msg["context"]["embeddingSource"] = "learned"
        setup = cs.parse_context_config(msg)
        cs.resolve_embeddings(setup)

        torch.manual_seed(0)
        x = torch.rand(8, 2, dtype=torch.double)
        xt = cs.append_task_column(x, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        y = torch.cat(
            [
                -((x - 0.4) ** 2).sum(dim=-1, keepdim=True),
                -((x - 0.6) ** 2).sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )

        mll, model = cs.build_contextual_model(xt, y, setup)
        fit_gpytorch_mll(mll)
        posterior = model.posterior(torch.rand(3, 2, dtype=torch.double))
        self.assertEqual(tuple(posterior.mean.shape), (3, 2))


    def _fit_context_covariance(self, cs, embeddings, source, seed, current, y_by_context=None):
        """Fit an LCE-M GP on contexts c0 (y = sum x) and c1 (y = -sum x); return the context
        covariance and the posterior mean of the current context at x = (0.9, 0.9)."""
        import contextlib
        import io

        from botorch.fit import fit_gpytorch_mll

        keys = [f"c{i}" for i in range(len(embeddings) if embeddings is not None else 3)]
        torch.manual_seed(seed)
        setup = cs.ContextSetup(keys=keys, current_key=current, embedding_source=source,
                                normalize_embeddings=True,
                                manual_embeddings=None if embeddings is None else np.asarray(embeddings, float))
        with contextlib.redirect_stdout(io.StringIO()):
            cs.resolve_embeddings(setup)
        g = torch.Generator().manual_seed(100)
        x = torch.rand(20, 2, generator=g, dtype=torch.double)
        tasks = np.array([0] * 10 + [1] * 10)
        y = torch.where(torch.tensor(tasks)[:, None] == 0, x.sum(-1, keepdim=True), -x.sum(-1, keepdim=True))
        mll, model = cs.build_contextual_model(cs.append_task_column(x, tasks), y, setup)
        fit_gpytorch_mll(mll)
        model.eval()
        covar = model._eval_context_covar().to_dense().detach()
        mean = model.posterior(torch.tensor([[0.9, 0.9]], dtype=torch.double)).mean.item()
        return covar, mean, model

    def test_new_context_similarity_comes_from_provided_embeddings_only(self):
        # Stock LCEMGP concatenated an untrained learned embedding to the provided one, so a
        # new participant's similarity to the others depended on the torch seed (corr(c1, c2)
        # for identical manual embeddings, seeds 0-5: 0.00, 0.66, 1.00, 1.00, 0.99, 0.51).
        cs = load_backend_module("context_support.py")
        same = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]   # c2 (no data) == c1
        near = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.9, 0.0]]   # c2 (no data) ~ c1
        for embeddings in (same, near):
            corrs, means = [], []
            for seed in range(4):
                covar, mean, model = self._fit_context_covariance(cs, embeddings, "manual", seed, "c2")
                corrs.append(covar[1, 2].item())
                means.append(mean)
            self.assertEqual(len(model.emb_layers), 0)
            self.assertEqual(model.task_covar_module_base.lengthscale.shape[-1], 3)
            self.assertLess(max(corrs) - min(corrs), 1e-6, corrs)
            self.assertLess(max(means) - min(means), 1e-6, means)
            # c2 is far more like c1 than like c0 ...
            self.assertGreater(covar[1, 2].item(), covar[0, 2].item() + 0.5, covar)
            # ... and inherits c1's behaviour (y = -sum x = -1.8 at (0.9, 0.9)), not c0's.
            self.assertLess(max(means), -0.3, means)
            if embeddings is same:
                # Identical embeddings: correlated up to the learned context-specific share.
                self.assertAlmostEqual(corrs[-1], 1.0 - model.context_specific_weight.item(), places=9)

    def test_context_with_a_twin_embedding_follows_its_own_data(self):
        # c1 (current, y = -sum x) has the same manual embedding as c0 (y = +sum x), e.g. two
        # participants of the same age. With the embeddings as the only context similarity, the
        # two were perfectly correlated whatever their data: c1's posterior after 10 own
        # observations ranked (0.9, 0.9) above (0.1, 0.1) for every seed (-0.27 vs -1.81; truth
        # -1.8 vs -0.2). The learned context-specific share lets the data separate them.
        cs = load_backend_module("context_support.py")
        for seed in range(3):
            _, mean_hi, model = self._fit_context_covariance(cs, [[25.0], [25.0], [40.0]], "manual", seed, "c1")
            mean_lo = model.posterior(torch.tensor([[0.1, 0.1]], dtype=torch.double)).mean.item()
            self.assertLess(abs(mean_hi - (-1.8)), 0.2, (seed, mean_hi, mean_lo))
            self.assertLess(abs(mean_lo - (-0.2)), 0.2, (seed, mean_hi, mean_lo))
            self.assertGreater(model.context_specific_weight.item(), 0.5)

    def test_unobserved_context_takes_the_observed_mean_level(self):
        # MultiTaskGP fits one constant mean per context; an unobserved context's constant stayed
        # at 0, so a new participant with the same embedding as the only observed one (correlation
        # 1) was predicted with an offset (RMSE 0.52 against 0.09 for the observed twin).
        import contextlib
        import io

        from botorch.fit import fit_gpytorch_mll

        cs = load_backend_module("context_support.py")
        g = torch.Generator().manual_seed(0)
        x = torch.rand(8, 2, generator=g, dtype=torch.double)
        y = 1 - 2 * ((x - 0.3) ** 2).sum(-1, keepdim=True)
        x_test = torch.rand(50, 2, generator=torch.Generator().manual_seed(1), dtype=torch.double)
        means = {}
        for current in ("c0", "c1"):
            torch.manual_seed(0)
            setup = cs.ContextSetup(keys=["c0", "c1"], current_key=current, embedding_source="manual",
                                    normalize_embeddings=True, manual_embeddings=np.array([[1.0, 2.0], [1.0, 2.0]]))
            with contextlib.redirect_stdout(io.StringIO()):
                cs.resolve_embeddings(setup)
            mll, model = cs.build_contextual_model(cs.append_task_column(x, np.zeros(8)), y, setup)
            fit_gpytorch_mll(mll)
            model.eval()
            means[current] = model.posterior(x_test).mean.detach()
            constants = [m.constant.item() for m in model.mean_module.base_means]
            self.assertEqual(constants[1], constants[0])
        torch.testing.assert_close(means["c1"], means["c0"], rtol=0.0, atol=1e-9)

        # Learned embeddings: the unobserved context gets the observed contexts' average level.
        _, _, model = self._fit_context_covariance(cs, None, "learned", 0, "c2")
        constants = [m.constant.item() for m in model.mean_module.base_means]
        self.assertAlmostEqual(constants[2], (constants[0] + constants[1]) / 2, places=12)

    def test_learned_embeddings_keep_one_learned_dimension(self):
        cs = load_backend_module("context_support.py")
        covar, _, model = self._fit_context_covariance(cs, None, "learned", 0, "c1")
        self.assertEqual(len(model.emb_layers), 1)
        self.assertEqual(model.task_covar_module_base.lengthscale.shape[-1], 1)
        self.assertEqual(tuple(covar.shape), (3, 3))

    def test_manual_embedding_magnitude_separates_contexts(self):
        # Ages 25 / 40 / 60 used to be L2-normalized to 1.0 each: the contexts became identical
        # (correlation 1) although c0 and c1 behave oppositely. Standardized, the model can
        # tell them apart.
        cs = load_backend_module("context_support.py")
        covar, _, _ = self._fit_context_covariance(cs, [[25.0], [40.0], [60.0]], "manual", 0, "c2")
        self.assertLess(covar[0, 1].item(), 0.5)


@unittest.skipUnless(HAS_REAL_STACK, "torch/botorch/moocore not installed")
class ImageEmbeddingGlueTests(unittest.TestCase):
    """Exercises the open_clip image-embedding glue with a mocked open_clip.

    Uses real PIL + torch so image loading, preprocessing, batching, and numpy
    conversion run for real; only the (multi-GB) vision model is faked.
    """

    def setUp(self):
        restore_real_modules()
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("pillow not installed")

    def test_embed_images_with_mocked_open_clip_model(self):
        import types

        import torch
        from PIL import Image

        cs = load_backend_module("context_support.py")

        recorded = {"model_names": [], "encoded_batches": 0}

        class _FakeVisionModel:
            def eval(self):
                return self

            def encode_image(self, batch):
                recorded["encoded_batches"] += 1
                assert batch.shape == (1, 3, 8, 8)
                return batch.reshape(1, -1)[:, :16]

        def fake_create_model_and_transforms(model_name, pretrained=None):
            recorded["model_names"].append((model_name, pretrained))
            preprocess = lambda img: torch.tensor(  # noqa: E731
                np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )
            return _FakeVisionModel(), None, preprocess

        fake_open_clip = types.ModuleType("open_clip")
        fake_open_clip.__spec__ = importlib.util.spec_from_loader("open_clip", loader=None)
        fake_open_clip.create_model_and_transforms = fake_create_model_and_transforms

        with tempfile.TemporaryDirectory() as tmp:
            img_path = pathlib.Path(tmp) / "ctx.png"
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)

            sys.modules["open_clip"] = fake_open_clip
            try:
                emb = cs._embed_images_with_open_clip(
                    [str(img_path)], model_name="ViT-bigG-14", pretrained="laion2b_s39b_b160k"
                )
            finally:
                del sys.modules["open_clip"]

        self.assertEqual(emb.shape, (1, 16))
        self.assertEqual(recorded["model_names"], [("ViT-bigG-14", "laion2b_s39b_b160k")])
        self.assertEqual(recorded["encoded_batches"], 1)
        self.assertEqual(emb.dtype, np.float64)
        # red pixels: first (R) channel slice of the flattened image is 1.0
        np.testing.assert_allclose(emb[0, :16], np.ones(16))


@unittest.skipUnless(HAS_REAL_STACK, "torch/botorch/moocore not installed")
class ContextualLoopIntegrationTests(unittest.TestCase):
    def setUp(self):
        restore_real_modules()

    def _run_in_tmp(self, module_file, num_objs, objective_payloads, new_participant=False):
        module = load_backend_module(module_file)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            init_dir = tmp_path / "InitData"
            init_dir.mkdir()
            _write_warmstart_csvs(init_dir, ["o0"] if module_file == "bo.py" else ["o0", "o1"])

            prev_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                setup = _configure_common(
                    module, num_objs,
                    _new_participant_context_init_msg() if new_participant else None,
                )
                self.assertEqual(setup.current_index, 2 if new_participant else 1)

                conn = _FakeConn(objective_payloads)
                if module_file == "bo.py":
                    metrics, train_x, train_y = module.bo_execute(
                        conn, seed=3, iterations=1, initial_samples=0
                    )
                else:
                    metrics, train_x, train_y = module.mobo_execute(
                        conn, seed=3, iterations=1, initial_samples=0
                    )

                # 6 warm-start rows + 1 optimization evaluation, task column appended
                self.assertEqual(tuple(train_x.shape), (7, 3))
                self.assertEqual(tuple(train_y.shape), (7, num_objs))
                # The new observation belongs to the current context.
                self.assertEqual(float(train_x[-1, -1].item()), float(setup.current_index))
                self.assertEqual(len(metrics), 2)

                sent = conn.sent_messages()
                types = [m.get("type") for m in sent]
                self.assertIn("parameters", types)
                self.assertIn("coverage", types)
                self.assertIn("optimization_finished", types)

                params_msg = next(m for m in sent if m.get("type") == "parameters")
                self.assertEqual(sorted(params_msg["values"].keys()), ["p0", "p1"])

                # Observation CSV must carry the Context column with the current key.
                import pandas as pd

                obs_csv = pathlib.Path(module.PROJECT_PATH) / "ObservationsPerEvaluation.csv"
                self.assertTrue(obs_csv.exists())
                df = pd.read_csv(obs_csv, delimiter=";")
                self.assertIn("Context", df.columns)
                self.assertEqual(df["Context"].tolist(), [setup.current_key])
                # Iteration counts current-context evaluations only: 3 warm-start rows for
                # user_B (none for a new participant) + 1 new optimization evaluation.
                warm_rows = 0 if new_participant else 3
                self.assertEqual(df["Iteration"].tolist(), [warm_rows + 1])

                # The metric log uses the same axis: the warm-start baseline at the last
                # current-context warm-start row, then the evaluation's own Iteration.
                metric_csv = pathlib.Path(module.PROJECT_PATH) / (
                    "BestObjectivePerEvaluation.csv" if module_file == "bo.py"
                    else "HypervolumePerEvaluation.csv"
                )
                metric_df = pd.read_csv(metric_csv, delimiter=";")
                self.assertEqual(metric_df["Iteration"].tolist(), [warm_rows, warm_rows + 1])
            finally:
                os.chdir(prev_cwd)

    def test_contextual_bo_loop_with_warm_start(self):
        self._run_in_tmp("bo.py", num_objs=1, objective_payloads=[{"o0": 6.5}])

    def test_contextual_mobo_loop_with_warm_start(self):
        self._run_in_tmp("mobo.py", num_objs=2, objective_payloads=[{"o0": 6.5, "o1": 6.0}])

    def test_contextual_bo_loop_for_a_context_without_data(self):
        # LCEMGP inherited MultiTaskGP.eval(), which crashed for unobserved contexts.
        self._run_in_tmp("bo.py", num_objs=1, objective_payloads=[{"o0": 6.5}],
                         new_participant=True)

    def test_contextual_mobo_loop_for_a_context_without_data(self):
        self._run_in_tmp("mobo.py", num_objs=2, objective_payloads=[{"o0": 6.5, "o1": 6.0}],
                         new_participant=True)

    def test_mobo_hypervolume_counts_designs_rated_worst_on_one_objective(self):
        # With the reference point at exactly [-1, -1], (1, -1) and (-1, 1) were both Pareto
        # optimal but added no hypervolume: coverage stayed 0 and qLogNEHVI saw no improvement.
        import pandas as pd

        mobo = load_backend_module("mobo.py")
        recorded_ref_points = []
        acq_class = mobo.qLogNoisyExpectedHypervolumeImprovement

        def recording_acq(*args, **kwargs):
            recorded_ref_points.append(list(kwargs["ref_point"]))
            return acq_class(*args, **kwargs)

        mobo.qLogNoisyExpectedHypervolumeImprovement = recording_acq
        with tempfile.TemporaryDirectory() as tmp:
            prev_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                mobo.USER_ID = mobo.USER_LOG_ID = "u"
                mobo.CONDITION_ID = mobo.CONDITION_LOG_ID = "c"
                mobo.GROUP_ID = "g"
                mobo.WARM_START = False
                mobo.CONTEXT_SETUP = None
                mobo.SEED = 3
                mobo.PROBLEM_DIM = 2
                mobo.NUM_OBJS = 2
                mobo.BATCH_SIZE = 1
                mobo.NUM_RESTARTS = 2
                mobo.RAW_SAMPLES = 16
                mobo.MC_SAMPLES = 16
                mobo.parameter_names = ["p0", "p1"]
                mobo.parameters_info = [(0.0, 1.0), (0.0, 1.0)]
                mobo.objective_names = ["o0", "o1"]
                mobo.objectives_info = [(0.0, 10.0, 0), (0.0, 10.0, 0)]
                mobo.problem_bounds = torch.stack(
                    [torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double)]
                )
                mobo.ref_point = mobo.reference_point(2)
                conn = _FakeConn([{"o0": 10.0, "o1": 0.0}, {"o0": 0.0, "o1": 10.0},
                                  {"o0": 5.0, "o1": 5.0}])
                hvs, _, _ = mobo.mobo_execute(conn, seed=3, iterations=1, initial_samples=2)
                run = pathlib.Path(mobo.PROJECT_PATH)
                obs = pd.read_csv(run / "ObservationsPerEvaluation.csv", delimiter=";", dtype=str)
                hv_log = pd.read_csv(run / "HypervolumePerEvaluation.csv", delimiter=";", dtype=str)
            finally:
                os.chdir(prev_cwd)

        self.assertEqual(recorded_ref_points, [[-1.1, -1.1]])
        np.testing.assert_allclose(hvs[:2], [0.1 * 2.1, 2 * 0.1 * 2.1 - 0.1 * 0.1])
        self.assertGreater(hvs[2], hvs[1])
        self.assertEqual(obs["IsPareto"].tolist()[:2], ["TRUE", "TRUE"])
        self.assertEqual(set(hv_log["ReferencePoint"]), {"[-1.1,-1.1]"})
        coverage = [m["value"] for m in conn.sent_messages() if m.get("type") == "coverage"]
        self.assertGreater(coverage[0], 0.0)

    def test_mobo_pareto_mask_of_a_single_numpy_row(self):
        # NumPy 2 arrays have .device too; the mask must stay numpy for numpy input,
        # otherwise a one-element torch bool mask indexes as the integer 1.
        mobo = load_backend_module("mobo.py")
        y = np.array([[0.1, 0.2]])
        mask = mobo.is_non_dominated(y)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(y[mask].shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
