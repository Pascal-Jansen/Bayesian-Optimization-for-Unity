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


def _configure_common(module, num_objs):
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
    module.ref_point = torch.full((num_objs,), -1.0, dtype=torch.double)
    setup = module.context_support.parse_context_config(_context_init_msg())
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

    def _run_in_tmp(self, module_file, num_objs, objective_payloads):
        module = load_backend_module(module_file)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            init_dir = tmp_path / "InitData"
            init_dir.mkdir()
            _write_warmstart_csvs(init_dir, ["o0"] if module_file == "bo.py" else ["o0", "o1"])

            prev_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                setup = _configure_common(module, num_objs)
                self.assertEqual(setup.current_index, 1)

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
                # The new observation belongs to the current context (index 1).
                self.assertEqual(float(train_x[-1, -1].item()), 1.0)
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
                self.assertEqual(df["Context"].tolist(), ["user_B"])
                # Iteration counts current-context evaluations only:
                # 3 warm-start rows for user_B + 1 new optimization evaluation.
                self.assertEqual(df["Iteration"].tolist(), [4])
            finally:
                os.chdir(prev_cwd)

    def test_contextual_bo_loop_with_warm_start(self):
        self._run_in_tmp("bo.py", num_objs=1, objective_payloads=[{"o0": 6.5}])

    def test_contextual_mobo_loop_with_warm_start(self):
        self._run_in_tmp("mobo.py", num_objs=2, objective_payloads=[{"o0": 6.5, "o1": 6.0}])


if __name__ == "__main__":
    unittest.main()
