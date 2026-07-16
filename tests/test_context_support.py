import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest
import uuid

import numpy as np
import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTEXT_SUPPORT_PATH = (
    REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization/context_support.py"
)


class FakeTensor:
    def __init__(self, data):
        self.arr = np.asarray(data, dtype=np.float64)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.arr, dtype=np.float64)

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, idx):
        out = self.arr[idx]
        if isinstance(out, np.ndarray):
            return FakeTensor(out)
        return float(out)


def install_torch_stub():
    torch_mod = types.ModuleType("torch")
    torch_mod.double = np.float64
    torch_mod.tensor = lambda data, dtype=None: FakeTensor(data)
    sys.modules["torch"] = torch_mod
    return torch_mod


def load_context_support():
    name = f"context_support_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, CONTEXT_SUPPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_init_msg(**overrides):
    context = {
        "enabled": True,
        "currentContext": "user_B",
        "embeddingSource": "manual",
        "normalizeEmbeddings": True,
        "contexts": [
            {"key": "user_A", "embedding": [3.0, 0.0]},
            {"key": "user_B", "embedding": [0.0, 4.0]},
        ],
    }
    context.update(overrides)
    return {"context": context}


class ParseContextConfigTests(unittest.TestCase):
    def setUp(self):
        self.cs = load_context_support()

    def test_missing_or_disabled_context_returns_none(self):
        self.assertIsNone(self.cs.parse_context_config({}))
        self.assertIsNone(self.cs.parse_context_config({"context": None}))
        self.assertIsNone(self.cs.parse_context_config(make_init_msg(enabled=False)))

    def test_valid_manual_config(self):
        setup = self.cs.parse_context_config(make_init_msg())
        self.assertEqual(setup.keys, ["user_A", "user_B"])
        self.assertEqual(setup.current_key, "user_B")
        self.assertEqual(setup.current_index, 1)
        self.assertEqual(setup.embedding_source, "manual")
        self.assertEqual(setup.num_contexts, 2)
        np.testing.assert_allclose(setup.manual_embeddings, np.array([[3.0, 0.0], [0.0, 4.0]]))

    def test_current_context_is_matched_case_insensitively(self):
        setup = self.cs.parse_context_config(make_init_msg(currentContext="USER_b"))
        self.assertEqual(setup.current_key, "user_B")
        self.assertEqual(setup.current_index, 1)

    def test_duplicate_context_keys_raise(self):
        msg = make_init_msg(contexts=[
            {"key": "user_A", "embedding": [1.0]},
            {"key": "USER_a", "embedding": [2.0]},
        ])
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(msg)

    def test_empty_context_list_raises(self):
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(make_init_msg(contexts=[]))

    def test_empty_key_raises(self):
        msg = make_init_msg(contexts=[{"key": "  ", "embedding": [1.0]}])
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(msg)

    def test_missing_current_context_raises(self):
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(make_init_msg(currentContext=""))

    def test_unknown_current_context_raises(self):
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(make_init_msg(currentContext="user_Z"))

    def test_invalid_embedding_source_raises(self):
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(make_init_msg(embeddingSource="magic"))

    def test_manual_embedding_length_mismatch_raises(self):
        msg = make_init_msg(contexts=[
            {"key": "user_A", "embedding": [1.0, 2.0]},
            {"key": "user_B", "embedding": [1.0]},
        ])
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(msg)

    def test_manual_missing_embedding_raises(self):
        msg = make_init_msg(contexts=[
            {"key": "user_A", "embedding": [1.0]},
            {"key": "user_B"},
        ])
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(msg)

    def test_manual_non_finite_embedding_raises(self):
        msg = make_init_msg(contexts=[
            {"key": "user_A", "embedding": [1.0]},
            {"key": "user_B", "embedding": [float("nan")]},
        ])
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(msg)

    def test_image_source_requires_image_paths(self):
        msg = make_init_msg(
            embeddingSource="image",
            contexts=[{"key": "user_A", "imagePath": "a.png"}, {"key": "user_B"}],
        )
        with self.assertRaises(ValueError):
            self.cs.parse_context_config(msg)

    def test_image_source_collects_paths_and_model_settings(self):
        msg = make_init_msg(
            embeddingSource="image",
            imageEmbeddingModel="ViT-B-32",
            imageEmbeddingPretrained="laion2b_s34b_b79k",
            contexts=[
                {"key": "user_A", "imagePath": "a.png"},
                {"key": "user_B", "imagePath": "b.png"},
            ],
        )
        setup = self.cs.parse_context_config(msg)
        self.assertEqual(setup.image_paths, ["a.png", "b.png"])
        self.assertEqual(setup.image_model, "ViT-B-32")
        self.assertEqual(setup.image_pretrained, "laion2b_s34b_b79k")

    def test_image_model_defaults_to_vit_g14(self):
        msg = make_init_msg(
            embeddingSource="image",
            contexts=[{"key": "user_A", "imagePath": "a.png"}],
            currentContext="user_A",
        )
        setup = self.cs.parse_context_config(msg)
        self.assertEqual(setup.image_model, "ViT-bigG-14")
        self.assertEqual(setup.image_pretrained, "laion2b_s39b_b160k")


class EmbeddingResolutionTests(unittest.TestCase):
    def setUp(self):
        self.cs = load_context_support()

    def test_l2_normalize_rows(self):
        arr = np.array([[3.0, 4.0], [0.0, 0.0]])
        out = self.cs.l2_normalize_rows(arr)
        np.testing.assert_allclose(out[0], np.array([0.6, 0.8]))
        np.testing.assert_allclose(out[1], np.array([0.0, 0.0]))

    def test_resolve_manual_normalized(self):
        setup = self.cs.parse_context_config(make_init_msg())
        emb = self.cs.resolve_embeddings(setup)
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), np.ones(2))
        self.assertTrue(setup.embeddings_resolved)

    def test_resolve_manual_unnormalized(self):
        setup = self.cs.parse_context_config(make_init_msg(normalizeEmbeddings=False))
        emb = self.cs.resolve_embeddings(setup)
        np.testing.assert_allclose(emb, np.array([[3.0, 0.0], [0.0, 4.0]]))

    def test_resolve_learned_keeps_none(self):
        setup = self.cs.parse_context_config(make_init_msg(embeddingSource="learned"))
        self.assertIsNone(self.cs.resolve_embeddings(setup))
        self.assertTrue(setup.embeddings_resolved)

    def test_resolve_image_missing_file_raises(self):
        setup = self.cs.parse_context_config(make_init_msg(
            embeddingSource="image",
            contexts=[{"key": "user_A", "imagePath": "does_not_exist.png"}],
            currentContext="user_A",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self.cs.resolve_embeddings(setup, init_root=tmp)

    def test_resolve_image_uses_backend_and_cache(self):
        calls = []

        def fake_backend(paths, model_name, pretrained):
            calls.append(list(paths))
            return np.array([[1.0, 1.0, 0.0] for _ in paths])

        self.cs.IMAGE_EMBEDDING_BACKEND = fake_backend
        with tempfile.TemporaryDirectory() as tmp:
            img_a = pathlib.Path(tmp) / "a.png"
            img_b = pathlib.Path(tmp) / "b.png"
            img_a.write_bytes(b"fake-image-a")
            img_b.write_bytes(b"fake-image-b")

            msg = make_init_msg(
                embeddingSource="image",
                contexts=[
                    {"key": "user_A", "imagePath": "a.png"},
                    {"key": "user_B", "imagePath": "b.png"},
                ],
            )
            setup = self.cs.parse_context_config(msg)
            emb = self.cs.resolve_embeddings(setup, init_root=tmp)

            self.assertEqual(len(calls), 1)
            self.assertEqual(emb.shape, (2, 3))
            # normalized by default
            np.testing.assert_allclose(np.linalg.norm(emb, axis=1), np.ones(2))

            cache_dir = pathlib.Path(tmp) / "ContextEmbeddingCache"
            self.assertTrue(cache_dir.is_dir())
            self.assertEqual(len(list(cache_dir.glob("*.npy"))), 2)

            # Second resolution hits the cache; the backend is not called again.
            setup2 = self.cs.parse_context_config(msg)
            emb2 = self.cs.resolve_embeddings(setup2, init_root=tmp)
            self.assertEqual(len(calls), 1)
            np.testing.assert_allclose(emb2, emb)

    def test_compute_image_embeddings_rejects_bad_backend_shape(self):
        self.cs.IMAGE_EMBEDDING_BACKEND = lambda paths, m, p: np.zeros((1, 2, 3))
        with tempfile.TemporaryDirectory() as tmp:
            img = pathlib.Path(tmp) / "a.png"
            img.write_bytes(b"x")
            with self.assertRaises(ValueError):
                self.cs.compute_image_embeddings([str(img)], cache_dir=None)


class WarmStartContextColumnTests(unittest.TestCase):
    def setUp(self):
        self.cs = load_context_support()
        self.setup = self.cs.parse_context_config(make_init_msg())

    def test_maps_context_column_case_insensitively(self):
        df = pd.DataFrame({"Context": ["user_A", "USER_B", "user_a"], "p0": [1, 2, 3]})
        idx = self.cs.context_indices_from_dataframe(df, self.setup, 3)
        np.testing.assert_array_equal(idx, np.array([0, 1, 0]))

    def test_missing_column_defaults_to_current_context(self):
        df = pd.DataFrame({"p0": [1, 2]})
        idx = self.cs.context_indices_from_dataframe(df, self.setup, 2)
        np.testing.assert_array_equal(idx, np.array([1, 1]))

    def test_unknown_context_key_raises(self):
        df = pd.DataFrame({"Context": ["user_Z"], "p0": [1]})
        with self.assertRaises(ValueError):
            self.cs.context_indices_from_dataframe(df, self.setup, 1)

    def test_none_setup_returns_none(self):
        df = pd.DataFrame({"p0": [1]})
        self.assertIsNone(self.cs.context_indices_from_dataframe(df, None, 1))


class TaskColumnTests(unittest.TestCase):
    def setUp(self):
        install_torch_stub()
        self.cs = load_context_support()

    def test_append_scalar_task_column(self):
        x = FakeTensor([[0.1, 0.2], [0.3, 0.4]])
        out = self.cs.append_task_column(x, 2)
        np.testing.assert_allclose(out.numpy(), np.array([[0.1, 0.2, 2.0], [0.3, 0.4, 2.0]]))

    def test_append_per_row_task_column(self):
        x = FakeTensor([[0.1], [0.3]])
        out = self.cs.append_task_column(x, [0, 1])
        np.testing.assert_allclose(out.numpy(), np.array([[0.1, 0.0], [0.3, 1.0]]))

    def test_append_row_count_mismatch_raises(self):
        x = FakeTensor([[0.1], [0.3]])
        with self.assertRaises(ValueError):
            self.cs.append_task_column(x, [0, 1, 2])

    def test_strip_task_column(self):
        x = FakeTensor([[0.1, 0.2, 1.0]])
        out = self.cs.strip_task_column(x)
        np.testing.assert_allclose(out.numpy(), np.array([[0.1, 0.2]]))


if __name__ == "__main__":
    unittest.main()
