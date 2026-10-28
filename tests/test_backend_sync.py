"""Validate generated mirrors against the canonical source in both repository layouts."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("backend_sync", Path(__file__).resolve().parents[1] / "tools/sync_backends.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class BackendSyncTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, data in {
            "optimizer.py": b"value = 1\r\n", "requirements.txt": b"numpy==2.5.4\r\n",
            "LICENSE": b"Example license\r\n", "InitData/example.csv": b"x;y\r\n1;2\r\n",
            "MetaSources/README.md": b"Source instructions\r\n",
        }.items():
            target = self.root / "Backend" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def test_both_unity_layouts_and_unreal_are_generated_from_one_source(self):
        for common in (False, True):
            with self.subTest(common=common):
                assets = self.root / ("Unity/Assets" if common else "Assets")
                assets.mkdir(parents=True, exist_ok=True)
                sync.sync(repository=self.root)
                data = assets / "StreamingAssets/BOData"
                self.assertEqual((data / "BayesianOptimization/optimizer.py").read_bytes(), b"value = 1\n")
                self.assertEqual((data / "Installation/requirements.txt").read_bytes(), b"numpy==2.5.4\n")
                self.assertEqual((self.root / "Unreal/Backend/optimizer.py").read_bytes(), b"value = 1\n")
                sync.sync(check=True, repository=self.root)

    def test_check_detects_drift_without_overwriting_the_change(self):
        sync.sync(repository=self.root)
        target = self.root / "Unreal/Backend/optimizer.py"
        target.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "differs from Backend"):
            sync.sync_engine("unreal", check=True, repository=self.root)
        self.assertEqual(target.read_text(encoding="utf-8"), "changed\n")

    def test_removed_source_removes_its_generated_file_and_unity_metadata(self):
        sync.sync(repository=self.root)
        unity_file = sync.unity_data_root(self.root) / "BayesianOptimization/optimizer.py"
        metadata = Path(str(unity_file) + ".meta")
        metadata.write_text("guid: example\n", encoding="utf-8")
        (self.root / "Backend/optimizer.py").unlink()
        sync.sync(repository=self.root)
        self.assertFalse(unity_file.exists())
        self.assertFalse(metadata.exists())
        self.assertFalse((self.root / "Unreal/Backend/optimizer.py").exists())

    def test_manifest_cannot_delete_an_unmanaged_or_external_file(self):
        sync.sync(repository=self.root)
        victim = self.root / "keep.txt"
        victim.write_text("keep\n", encoding="utf-8")
        for name in ("../keep.txt", "other/keep.txt", "C:/keep.txt"):
            with self.subTest(name=name):
                sync.manifest_path("unreal", self.root).write_text(json.dumps({name: "digest"}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    sync.sync_engine("unreal", repository=self.root)
                self.assertEqual(victim.read_text(encoding="utf-8"), "keep\n")

    def test_metadata_is_preserved_and_line_endings_do_not_change_identity(self):
        sync.sync(repository=self.root)
        target = sync.unity_data_root(self.root) / "BayesianOptimization/optimizer.py"
        metadata = Path(str(target) + ".meta")
        metadata.write_bytes(b"guid: keep-this-guid\n")
        target.write_bytes(b"value = 1\r\n")
        sync.sync(check=True, repository=self.root)
        sync.sync(repository=self.root)
        self.assertEqual(metadata.read_bytes(), b"guid: keep-this-guid\n")

    def test_unity_metadata_is_created_for_new_files_and_folders_and_checked(self):
        (self.root / "Backend/package").mkdir()
        (self.root / "Backend/package/helper.py").write_bytes(b"value = 2\n")
        sync.sync(repository=self.root)
        data = sync.unity_data_root(self.root)
        folder = data / "BayesianOptimization/package.meta"
        helper = data / "BayesianOptimization/package/helper.py.meta"
        self.assertIn("folderAsset: yes", folder.read_text(encoding="utf-8"))
        guid = helper.read_text(encoding="utf-8")
        self.assertNotIn("folderAsset", guid)
        helper.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing Unity .meta file"):
            sync.sync_engine("unity", check=True, repository=self.root)
        sync.sync(repository=self.root)
        self.assertEqual(helper.read_text(encoding="utf-8"), guid)

    def test_a_folder_emptied_by_a_removal_goes_with_its_unity_metadata(self):
        (self.root / "Backend/package").mkdir()
        (self.root / "Backend/package/helper.py").write_bytes(b"value = 2\n")
        sync.sync(repository=self.root)
        folder = sync.unity_data_root(self.root) / "BayesianOptimization/package"
        self.assertTrue(Path(str(folder) + ".meta").is_file())
        (self.root / "Backend/package/helper.py").unlink()
        (self.root / "Backend/package").rmdir()
        sync.sync(repository=self.root)
        self.assertFalse(folder.exists())
        self.assertFalse(Path(str(folder) + ".meta").exists())
        self.assertFalse((self.root / "Unreal/Backend/package").exists())
        sync.sync(check=True, repository=self.root)
        orphan = sync.unity_data_root(self.root) / "BayesianOptimization/stale.py.meta"
        orphan.write_text("guid: stale\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "without its asset"):
            sync.sync_engine("unity", check=True, repository=self.root)

    def test_runtime_embedding_cache_is_not_distributed(self):
        cache = self.root / "Backend/InitData/ContextEmbeddingCache/vector.npy"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"cached")
        manifests = sync.sync(repository=self.root)
        for engine, manifest in manifests.items():
            with self.subTest(engine=engine):
                self.assertFalse(any("ContextEmbeddingCache" in name for name in manifest))
        written = self.root / "Unreal/InitData/ContextEmbeddingCache/other.npy"
        written.parent.mkdir(parents=True)
        written.write_bytes(b"cached")
        sync.sync(check=True, repository=self.root)


if __name__ == "__main__":
    unittest.main()
