"""Distribution guards prevent unrelated assets and installers entering releases."""

import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("build_packages", _ROOT / "tools/build_packages.py")
packaging = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(packaging)


class PackagingTests(unittest.TestCase):
    def test_archive_cannot_include_its_own_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            output = source / "release.zip"
            with self.assertRaises(ValueError):
                packaging.archive_folder(source, output)
            self.assertFalse(output.exists())

    def test_release_output_cannot_be_inside_an_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for engine in ("Unity", "Unreal"):
                with self.subTest(engine=engine), self.assertRaises(ValueError):
                    packaging.build(root, root / engine / "release", "2.0.0-rc.1", None)

    def test_archive_excludes_generated_caches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "Unreal"
            source.mkdir()
            (source / "README.md").write_bytes(b"Example\n")
            (source / "__pycache__").mkdir()
            (source / "__pycache__/cached.pyc").write_bytes(b"cache")
            (source / "LogData").mkdir()
            (source / "LogData/private.csv").write_bytes(b"private measurements")
            (source / "LogData.meta").write_bytes(b"folder metadata")
            output = root / "release.zip"
            packaging.archive_folder(source, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.namelist(), ["Unreal/README.md"])
                self.assertEqual(archive.read("Unreal/README.md"), b"Example\n")

    def test_a_checkout_ships_only_tracked_files(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "Unreal"
            (source / "runs/P01").mkdir(parents=True)
            (source / "README.md").write_bytes(b"Example\n")
            git = ["git", "-c", "user.name=iBO", "-c", "user.email=ibo@example.invalid"]
            subprocess.run([*git, "init", "-q"], cwd=source, check=True)
            subprocess.run([*git, "add", "README.md"], cwd=source, check=True)
            (source / "runs/P01/ObservationsPerEvaluation.csv").write_bytes(b"private measurements")
            (source / "Unreal.sln").write_bytes(b"local project file")
            self.assertEqual(packaging.distribution_files(source), [source / "README.md"])

    def test_an_empty_package_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "Unreal"
            (source / "LogData").mkdir(parents=True)
            (source / "LogData/private.csv").write_bytes(b"private measurements")
            with self.assertRaisesRegex(ValueError, "No files to package"):
                packaging.archive_folder(source, Path(folder) / "release.zip")
            self.assertFalse((Path(folder) / "release.zip").exists())

    def test_questionnaire_addon_is_left_out_of_the_unity_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "Unity"
            addon = source / "Assets/iBO-QuestionnaireToolkit"
            (addon / "QuestionnaireToolkit/Plugins").mkdir(parents=True)
            (addon / "QuestionnaireToolkit/Plugins/Ookii.Dialogs.dll").write_bytes(b"native")
            (source / "Assets/iBO-QuestionnaireToolkit.meta").write_bytes(b"folder metadata")
            (source / "Assets/README.md").write_bytes(b"Example\n")
            output = root / "release.zip"
            packaging.archive_folder(source, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.namelist(), ["Unity/Assets/README.md"])

    def test_installers_and_commercial_toolkit_are_refused(self):
        for name in ("python.exe", "QuestionnaireToolkit/source.cs", "TextMesh Pro/example.cs"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"unexpected")
                with self.assertRaises(ValueError):
                    packaging.distribution_files(root)

    def test_asset_package_paths_are_checked(self):
        for path, expected in (("Assets/iBO/Scripts/Optimizer.cs", True),
                               ("Assets/QuestionnaireToolkit/source.cs", False),
                               ("Assets/StreamingAssets/python.exe", False)):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as folder:
                package = Path(folder) / "test.unitypackage"
                payload = path.encode("utf-8")
                with tarfile.open(package, "w:gz") as archive:
                    item = tarfile.TarInfo("123/pathname")
                    item.size = len(payload)
                    archive.addfile(item, io.BytesIO(payload))
                if expected:
                    self.assertEqual(packaging.inspect_unitypackage(package), 1)
                else:
                    with self.assertRaises(ValueError):
                        packaging.inspect_unitypackage(package)

    def test_existing_archives_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "Unreal"
            source.mkdir()
            output = Path(folder) / "release.zip"
            output.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                packaging.archive_folder(source, output)
            self.assertEqual(output.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
