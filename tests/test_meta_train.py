"""Tier-1 tests for meta_train.py's command line: naming, publishing and the population manifest.

Runs in the numpy+pandas-only CI job: the torch/openbo stack is replaced by a fake whose
loader replays the trajectory written by write_artifact, and the GP fit by fixed
hyperparameters. What is verified is the tool's contract around the fit -- which artifacts
end up in --out under which names, what is refused, and what the runtime can rely on.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
import uuid
from unittest import mock

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BO_DIR = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization"
TRAIN_PATH = BO_DIR / "meta_train.py"


def load_meta_train():
    name = f"meta_train_cli_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FRAME = {
    "parameters": [{"key": "p0", "low": 0.0, "high": 1.0}, {"key": "p1", "low": 2.0, "high": 6.0}],
    "objectives": [{"key": "o0", "low": 0.0, "high": 10.0, "minimize": 1},
                   {"key": "o1", "low": 0.0, "high": 10.0, "minimize": 0}],
}

ENTRY = {"kernel_type": "matern52", "lengthscale": [0.3, 0.3], "variance": 1.0, "noise": 1e-4,
         "mean_constant": 0.0, "standardize_targets": True, "optimize_noise": True}


def fake_stack():
    def loader(directory, expected_m=None, expected_d=None):
        sources = []
        for path in sorted((pathlib.Path(directory) / "trajectories").glob("*.json")):
            y = np.asarray(json.loads(path.read_text(encoding="utf-8"))["y_values"], dtype=np.float64)

            class _Source:
                name = path.stem

                def posterior_mean(self, x, y=y):
                    return y

            sources.append(_Source())
        return sources

    return (None,) * 7 + (lambda y: np.asarray(y, dtype=np.float64), loader)


def write_run(root, participant, condition="main", run="run", shift=0.0, rows=4):
    """A completed run's ObservationsPerEvaluation.csv under root/<participant>/<condition>/<run>."""
    run_dir = pathlib.Path(root) / participant / condition / run
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "UserID": [participant] * rows, "Iteration": list(range(1, rows + 1)),
        "o0": [2.0 + shift, 4.0, 1.0, 3.0][:rows], "o1": [7.0, 6.0 + shift, 8.0, 9.0][:rows],
        "p0": [0.1, 0.5, 0.9, 0.3][:rows], "p1": [2.5, 3.0 + shift, 5.5, 4.0][:rows],
    })
    df.to_csv(run_dir / "ObservationsPerEvaluation.csv", sep=";", index=False)
    return run_dir


class _CliMixin:
    def setUp(self):
        self.mt = load_meta_train()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.out = self.tmp / "MetaSources"
        self.frame_path = self.tmp / "frame.json"
        self.frame_path.write_text(json.dumps(FRAME), encoding="utf-8")
        self.logs = self.tmp / "LogData"

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *args, stamps=True, frame=None):
        argv = ["--frame", str(frame or self.frame_path), "--out", str(self.out)]
        if stamps:
            argv += ["--source-type", "synthetic", "--y-calibration", "generated"]
        argv += [str(a) for a in args]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(self.mt, "_import_stack", fake_stack), \
                mock.patch.object(self.mt, "fit_per_objective_hyperparameters",
                                  lambda x, y, stack: [dict(ENTRY), dict(ENTRY)]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                self.mt.main(argv)
                code = 0
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def artifacts(self):
        return sorted(p.stem for p in (self.out / "gp_states").glob("*.json"))


class NamingTests(_CliMixin, unittest.TestCase):
    def test_name_is_the_path_tail_not_the_command_line_position(self):
        run = write_run(self.logs, "p01")
        self.assertEqual(self.mt.derive_name(str(run)), "p01_main_run")
        csv_path = run / "ObservationsPerEvaluation.csv"
        self.assertEqual(self.mt.derive_name(str(csv_path)), "p01_main_run")  # participant kept

    def test_rebuilding_with_one_more_participant_adds_exactly_one_source(self):
        """Reproduction: indices in the names made a rebuild with one more (or reordered)
        run publish renamed copies; p01/p02 then counted twice in every later run."""
        p01, p02, p03 = (write_run(self.logs, p) for p in ("p01", "p02", "p03"))
        self.assertEqual(self.run_cli(p01, p02)[0], 0)
        code, out, _ = self.run_cli(p03, p01, p02)
        self.assertEqual(code, 0)
        self.assertEqual(self.artifacts(), ["p01_main_run", "p02_main_run", "p03_main_run"])
        self.assertEqual(out.count("[KEEP]"), 2)

    def test_colliding_tails_get_a_path_hash(self):
        a = write_run(self.tmp / "studyA", "p01")
        b = write_run(self.tmp / "studyB", "p01", shift=0.5)
        code, _, _ = self.run_cli(a, b)
        self.assertEqual(code, 0)
        expected = sorted(f"p01_main_run-{self.mt.path_hash(str(r))}" for r in (a, b))
        self.assertEqual(self.artifacts(), expected)
        # Rebuilding one of them alone finds its existing name again.
        code, out, _ = self.run_cli(a)
        self.assertIn("[KEEP]", out)
        self.assertEqual(self.artifacts(), expected)

    def test_existing_artifact_keeps_its_plain_name_on_a_colliding_rebuild(self):
        a = write_run(self.tmp / "studyA", "p01")
        b = write_run(self.tmp / "studyB", "p01", shift=0.5)
        self.run_cli(a)
        self.run_cli(a, b)
        self.assertEqual(self.artifacts(),
                         sorted(["p01_main_run", f"p01_main_run-{self.mt.path_hash(str(b))}"]))

    def test_older_index_prefixed_population_is_extended_not_duplicated(self):
        """A folder built by meta_train.py 1.8.0 names sources '00_p01_main_run', ...; a
        rebuild with one more participant published every earlier run a second time under
        its new name (reproduced: 2 + 3 = 5 sources, each earlier participant counted twice)."""
        p01, p02, p03 = (write_run(self.logs, p, shift=s) for p, s in (("p01", 0.0), ("p02", 0.5), ("p03", 1.0)))
        self.run_cli(p01, p02, "--names", "00_p01_main_run,01_p02_main_run")  # the 1.8.0 names
        code, out, _ = self.run_cli(p01, p02, p03)
        self.assertEqual(code, 0)
        self.assertEqual(self.artifacts(), ["00_p01_main_run", "01_p02_main_run", "p03_main_run"])
        self.assertIn("[KEEP] 00_p01_main_run", out)
        self.assertNotIn("[WARN]", out)
        # --force rebuilds them under the names they have.
        self.assertEqual(self.run_cli(p01, p02, p03, "--force")[0], 0)
        self.assertEqual(self.artifacts(), ["00_p01_main_run", "01_p02_main_run", "p03_main_run"])

    def test_population_built_from_another_path_is_extended_not_duplicated(self):
        """The provenance records the absolute run path. After the project folder moved (or
        on a clone on another machine) every earlier run looked new: a rebuild with one more
        participant added a hashed second copy of each (reproduced: 2 + 3 = 5 sources)."""
        old = self.tmp / "ProjectA" / "LogData"
        p01, p02 = write_run(old, "p01"), write_run(old, "p02", shift=0.5)
        self.run_cli(p01, p02)
        moved = self.tmp / "ProjectB" / "LogData"
        os.makedirs(moved.parent)
        os.rename(old, moved)
        runs = [moved / p / "main" / "run" for p in ("p01", "p02")] + [write_run(moved, "p03", shift=1.0)]
        code, out, _ = self.run_cli(*runs)
        self.assertEqual(code, 0)
        self.assertEqual(self.artifacts(), ["p01_main_run", "p02_main_run", "p03_main_run"])
        self.assertEqual(out.count("[KEEP]"), 2)

    def test_same_run_under_another_explicit_name_is_warned(self):
        run = write_run(self.logs, "p01")
        self.run_cli(run, "--name", "00_p01_main_run")
        code, out, _ = self.run_cli(run, "--name", "p01")
        self.assertEqual(code, 0)
        self.assertIn("[WARN]", out)
        self.assertIn("'00_p01_main_run'", out)

    def test_another_spelling_of_the_same_file_is_not_a_second_copy(self):
        """On a case-insensitive file system 'P01_main_run' and 'p01_main_run' are one file;
        the warning told the user to delete the artifact that was just kept."""
        run = write_run(self.logs, "p01")
        self.run_cli(run)
        gp_dir = self.out / "gp_states"
        os.link(gp_dir / "p01_main_run.json", gp_dir / "alias.json")  # the same file, two names
        existing = self.mt.existing_artifacts(str(self.out))
        self.assertEqual(self.mt.same_run_other_names(str(run), "p01_main_run", existing, str(self.out)), [])
        self.assertEqual(self.mt.same_run_other_names(str(run), "p01_main_run", existing), ["alias"])

    def test_same_run_recognizes_another_spelling_on_a_case_insensitive_file_system(self):
        run = write_run(self.logs, "p01")
        other_spelling = self.logs / "P01" / "main" / "run"
        if not other_spelling.is_dir():
            self.skipTest("case-sensitive file system")
        self.assertTrue(self.mt.same_run(str(run), str(other_spelling)))

    def test_file_systems_without_file_identities_compare_paths(self):
        """Some network file systems report inode 0 for every file: os.path.samefile then
        calls any two runs the same, and the second run would take the first one's name."""
        p01, p02 = write_run(self.logs, "p01"), write_run(self.logs, "p02")
        real_stat = os.stat

        def stat_without_inode(path, *args, **kwargs):
            st = real_stat(path, *args, **kwargs)
            return os.stat_result((st.st_mode, 0, 0, *tuple(st)[3:]))

        with mock.patch.object(self.mt.os, "stat", stat_without_inode):
            self.assertFalse(self.mt.same_run(str(p01), str(p02)))
            self.assertTrue(self.mt.same_run(str(p01), str(p01 / "ObservationsPerEvaluation.csv")))

    def test_run_listed_twice_is_refused(self):
        run = write_run(self.logs, "p01")
        code, _, err = self.run_cli(run, run / "ObservationsPerEvaluation.csv")
        self.assertEqual(code, 2)
        self.assertIn("more than once", err)


class CliTests(_CliMixin, unittest.TestCase):
    def test_names_before_the_runs_is_refused_with_a_fix(self):
        p01, p02 = write_run(self.logs, "p01"), write_run(self.logs, "p02")
        code, _, err = self.run_cli("--names", "a", "b", p01, p02)
        self.assertEqual(code, 2)
        self.assertIn("put --names after the run paths", err)
        self.assertFalse(self.out.exists())

    def test_names_after_the_runs_still_work(self):
        p01, p02 = write_run(self.logs, "p01"), write_run(self.logs, "p02")
        self.assertEqual(self.run_cli(p01, p02, "--names", "a", "b")[0], 0)
        self.assertEqual(self.artifacts(), ["a", "b"])

    def test_comma_list_and_repeated_name(self):
        p01, p02 = write_run(self.logs, "p01"), write_run(self.logs, "p02")
        self.assertEqual(self.run_cli("--names", "a,b", p01, p02)[0], 0)
        self.assertEqual(self.run_cli("--name", "c", "--name", "d", p01, p02, "--force")[0], 0)
        self.assertEqual(self.artifacts(), ["a", "b", "c", "d"])

    def test_name_may_sit_next_to_its_run(self):
        """'--name p01 run1 --name p02 run2' failed with 'unrecognized arguments: run2'."""
        p01, p02 = write_run(self.logs, "p01"), write_run(self.logs, "p02", shift=0.5)
        self.assertEqual(self.run_cli("--name", "a", p01, "--name", "b", p02)[0], 0)
        self.assertEqual(self.artifacts(), ["a", "b"])
        provenance = json.loads((self.out / "gp_states" / "b.json").read_text(encoding="utf-8"))["provenance"]
        self.assertTrue(self.mt.same_run(provenance["run_path"], str(p02)))

    def test_name_count_must_match(self):
        p01, p02 = write_run(self.logs, "p01"), write_run(self.logs, "p02")
        code, _, err = self.run_cli(p01, p02, "--names", "a")
        self.assertEqual(code, 2)
        self.assertIn("1 name(s) for 2 run(s)", err)

    def test_frame_objective_without_minimize_is_refused(self):
        """The runtime requires the flag; a silent default of 0 turned a forgotten minimize
        objective into an inverted source whose frame stamp still matched."""
        frame = json.loads(json.dumps(FRAME))
        del frame["objectives"][0]["minimize"]
        path = self.tmp / "frame_no_min.json"
        path.write_text(json.dumps(frame), encoding="utf-8")
        code, _, _ = self.run_cli(write_run(self.logs, "p01"), frame=path)
        self.assertIn("minimize", str(code))
        self.assertFalse(self.out.exists())

    def test_frame_minimize_must_be_0_or_1(self):
        frame = json.loads(json.dumps(FRAME))
        frame["objectives"][1]["minimize"] = "yes"
        path = self.tmp / "frame_bad_min.json"
        path.write_text(json.dumps(frame), encoding="utf-8")
        code, _, _ = self.run_cli(write_run(self.logs, "p01"), frame=path)
        self.assertIn("minimize", str(code))

    def test_dry_run_writes_nothing_and_needs_no_stack(self):
        p01 = write_run(self.logs, "p01")
        argv = ["--frame", str(self.frame_path), "--out", str(self.out), "--source-type", "human",
                "--y-calibration", "measured", "--dry-run", str(p01)]
        out = io.StringIO()
        with mock.patch.object(self.mt, "_import_stack", side_effect=AssertionError("stack imported")), \
                contextlib.redirect_stdout(out):
            self.mt.main(argv)
        self.assertIn("[DRY]  p01_main_run", out.getvalue())
        self.assertFalse(self.out.exists())

    def test_existing_artifact_is_kept_unless_forced(self):
        run = write_run(self.logs, "p01")
        self.run_cli(run)
        gp = self.out / "gp_states" / "p01_main_run.json"
        before = gp.read_bytes()
        write_run(self.logs, "p01", shift=1.0)  # the run's data changed since
        code, out, _ = self.run_cli(run)
        self.assertNotEqual(code, 0)  # nothing written
        self.assertIn("[KEEP] p01_main_run", out)
        self.assertEqual(gp.read_bytes(), before)
        self.assertEqual(self.run_cli(run, "--force")[0], 0)
        self.assertNotEqual(gp.read_bytes(), before)


class PublishingTests(_CliMixin, unittest.TestCase):
    def test_gp_state_carries_the_trajectory_hash(self):
        self.run_cli(write_run(self.logs, "p01"))
        payload = json.loads((self.out / "gp_states" / "p01_main_run.json").read_text(encoding="utf-8"))
        traj = (self.out / "trajectories" / "p01_main_run.json").read_bytes()
        self.assertEqual(payload["provenance"]["trajectory_sha256"], hashlib.sha256(traj).hexdigest())

    def test_replace_is_retried_while_the_target_is_held(self):
        real_replace = os.replace
        failures = {"left": 2}

        def flaky(src, dst):
            if failures["left"] and dst.endswith("p01_main_run.json") and "gp_states" in dst:
                failures["left"] -= 1
                raise PermissionError(13, "The process cannot access the file", dst)
            return real_replace(src, dst)

        with mock.patch.object(self.mt.os, "replace", flaky), mock.patch.object(self.mt.time, "sleep"):
            code, _, _ = self.run_cli(write_run(self.logs, "p01"))
        self.assertEqual(code, 0)
        self.assertEqual(failures["left"], 0)
        self.assertEqual(self.artifacts(), ["p01_main_run"])

    def test_failed_gp_state_replace_leaves_the_previous_pair(self):
        """Trajectory replaced, gp_state replace failing (held open on Windows): the new
        trajectory sat next to the OLD gp_state, which the frame check accepted."""
        run = write_run(self.logs, "p01")
        self.run_cli(run)
        traj = self.out / "trajectories" / "p01_main_run.json"
        before = traj.read_bytes()
        write_run(self.logs, "p01", shift=1.0)
        real_replace = os.replace

        def held(src, dst):
            if "gp_states" in dst and dst.endswith("p01_main_run.json"):
                raise PermissionError(13, "The process cannot access the file", dst)
            return real_replace(src, dst)

        with mock.patch.object(self.mt.os, "replace", held), \
                mock.patch.object(self.mt, "REPLACE_RETRY_SEC", 0.0):
            code, out, _ = self.run_cli(run, "--force")
        self.assertIn("[SKIP]", out)
        self.assertEqual(traj.read_bytes(), before)
        payload = json.loads((self.out / "gp_states" / "p01_main_run.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["provenance"]["trajectory_sha256"], hashlib.sha256(before).hexdigest())

    def test_json_is_fsynced_before_publishing(self):
        with mock.patch.object(self.mt.os, "fsync") as fsync:
            self.mt._write_json(str(self.tmp / "x.json"), {"a": 1})
        self.assertEqual(fsync.call_count, 1)

    def test_reference_point_is_the_runtime_one(self):
        import bo_normalize
        self.assertEqual(self.mt.REF_POINT_VALUE, bo_normalize.HYPERVOLUME_REFERENCE_VALUE)
        # A front rated worst (-1) in one objective still dominates the -1.1 reference.
        self.mt.check_dominates_reference(np.array([[-1.0, 0.4], [-1.0, -0.2]]))
        with self.assertRaises(ValueError):
            self.mt.check_dominates_reference(np.array([[-1.2, 0.4]]))


class ManifestTests(_CliMixin, unittest.TestCase):
    def manifest(self):
        return json.loads((self.out / self.mt.MANIFEST_FILENAME).read_text(encoding="utf-8"))

    def test_build_writes_the_population_manifest(self):
        p01, p02 = write_run(self.logs, "p01"), write_run(self.logs, "p02", shift=0.5)
        self.run_cli(p01, p02)
        manifest = self.manifest()
        frame = self.mt.load_frame(str(self.frame_path))[0]
        self.assertEqual(manifest["frame_digest"], self.mt.meta_fingerprint.frame_digest(frame))
        names = [s["name"] for s in manifest["sources"]]
        self.assertEqual(names, ["p01_main_run", "p02_main_run"])
        for source in manifest["sources"]:
            traj = (self.out / "trajectories" / f"{source['name']}.json").read_bytes()
            gp = (self.out / "gp_states" / f"{source['name']}.json").read_bytes()
            self.assertEqual(source["trajectory_sha256"], hashlib.sha256(traj).hexdigest())
            self.assertEqual(source["gp_state_sha256"], hashlib.sha256(gp).hexdigest())

    def test_manifest_covers_the_whole_folder_and_skips_foreign_frames(self):
        self.run_cli(write_run(self.logs, "p01"))
        self.run_cli(write_run(self.logs, "p02", shift=0.5))
        # An artifact built for another frame is not part of the population.
        foreign = json.loads((self.out / "gp_states" / "p02_main_run.json").read_text(encoding="utf-8"))
        foreign["frame"]["objs"][0][3] = 0
        (self.out / "gp_states" / "zz.json").write_text(json.dumps(foreign), encoding="utf-8")
        (self.out / "trajectories" / "zz.json").write_bytes(
            (self.out / "trajectories" / "p02_main_run.json").read_bytes())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.mt.write_manifest(str(self.out), self.mt.load_frame(str(self.frame_path))[0])
        self.assertEqual([s["name"] for s in self.manifest()["sources"]], ["p01_main_run", "p02_main_run"])
        self.assertIn("'zz' was built for another frame", out.getvalue())

    def test_line_ending_conversion_is_not_a_change(self):
        """git (core.autocrlf) or a sync tool converting line endings on checkout changed
        every gp_state hash, so the runtime refused an unchanged population on every machine
        except the one that built it."""
        self.run_cli(write_run(self.logs, "p01"))
        gp = self.out / "gp_states" / "p01_main_run.json"
        self.assertIn(b"\n", gp.read_bytes())
        self.assertNotIn(b"\r\n", gp.read_bytes())  # written with LF on every OS
        before = self.manifest()["sources"]
        gp.write_bytes(gp.read_bytes().replace(b"\n", b"\r\n"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.mt.write_manifest(str(self.out), self.mt.load_frame(str(self.frame_path))[0])
        self.assertEqual(self.manifest()["sources"], before)

    def test_manifest_only_needs_no_runs_or_stamps(self):
        self.run_cli(write_run(self.logs, "p01"))
        (self.out / self.mt.MANIFEST_FILENAME).unlink()
        code, _, _ = self.run_cli("--manifest-only", stamps=False)
        self.assertEqual(code, 0)
        self.assertEqual([s["name"] for s in self.manifest()["sources"]], ["p01_main_run"])


if __name__ == "__main__":
    unittest.main()
