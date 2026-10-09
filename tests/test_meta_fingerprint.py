"""Tier-1 tests for meta_fingerprint.py — Meta-TAF source/target frame compatibility.

Pure stdlib: runs in the numpy+pandas-only CI job.

The decisive case is `test_flipped_minimize_flag_is_detected`. Two studies with the same d and
M but an opposite minimize flag produce artifacts that are indistinguishable by shape, because
everything is stored already normalized. Loading such a source transfers an exactly inverted
response surface. Only the fingerprint catches it.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
import uuid

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FINGERPRINT_PATH = (
    REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization/meta_fingerprint.py"
)


def load_fingerprint():
    name = f"meta_fingerprint_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, FINGERPRINT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARAM_NAMES = ["speed", "gap"]
PARAMS_INFO = [(0.0, 10.0), (1.0, 5.0)]
OBJ_NAMES = ["comfort", "duration"]
OBJS_INFO = [(0.0, 100.0, 0), (0.0, 60.0, 1)]


class CanonicalFrameTests(unittest.TestCase):
    def setUp(self):
        self.fp = load_fingerprint()

    def frame(self, **over):
        return self.fp.canonical_frame(
            over.get("parameter_names", PARAM_NAMES),
            over.get("parameters_info", PARAMS_INFO),
            over.get("objective_names", OBJ_NAMES),
            over.get("objectives_info", OBJS_INFO),
        )

    def test_frame_records_shape_names_bounds_and_minflags(self):
        f = self.frame()
        self.assertEqual(f["d"], 2)
        self.assertEqual(f["M"], 2)
        self.assertEqual(f["v"], self.fp.FRAME_SCHEMA_VERSION)
        self.assertEqual([p[0] for p in f["params"]], PARAM_NAMES)
        self.assertEqual([o[0] for o in f["objs"]], OBJ_NAMES)
        self.assertEqual([o[3] for o in f["objs"]], [0, 1])

    def test_identical_frames_are_compatible(self):
        self.assertTrue(self.fp.frames_compatible(self.frame(), self.frame()))
        self.assertEqual(self.fp.frame_differences(self.frame(), self.frame()), [])

    def test_int_vs_float_bounds_do_not_spuriously_differ(self):
        a = self.frame(parameters_info=[(0, 10), (1, 5)])
        b = self.frame(parameters_info=[(0.0, 10.0), (1.0, 5.0)])
        self.assertTrue(self.fp.frames_compatible(a, b))
        self.assertEqual(self.fp.frame_digest(a), self.fp.frame_digest(b))

    def test_flipped_minimize_flag_is_detected(self):
        flipped = list(OBJS_INFO)
        flipped[1] = (0.0, 60.0, 0)  # was minimized
        diffs = self.fp.frame_differences(self.frame(), self.frame(objectives_info=flipped))
        self.assertTrue(diffs)
        joined = " ".join(diffs)
        self.assertIn("minimize flag", joined)
        self.assertIn("invert", joined)

    def test_different_objective_bounds_are_detected(self):
        other = list(OBJS_INFO)
        other[0] = (0.0, 50.0, 0)
        diffs = self.fp.frame_differences(self.frame(), self.frame(objectives_info=other))
        self.assertTrue(any("bounds" in d for d in diffs))

    def test_different_parameter_bounds_are_detected(self):
        other = list(PARAMS_INFO)
        other[0] = (0.0, 20.0)
        diffs = self.fp.frame_differences(self.frame(), self.frame(parameters_info=other))
        self.assertTrue(any("bounds" in d for d in diffs))

    def test_renamed_parameter_is_detected(self):
        diffs = self.fp.frame_differences(
            self.frame(), self.frame(parameter_names=["velocity", "gap"])
        )
        self.assertTrue(any("name" in d for d in diffs))

    def test_reordered_objectives_are_detected(self):
        diffs = self.fp.frame_differences(
            self.frame(),
            self.frame(
                objective_names=list(reversed(OBJ_NAMES)),
                objectives_info=list(reversed(OBJS_INFO)),
            ),
        )
        self.assertTrue(diffs)

    def test_dimension_mismatch_is_detected_and_short_circuits(self):
        diffs = self.fp.frame_differences(
            self.frame(),
            self.frame(parameter_names=["speed"], parameters_info=[(0.0, 10.0)]),
        )
        self.assertTrue(any("parameter count" in d for d in diffs))

    def test_objective_count_mismatch_is_detected(self):
        diffs = self.fp.frame_differences(
            self.frame(),
            self.frame(objective_names=["comfort"], objectives_info=[(0.0, 100.0, 0)]),
        )
        self.assertTrue(any("objective count" in d for d in diffs))

    def test_schema_version_mismatch_short_circuits(self):
        stale = self.frame()
        stale["v"] = self.fp.FRAME_SCHEMA_VERSION + 1
        diffs = self.fp.frame_differences(self.frame(), stale)
        self.assertEqual(len(diffs), 1)
        self.assertIn("schema version", diffs[0])

    def test_missing_frame_is_reported_not_crashed(self):
        for bad in (None, "nope", 42):
            with self.subTest(bad=bad):
                diffs = self.fp.frame_differences(self.frame(), bad)
                self.assertTrue(diffs)


class TruncatedOrMalformedFrameTests(unittest.TestCase):
    """zip() stopped at the shorter list, so a frame whose params/objs were missing or cut
    short compared as compatible without a single entry being checked."""

    def setUp(self):
        self.fp = load_fingerprint()
        self.expected = self.fp.canonical_frame(PARAM_NAMES, PARAMS_INFO, OBJ_NAMES, OBJS_INFO)

    def actual(self, **changes):
        frame = json.loads(json.dumps(self.expected))
        frame.update(changes)
        return frame

    def assert_incompatible(self, actual, snippet):
        diffs = self.fp.frame_differences(self.expected, actual)
        self.assertTrue(diffs)
        self.assertFalse(self.fp.frames_compatible(self.expected, actual))
        self.assertIn(snippet, " ".join(diffs))

    def test_missing_params_or_objs_are_incompatible(self):
        for key in ("params", "objs"):
            with self.subTest(key=key):
                frame = self.actual()
                del frame[key]
                self.assert_incompatible(frame, f"'{key}' must list 2")

    def test_shortened_lists_are_incompatible(self):
        self.assert_incompatible(self.actual(params=self.expected["params"][:1]), "'params' must list 2")
        self.assert_incompatible(self.actual(objs=[]), "'objs' must list 2")
        self.assert_incompatible(self.actual(objs="comfort,duration"), "'objs' must list 2")

    def test_malformed_entries_are_incompatible(self):
        cases = [
            ("params", [["speed", "0", "10"], ["gap", "1"]], "parameter 1 is not a list of 3"),
            ("params", [["speed", "zero", "10"], ["gap", "1", "5"]], "not a finite number"),
            ("params", [["speed", "0", "nan"], ["gap", "1", "5"]], "not a finite number"),
            ("params", [[7, "0", "10"], ["gap", "1", "5"]], "name is not a string"),
            ("objs", [["comfort", "0", "100", 0], ["duration", "0", "60", 2]], "minimize flag 2"),
            ("objs", [["comfort", "0", "100", 0], ["duration", "0", "60"]], "objective 1 is not a list of 4"),
            ("objs", [None, ["duration", "0", "60", 1]], "objective 0 is not a list of 4"),
        ]
        for key, value, snippet in cases:
            with self.subTest(value=value):
                self.assert_incompatible(self.actual(**{key: value}), snippet)

    def test_negative_zero_bound_matches_zero(self):
        """format(-0.0, '.12g') is '-0': a bound sent as -0 never matched a frame with 0."""
        neg = self.fp.canonical_frame(PARAM_NAMES, [(-0.0, 10.0), (1.0, 5.0)], OBJ_NAMES,
                                      [(-0.0, 100.0, 0), (0.0, 60.0, 1)])
        self.assertEqual(neg["params"][0][1], "0")
        self.assertTrue(self.fp.frames_compatible(self.expected, neg))
        self.assertEqual(self.fp.frame_digest(neg), self.fp.frame_digest(self.expected))

    def test_artifact_stored_with_negative_zero_still_matches(self):
        stored = self.actual()
        stored["params"][0][1] = "-0"  # written before the normalization
        self.assertEqual(self.fp.frame_differences(self.expected, stored), [])

    def test_numeric_bounds_in_a_hand_written_frame_compare_by_value(self):
        stored = self.actual()
        stored["params"][1][1:] = [1, 5.0]
        self.assertEqual(self.fp.frame_differences(self.expected, stored), [])


class DigestTests(unittest.TestCase):
    def setUp(self):
        self.fp = load_fingerprint()

    def frame(self, **over):
        return self.fp.canonical_frame(
            over.get("parameter_names", PARAM_NAMES),
            over.get("parameters_info", PARAMS_INFO),
            over.get("objective_names", OBJ_NAMES),
            over.get("objectives_info", OBJS_INFO),
        )

    def test_digest_is_stable_and_short(self):
        self.assertEqual(self.fp.frame_digest(self.frame()), self.fp.frame_digest(self.frame()))
        self.assertEqual(len(self.fp.frame_digest(self.frame())), 16)

    def test_digest_changes_when_minflag_flips(self):
        flipped = list(OBJS_INFO)
        flipped[1] = (0.0, 60.0, 0)
        self.assertNotEqual(
            self.fp.frame_digest(self.frame()),
            self.fp.frame_digest(self.frame(objectives_info=flipped)),
        )


class ArtifactHashTests(unittest.TestCase):
    """The file identity in population.json and the trajectory stamp."""

    def setUp(self):
        self.fp = load_fingerprint()

    def _hash(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.json")
            with open(path, "wb") as f:
                f.write(data)
            return self.fp.artifact_sha256(path)

    def test_line_endings_do_not_change_the_identity(self):
        """A git checkout with core.autocrlf (or the reverse on commit) converts them."""
        lf = json.dumps({"gp_state": {"noise": 0.1}, "frame": None}, indent=1).encode("utf-8")
        self.assertEqual(self._hash(lf), hashlib.sha256(lf).hexdigest())  # LF: the plain hash
        self.assertEqual(self._hash(lf.replace(b"\n", b"\r\n")), self._hash(lf))

    def test_content_changes_change_the_identity(self):
        self.assertNotEqual(self._hash(b'{"noise": 0.1}'), self._hash(b'{"noise": 0.2}'))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.fp = load_fingerprint()

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            self.fp.canonical_frame(["a", "b"], [(0.0, 1.0)], OBJ_NAMES, OBJS_INFO)
        with self.assertRaises(ValueError):
            self.fp.canonical_frame(PARAM_NAMES, PARAMS_INFO, ["a", "b"], [(0.0, 1.0, 0)])

    def test_empty_frame_raises(self):
        with self.assertRaises(ValueError):
            self.fp.canonical_frame([], [], OBJ_NAMES, OBJS_INFO)
        with self.assertRaises(ValueError):
            self.fp.canonical_frame(PARAM_NAMES, PARAMS_INFO, [], [])

    def test_bad_minimize_flag_raises(self):
        with self.assertRaises(ValueError):
            self.fp.canonical_frame(
                PARAM_NAMES, PARAMS_INFO, OBJ_NAMES, [(0.0, 1.0, 0), (0.0, 1.0, 7)]
            )


if __name__ == "__main__":
    unittest.main()
