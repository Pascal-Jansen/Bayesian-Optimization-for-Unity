"""Tier-1 tests for bo_normalize.py — the canonical [0,1]^d / [-1,1]-maximization frame.

Pure numpy: runs in the numpy+pandas-only CI job with no torch/botorch present.

The parity tests are the point of this file. Offline Meta-TAF source generators normalize with
bo_normalize while the live optimizer normalizes inside mobo.py; if those two ever diverge, a
source is fitted in a different frame than the target it transfers into, which is silent and
unrecoverable. These tests pin them together.
"""

import importlib.util
import os
import pathlib
import sys
import unittest
import uuid

import numpy as np

# Support both `discover tests` (tests/ on sys.path) and direct module runs.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import install_stub_modules  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BO_DIR = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization"
NORMALIZE_PATH = BO_DIR / "bo_normalize.py"
MOBO_PATH = BO_DIR / "mobo.py"


def _load(path, prefix):
    name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_normalize():
    # No stubs needed: bo_normalize imports numpy only.
    return _load(NORMALIZE_PATH, "bo_normalize_test")


def load_mobo():
    install_stub_modules()
    return _load(MOBO_PATH, "mobo_parity_test")


class ObjectiveValueTests(unittest.TestCase):
    def setUp(self):
        self.n = load_normalize()

    def test_maximized_objective_maps_bounds_to_minus_one_and_one(self):
        self.assertAlmostEqual(self.n.normalize_objective_value(0.0, 0.0, 10.0, 0), -1.0)
        self.assertAlmostEqual(self.n.normalize_objective_value(10.0, 0.0, 10.0, 0), 1.0)
        self.assertAlmostEqual(self.n.normalize_objective_value(5.0, 0.0, 10.0, 0), 0.0)

    def test_minimized_objective_is_sign_flipped(self):
        # minflag=1 means "smaller is better", so the low bound is the BEST outcome.
        self.assertAlmostEqual(self.n.normalize_objective_value(0.0, 0.0, 10.0, 1), 1.0)
        self.assertAlmostEqual(self.n.normalize_objective_value(10.0, 0.0, 10.0, 1), -1.0)

    def test_degenerate_interval_maps_to_zero(self):
        self.assertEqual(self.n.normalize_objective_value(3.0, 3.0, 3.0, 0), 0.0)

    def test_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            self.n.normalize_objective_value(11.0, 0.0, 10.0, 0)
        with self.assertRaises(ValueError):
            self.n.normalize_objective_value(-1.0, 0.0, 10.0, 0)

    def test_non_finite_raises(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.n.normalize_objective_value(bad, 0.0, 10.0, 0)

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            self.n.normalize_objective_value("abc", 0.0, 10.0, 0)

    def test_round_trip_through_denormalize(self):
        for raw, minflag in ((2.5, 0), (2.5, 1), (7.5, 0), (7.5, 1)):
            f = self.n.normalize_objective_value(raw, 0.0, 10.0, minflag)
            back = self.n.denormalize_to_original_obj(f, 0.0, 10.0, minflag)
            self.assertAlmostEqual(float(back), raw, places=6)

    def test_parameter_round_trip(self):
        back = self.n.denormalize_to_original_param(0.25, 4.0, 8.0, decimals=None)
        self.assertAlmostEqual(back, 5.0)


class ObjectiveColumnTests(unittest.TestCase):
    def setUp(self):
        self.n = load_normalize()

    def test_raw_column_matches_scalar_transform(self):
        col = [0.0, 2.5, 5.0, 10.0]
        got = self.n.normalize_obj_column(col, 0.0, 10.0, 0, fmt="raw")
        want = [self.n.normalize_objective_value(v, 0.0, 10.0, 0) for v in col]
        np.testing.assert_allclose(got, want)

    def test_raw_column_minflag_matches_scalar_transform(self):
        col = [0.0, 2.5, 5.0, 10.0]
        got = self.n.normalize_obj_column(col, 0.0, 10.0, 1, fmt="raw")
        want = [self.n.normalize_objective_value(v, 0.0, 10.0, 1) for v in col]
        np.testing.assert_allclose(got, want)

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            self.n.normalize_obj_column([1.0], 0.0, 10.0, 0, fmt="nonsense")

    def test_raw_format_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            self.n.normalize_obj_column([50.0], 0.0, 10.0, 0, fmt="raw")

    def test_format_is_a_parameter_not_a_global(self):
        # Same input, two formats, two different results -- proves no hidden module state.
        col = [-1.0, 0.0, 1.0]
        as_norm = self.n.normalize_obj_column(col, -1.0, 1.0, 1, fmt="normalized_max")
        as_native = self.n.normalize_obj_column(col, -1.0, 1.0, 1, fmt="normalized_native")
        np.testing.assert_allclose(as_norm, [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(as_native, [1.0, 0.0, -1.0])

    def test_ambiguous_warning_is_routed_to_callback(self):
        seen = []
        # Bounds [-1,1] make raw and normalized ranges coincide; for a minimized objective the
        # two readings differ in sign.
        got = self.n.normalize_obj_column([0.5], -1.0, 1.0, 1, fmt="auto", warn=seen.append)
        np.testing.assert_allclose(got, [-0.5])  # raw reading
        self.assertTrue(any("ambiguous" in m and "assuming raw" in m for m in seen))

    def test_coinciding_objective_readings_are_not_announced(self):
        # Bounds [-1,1] on a maximized objective: raw and normalized readings are identical.
        seen = []
        got = self.n.normalize_obj_column([0.5, -0.2], -1.0, 1.0, 0, fmt="auto", warn=seen.append)
        np.testing.assert_allclose(got, [0.5, -0.2])
        self.assertEqual(seen, [])

    def test_objective_column_rounding_artifact_stays_raw(self):
        # Minimized objective on [0, 0.0045] from a log written at 3 decimals: the float32
        # bound 0.0045 was logged as 0.005. The column used to be read as already normalized
        # (every value then sat at mid-scale, ~0); it is raw.
        seen = []
        got = self.n.normalize_obj_column([0.0, 0.002, 0.005], 0.0, 0.0045, 1, fmt="auto", warn=seen.append)
        np.testing.assert_allclose(got, [1.0, 1.0 - 2 * 0.002 / 0.0045, -1.0])
        self.assertTrue(any("assuming raw" in m for m in seen))
        # The explicit raw format accepts the same artifact (it used to raise).
        np.testing.assert_allclose(
            self.n.normalize_obj_column([0.0, 0.005], 0.0, 0.0045, 1, fmt="raw"), [1.0, -1.0]
        )

    def test_objective_column_tolerance_does_not_grow_with_the_range(self):
        # Only log rounding counts as a raw artifact: 100.0004 on [0, 100] does, 100.04 does not.
        np.testing.assert_allclose(
            self.n.normalize_obj_column([0.0, 100.0004], 0.0, 100.0, 0, fmt="raw"), [-1.0, 1.0]
        )
        with self.assertRaises(ValueError):
            self.n.normalize_obj_column([0.0, 100.04], 0.0, 100.0, 0, fmt="raw")
        # A tolerance of 0.05% of the range swallowed the values just below 0 that tell a
        # normalized column from a raw one: normalized SUS scores [-0.04, 0.3, 0.8] on [0, 100]
        # and normalized task times on [0, 10000] (tolerance 5: every normalized column) were
        # read as raw, i.e. as the worst / best values of the range.
        for col, lo, hi, minflag in (([-0.04, 0.3, 0.8], 0.0, 100.0, 0),
                                     ([-0.95, 0.2, 0.9], 0.0, 10000.0, 1)):
            with self.subTest(hi=hi):
                seen = []
                got = self.n.normalize_obj_column(col, lo, hi, minflag, fmt="auto", warn=seen.append)
                np.testing.assert_allclose(got, col)
                self.assertTrue(any("already normalized" in m for m in seen))

    def test_objective_raw_bounds_tolerance(self):
        # Half a unit of the third decimal, or of the 10th significant digit above 1e6.
        self.assertAlmostEqual(self.n.objective_raw_bounds_tolerance(0.0, 0.0045), 5e-4, places=7)
        self.assertAlmostEqual(self.n.objective_raw_bounds_tolerance(0.0, 10000.0), 5e-4, places=7)
        self.assertAlmostEqual(self.n.objective_raw_bounds_tolerance(-1e8, 0.0), 0.05, places=7)
        # Current logs keep 10 significant digits: a bound such as 123456789.98 logs as
        # 123456790.0, still read as raw.
        np.testing.assert_allclose(
            self.n.normalize_obj_column([0.0, 123456790.0], 0.0, 123456789.98, 0, fmt="raw"), [-1.0, 1.0]
        )

    def test_objective_column_normalized_fallback_is_announced(self):
        seen = []
        got = self.n.normalize_obj_column([-0.5, 0.25], 2.0, 8.0, 1, fmt="auto", warn=seen.append)
        np.testing.assert_allclose(got, [-0.5, 0.25])
        self.assertTrue(any("already normalized" in m for m in seen))

    def test_objective_column_explicit_formats_are_not_announced(self):
        seen = []
        self.n.normalize_obj_column([-0.5, 0.25], 2.0, 8.0, 1, fmt="normalized_max", warn=seen.append)
        self.n.normalize_obj_column([0.5], -1.0, 1.0, 1, fmt="raw", warn=seen.append)
        self.assertEqual(seen, [])

    def test_raw_bounds_tolerance(self):
        # Half a unit of the third decimal (3-decimal logs), or 0.05% of a range wider than 1.
        self.assertAlmostEqual(self.n.raw_bounds_tolerance(0.0, 0.0045), 5e-4, places=7)
        self.assertAlmostEqual(self.n.raw_bounds_tolerance(0.0, 1.0), 5e-4, places=7)
        self.assertAlmostEqual(self.n.raw_bounds_tolerance(0.0, 100.0), 0.05, places=7)

    def test_param_column_raw_and_normalized(self):
        np.testing.assert_allclose(
            self.n.normalize_param_column([4.0, 6.0, 8.0], 4.0, 8.0), [0.0, 0.5, 1.0]
        )
        # Values already inside [0,1] but outside raw bounds fall back to normalized.
        np.testing.assert_allclose(
            self.n.normalize_param_column([0.0, 0.5, 1.0], 4.0, 8.0), [0.0, 0.5, 1.0]
        )

    def test_param_column_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            self.n.normalize_param_column([-5.0, 99.0], 4.0, 8.0)

    def test_param_column_rounding_artifact_stays_raw(self):
        # 0.1235 logged as 0.124 at 3 decimals sits just outside [0.1, 0.1235]; it must not
        # flip the whole column to the "already normalized" reading.
        seen = []
        got = self.n.normalize_param_column([0.1, 0.11, 0.1235000001], 0.1, 0.1235, warn=seen.append)
        np.testing.assert_allclose(got, [0.0, 0.01 / 0.0235, 1.0])
        self.assertTrue(any("assuming raw" in m for m in seen))

    def test_param_column_three_decimal_log_on_a_narrow_range_stays_raw(self):
        # 0.1235 logged as 0.124 at 3 decimals: 0.0005 outside [0.1, 0.1235], far more than
        # 0.05% of that range. It used to switch the column to the normalized reading.
        seen = []
        got = self.n.normalize_param_column([0.1, 0.11, 0.124], 0.1, 0.1235, warn=seen.append)
        np.testing.assert_allclose(got, [0.0, 0.01 / 0.0235, 1.0])
        self.assertTrue(any("assuming raw" in m for m in seen))

    def test_param_column_normalized_fallback_is_announced(self):
        seen = []
        got = self.n.normalize_param_column([0.0, 0.5, 1.0], 4.0, 8.0, warn=seen.append)
        np.testing.assert_allclose(got, [0.0, 0.5, 1.0])
        self.assertTrue(any("already normalized" in m for m in seen))

    def test_param_column_unit_bounds_are_not_ambiguous(self):
        seen = []
        self.n.normalize_param_column([0.2, 0.9], 0.0, 1.0, warn=seen.append)
        self.assertEqual(seen, [])


class LogPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.n = load_normalize()

    def test_small_ranges_survive_logging(self):
        # 3 fixed decimals logged 0.000274 on [0, 0.004] as 0.0.
        self.assertAlmostEqual(self.n.denormalize_to_original_param(0.0685, 0.0, 0.004), 0.000274)
        self.assertNotEqual(
            self.n.denormalize_to_original_obj(0.0, 0.0, 0.00824, 0),
            self.n.denormalize_to_original_obj(0.0, 0.0, 0.00700, 0),
        )

    def test_float64_round_trip_noise_is_hidden(self):
        f = self.n.normalize_objective_value(6.0, 0.0, 10.0, 1)
        self.assertEqual(self.n.denormalize_to_original_obj(f, 0.0, 10.0, 1), 6.0)
        self.assertEqual(self.n.denormalize_to_original_param(0.3, 0.0, 1.0), 0.3)


class MoboFrameParityTests(unittest.TestCase):
    """bo_normalize must reproduce mobo.py's frame exactly."""

    def setUp(self):
        self.n = load_normalize()
        self.mobo = load_mobo()

    def test_objective_column_parity_across_bounds_and_minflags(self):
        cases = [
            ([0.0, 1.0, 5.0, 10.0], 0.0, 10.0, 0),
            ([0.0, 1.0, 5.0, 10.0], 0.0, 10.0, 1),
            ([-5.0, 0.0, 5.0], -5.0, 5.0, 0),
            ([-5.0, 0.0, 5.0], -5.0, 5.0, 1),
            ([1.0, 2.0, 3.0], 1.0, 3.0, 1),
        ]
        for col, lo, hi, minflag in cases:
            with self.subTest(lo=lo, hi=hi, minflag=minflag):
                self.mobo.WARM_START_OBJECTIVE_FORMAT = "raw"
                want = self.mobo.normalize_obj_column(col, lo, hi, minflag)
                got = self.n.normalize_obj_column(col, lo, hi, minflag, fmt="raw")
                np.testing.assert_array_equal(got, want)

    def test_parameter_column_parity(self):
        for col, lo, hi in (([4.0, 6.0, 8.0], 4.0, 8.0), ([0.0, 0.5, 1.0], 0.0, 1.0)):
            with self.subTest(lo=lo, hi=hi):
                np.testing.assert_array_equal(
                    self.n.normalize_param_column(col, lo, hi),
                    self.mobo.normalize_param_column(col, lo, hi),
                )

    def test_denormalize_parity(self):
        for v, lo, hi, minflag in ((0.5, 0.0, 10.0, 0), (-0.5, 0.0, 10.0, 1)):
            with self.subTest(v=v, minflag=minflag):
                self.assertEqual(
                    float(self.n.denormalize_to_original_obj(v, lo, hi, minflag)),
                    float(self.mobo.denormalize_to_original_obj(v, lo, hi, minflag)),
                )

    def test_scalar_transform_matches_mobo_live_objective_math(self):
        """The scalar path must equal the arithmetic mobo's objective_function used inline before it
        called bo_normalize (see test_bo.test_objective_function_matches_former_inline_transform)."""
        for raw, lo, hi, minflag in ((2.0, 0.0, 10.0, 0), (2.0, 0.0, 10.0, 1), (7.0, 1.0, 9.0, 1)):
            with self.subTest(raw=raw, minflag=minflag):
                f = (raw - lo) / (hi - lo) * 2 - 1
                if int(minflag) == 1:
                    f *= -1
                expected = float(np.clip(f, -1.0, 1.0))
                self.assertEqual(
                    self.n.normalize_objective_value(raw, lo, hi, minflag), expected
                )


if __name__ == "__main__":
    unittest.main()
