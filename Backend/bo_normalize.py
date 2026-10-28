# bo_normalize.py — canonical parameter/objective normalization for every BOforUnity backend.
#
# Single source of truth for "the frame": parameters live in [0,1]^d, objectives live in
# [-1,1] as a MAXIMIZATION problem (minimized objectives are sign-flipped). bo.py, mobo.py
# and the Meta-TAF source generators must all agree bit-for-bit, otherwise a transfer source
# is fitted in a different frame than the target it is transferred into.
#
# Pure numpy on purpose: no torch/botorch import, so the CI test job (numpy+pandas only)
# can exercise the frame directly.
#
# NOTE ON GLOBALS: the objective-column format is an explicit `fmt` argument, never a module
# global. mobo.py/bo.py pass their WARM_START_OBJECTIVE_FORMAT; offline source generators pass
# fmt="raw" because ObservationsPerEvaluation.csv is raw by construction.

import numpy as np

VALID_OBJECTIVE_FORMATS = ("auto", "raw", "normalized_max", "normalized_native")

# Tolerances kept identical to the historical inline implementations.
_PARAM_EPS = 1e-8
_COL_EPS = 1e-8
_VALUE_EPS = 1e-9
# A raw warm-start parameter this close outside its bounds (as a fraction of the range) is a
# rounding artifact -- logs used to be written at 3 decimals -- not evidence that the whole
# column was already normalized.
_PARAM_RAW_TOL_FRACTION = 5e-4
# Logs written before 1.8.0 rounded every value to 3 decimals, so a raw value there can lie up
# to half a unit of the third decimal outside its bounds whatever the range (0.0045 on
# [0, 0.0045] logged as 0.005; 0.1235 on [0.1, 0.1235] as 0.124).
_LEGACY_LOG_ROUNDING = 5e-4


def raw_bounds_tolerance(lo, hi):
    """How far outside [lo, hi] a warm-start parameter may lie and still be read as raw.

    Covers 3-decimal logs (half a unit of the third decimal) and, on ranges wider than 1,
    0.05% of the range. Within the tolerance a value is a rounding artifact, not evidence that
    the whole column is already normalized. Objective columns use the narrower
    objective_raw_bounds_tolerance.
    """
    return max(_PARAM_RAW_TOL_FRACTION * (hi - lo), _LEGACY_LOG_ROUNDING) + _PARAM_EPS


def objective_raw_bounds_tolerance(lo, hi):
    """How far outside [lo, hi] a warm-start objective value may lie and still be read as raw.

    Log rounding only: half a unit of the third decimal (logs written before 1.8.0) or of the
    last of LOG_SIGNIFICANT_DIGITS significant digits (current logs; matters above 1e6). No
    fraction of the range as for parameters: objective values are checked against their bounds
    before they are logged, and on a range starting at 0 the values just below 0 are what tells
    a normalized column from a raw one -- 0.05% of [0, 10000] (5) would read every normalized
    column as raw.
    """
    magnitude = max(abs(lo), abs(hi))
    return max(_LEGACY_LOG_ROUNDING, 0.5 * 10.0 ** (1 - LOG_SIGNIFICANT_DIGITS) * magnitude) + _COL_EPS


# -------------------- denormalization (frame -> original units) --------------------
# Values written to the CSV logs keep 10 significant digits: every bit of the float32 values
# Unity exchanges (~7 digits) survives, while float64 round-trip noise (5.999999999999999) is
# hidden. The former fixed 3 decimals erased small ranges (0.000274 on [0, 0.004] -> 0.0), so
# the logs could not reproduce the evaluated design, and FinalDesignSelector, warm starts and
# Meta-TAF sources then worked from different values than the participant experienced.
LOG_SIGNIFICANT_DIGITS = 10


# Hypervolume reference point per objective, in the [-1,1] maximization space shared by MOBO and
# MetaTAF. It lies slightly beyond the worst possible value: with exactly -1, a Pareto-optimal
# design rated worst on one objective (common with extreme Likert ratings) added no hypervolume,
# so it counted for nothing in the coverage metric and in qLogNEHVI's acquisition.
HYPERVOLUME_REFERENCE_VALUE = -1.1


def round_for_log(v):
    """Round to LOG_SIGNIFICANT_DIGITS significant digits for the CSV logs."""
    return float(f"{float(v):.{LOG_SIGNIFICANT_DIGITS}g}")


def denormalize_to_original_param(val01, lo, hi, decimals="log"):
    """Map a [0,1] parameter back to its original range.

    decimals="log" (default) rounds for the CSV logs (see round_for_log), None keeps full
    precision (the value sent to Unity), and an int rounds to that many decimals.
    """
    v = lo + val01 * (hi - lo)
    if decimals is None:
        return float(v)
    if decimals == "log":
        return round_for_log(v)
    return np.round(v, decimals)


def denormalize_to_original_obj(v_m1p1, lo, hi, smaller_is_better):
    """Map a [-1,1] maximization objective back to original units (undoing the sign flip)."""
    v = -v_m1p1 if int(smaller_is_better) == 1 else v_m1p1
    return round_for_log(lo + (v + 1) * 0.5 * (hi - lo))


# -------------------- the live scalar transform --------------------
def normalize_objective_value(val, lo, hi, minflag, name=None):
    """Normalize ONE raw objective value to [-1,1] maximization.

    This is the transform the live optimizer applies to every value Unity reports, and it is
    the definition offline source generators must reproduce. Raises on non-finite or
    out-of-bounds input rather than silently clipping, so a mis-framed source fails loudly.
    """
    label = f"'{name}'" if name else "value"
    try:
        val = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Objective {label} must be numeric, got {val!r}") from e
    if not np.isfinite(val):
        raise ValueError(f"Objective {label} is non-finite: {val}")

    if hi == lo:
        if not np.isclose(val, lo, rtol=0.0, atol=_VALUE_EPS):
            raise ValueError(
                f"Objective {label} value {val} is out of bounds for degenerate interval [{lo}, {hi}]"
            )
        f = 0.0
    else:
        if val < (lo - _VALUE_EPS) or val > (hi + _VALUE_EPS):
            raise ValueError(f"Objective {label} value {val} is out of bounds [{lo}, {hi}]")
        f = (val - lo) / (hi - lo) * 2 - 1
    if int(minflag) == 1:
        f *= -1
    return float(np.clip(f, -1.0, 1.0))


# -------------------- column transforms (warm start / offline generators) --------------------
def normalize_param_column(col, lo, hi, warn=None):
    """Normalize a raw (or already-normalized) parameter column into [0,1].

    The scale is inferred: values within the raw bounds are read as raw, otherwise values
    within [0,1] as already normalized. Both choices are announced through ``warn``
    (default: print) whenever the other reading would also have fit, because the two
    differ silently by up to the whole range.
    """
    if warn is None:
        def warn(msg):
            print(msg, flush=True)

    col = np.asarray(col, dtype=np.float64)
    raw_tol = raw_bounds_tolerance(lo, hi)
    in_raw_range = np.all((lo - raw_tol <= col) & (col <= hi + raw_tol))
    in_norm_range = np.all((-_PARAM_EPS <= col) & (col <= 1.0 + _PARAM_EPS))

    if hi == lo:
        if np.allclose(col, lo, rtol=0.0, atol=_PARAM_EPS):
            return np.zeros_like(col)
        if in_norm_range and np.allclose(col, 0.0, rtol=0.0, atol=_PARAM_EPS):
            return np.zeros_like(col)
        raise ValueError(
            f"Warm-start parameter values out of bounds for degenerate interval [{lo}, {hi}]"
        )

    if in_raw_range:
        if in_norm_range and (lo, hi) != (0.0, 1.0):
            warn(
                f"Warning: warm-start parameter values [{np.min(col)}, {np.max(col)}] fit both the raw "
                f"bounds [{lo}, {hi}] and [0,1]; assuming raw values."
            )
        return np.clip((col - lo) / (hi - lo), 0.0, 1.0)
    if in_norm_range:
        # Fallback for previously normalized warm-start files.
        warn(
            f"Warning: warm-start parameter values [{np.min(col)}, {np.max(col)}] fall outside the raw "
            f"bounds [{lo}, {hi}] but inside [0,1]; treating the column as already normalized."
        )
        return np.clip(col, 0.0, 1.0)
    raise ValueError(
        f"Warm-start parameter values must be within raw bounds [{lo}, {hi}] or normalized [0,1], "
        f"got range [{np.min(col)}, {np.max(col)}]"
    )


def normalize_obj_column(col, lo, hi, minflag, fmt="auto", warn=None):
    """Normalize a raw (or already-normalized) objective column into [-1,1] maximization.

    ``fmt`` selects how the incoming scale is interpreted and must be one of
    VALID_OBJECTIVE_FORMATS. With "auto", values within the raw bounds (up to log rounding,
    objective_raw_bounds_tolerance) are read as raw, otherwise values within
    [-1,1] as already normalized (maximize-space). ``warn`` (default: print) announces the
    choice whenever the other reading would also have fit and differs, and always announces the
    normalized fallback, because the two readings differ silently by up to the whole range.
    """
    if fmt not in VALID_OBJECTIVE_FORMATS:
        raise ValueError(
            f"fmt must be one of {VALID_OBJECTIVE_FORMATS}; got {fmt!r}"
        )
    if warn is None:
        def warn(msg):
            print(msg, flush=True)

    col = np.asarray(col, dtype=np.float64)
    raw_tol = objective_raw_bounds_tolerance(lo, hi)
    raw_range_detected =np.all((lo - raw_tol <= col) & (col <= hi + raw_tol))
    norm_range_detected = np.all((-1.0 - _COL_EPS <= col) & (col <= 1.0 + _COL_EPS))
    in_raw_range = raw_range_detected
    in_norm_range = norm_range_detected

    if fmt == "raw":
        if not raw_range_detected:
            raise ValueError(
                f"warmStartObjectiveFormat=raw requires values in [{lo},{hi}], "
                f"but received range [{np.min(col)}, {np.max(col)}]"
            )
        in_raw_range = True
        in_norm_range = False
    elif fmt == "normalized_max":
        if not norm_range_detected:
            raise ValueError(
                f"warmStartObjectiveFormat=normalized_max requires values in [-1,1], "
                f"but received range [{np.min(col)}, {np.max(col)}]"
            )
        in_raw_range = False
        in_norm_range = True
    elif fmt == "normalized_native":
        if not norm_range_detected:
            raise ValueError(
                f"warmStartObjectiveFormat=normalized_native requires values in [-1,1], "
                f"but received range [{np.min(col)}, {np.max(col)}]"
            )
        in_raw_range = False
        in_norm_range = True

    if in_raw_range:
        # Bounds [-1,1] on a maximized objective: both readings give the same values.
        readings_coincide = (lo, hi) == (-1.0, 1.0) and int(minflag) == 0
        if fmt == "auto" and in_norm_range and not readings_coincide:
            warn(
                f"Warning: warm-start objective values [{np.min(col)}, {np.max(col)}] are ambiguous: they "
                f"fit both the raw bounds [{lo}, {hi}] and [-1,1]; assuming raw values. Set the warm-start "
                "objective format if they are normalized."
            )
        if hi == lo:
            y = np.zeros_like(col)
        else:
            y = (col - lo) / (hi - lo) * 2.0 - 1.0
            if int(minflag) == 1:
                y = -y
    elif in_norm_range:
        if fmt == "auto":
            warn(
                f"Warning: warm-start objective values [{np.min(col)}, {np.max(col)}] fall outside the raw "
                f"bounds [{lo}, {hi}] but inside [-1,1]; treating the column as already normalized "
                "(maximize-space)."
            )
        # already normalized
        y = np.clip(col, -1.0, 1.0)
        if fmt == "normalized_native" and int(minflag) == 1:
            y = -y
    else:
        raise ValueError(
            f"Warm-start objective values must be within raw bounds [{lo}, {hi}] or normalized [-1,1], "
            f"got range [{np.min(col)}, {np.max(col)}]"
        )
    return np.clip(y, -1.0, 1.0)
