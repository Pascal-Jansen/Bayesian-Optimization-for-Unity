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


# -------------------- denormalization (frame -> original units) --------------------
def denormalize_to_original_param(val01, lo, hi, decimals=3):
    """Map a [0,1] parameter back to its original range. decimals=None keeps full precision."""
    v = lo + val01 * (hi - lo)
    if decimals is None:
        return float(v)
    return np.round(v, decimals)


def denormalize_to_original_obj(v_m1p1, lo, hi, smaller_is_better):
    """Map a [-1,1] maximization objective back to original units (undoing the sign flip)."""
    v = -v_m1p1 if int(smaller_is_better) == 1 else v_m1p1
    return np.round(lo + (v + 1) * 0.5 * (hi - lo), 3)


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
def normalize_param_column(col, lo, hi):
    """Normalize a raw (or already-normalized) parameter column into [0,1]."""
    col = np.asarray(col, dtype=np.float64)
    in_raw_range = np.all((lo - _PARAM_EPS <= col) & (col <= hi + _PARAM_EPS))
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
        return np.clip((col - lo) / (hi - lo), 0.0, 1.0)
    if in_norm_range:
        # Fallback for previously normalized warm-start files.
        return np.clip(col, 0.0, 1.0)
    raise ValueError(
        f"Warm-start parameter values must be within raw bounds [{lo}, {hi}] or normalized [0,1], "
        f"got range [{np.min(col)}, {np.max(col)}]"
    )


def normalize_obj_column(col, lo, hi, minflag, fmt="auto", warn=None):
    """Normalize a raw (or already-normalized) objective column into [-1,1] maximization.

    ``fmt`` selects how the incoming scale is interpreted and must be one of
    VALID_OBJECTIVE_FORMATS. ``warn`` is an optional callable used for the ambiguous-scale
    notice; it defaults to printing, matching the historical behaviour.
    """
    if fmt not in VALID_OBJECTIVE_FORMATS:
        raise ValueError(
            f"fmt must be one of {VALID_OBJECTIVE_FORMATS}; got {fmt!r}"
        )
    if warn is None:
        def warn(msg):
            print(msg, flush=True)

    col = np.asarray(col, dtype=np.float64)
    raw_range_detected = np.all((lo - _COL_EPS <= col) & (col <= hi + _COL_EPS))
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
        if fmt == "auto" and in_norm_range:
            warn(
                "Warning: warm-start objective values are ambiguous (fit both raw bounds and [-1,1]); assuming raw scale."
            )
        if hi == lo:
            y = np.zeros_like(col)
        else:
            y = (col - lo) / (hi - lo) * 2.0 - 1.0
            if int(minflag) == 1:
                y = -y
    elif in_norm_range:
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
