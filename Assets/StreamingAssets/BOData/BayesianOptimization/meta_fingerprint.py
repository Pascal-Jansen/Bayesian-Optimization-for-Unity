# meta_fingerprint.py — canonical "frame" fingerprint for Meta-TAF source artifacts.
#
# WHY THIS EXISTS
# A transfer source is only meaningful in the frame it was fitted in. Two studies can share
# the same parameter count d and objective count M yet differ in raw bounds or in an
# objective's minimize flag. Because every artifact is stored already normalized ([0,1]^d
# parameters, [-1,1] maximization objectives), such a mismatch is INVISIBLE to a shape check:
# the source loads cleanly and transfers a rescaled -- or, for a flipped minflag, an exactly
# INVERTED -- response surface into the target. That is silent scientific corruption, not a
# crash, so it must be caught structurally.
#
# Every source artifact therefore stores the canonical frame it was built from, and the
# runtime refuses (or skips) any source whose frame disagrees with the live study.
#
# Pure stdlib on purpose: imported by the runtime loader AND by the offline generators, and
# unit-testable in the numpy+pandas-only CI job.

import hashlib
import json

FRAME_SCHEMA_VERSION = 1


def _fmt(v):
    """Format a bound as a stable string.

    Bounds are compared as formatted strings rather than floats so that a value which
    round-trips through JSON, C#, and different platforms cannot drift in the last bit and
    turn a matching frame into a mismatched one. 12 significant digits is well inside double
    precision while staying insensitive to repr differences.
    """
    return format(float(v), ".12g")


def canonical_frame(parameter_names, parameters_info, objective_names, objectives_info):
    """Build the canonical frame descriptor.

    parameters_info: sequence of (lo, hi) aligned with parameter_names.
    objectives_info: sequence of (lo, hi, minflag) aligned with objective_names.
    """
    parameter_names = list(parameter_names)
    objective_names = list(objective_names)
    parameters_info = list(parameters_info)
    objectives_info = list(objectives_info)

    if len(parameter_names) != len(parameters_info):
        raise ValueError(
            f"parameter_names ({len(parameter_names)}) and parameters_info "
            f"({len(parameters_info)}) must be the same length"
        )
    if len(objective_names) != len(objectives_info):
        raise ValueError(
            f"objective_names ({len(objective_names)}) and objectives_info "
            f"({len(objectives_info)}) must be the same length"
        )
    if not parameter_names:
        raise ValueError("a frame needs at least one parameter")
    if not objective_names:
        raise ValueError("a frame needs at least one objective")

    params = []
    for name, bounds in zip(parameter_names, parameters_info):
        lo, hi = bounds[0], bounds[1]
        params.append([str(name), _fmt(lo), _fmt(hi)])

    objs = []
    for name, info in zip(objective_names, objectives_info):
        lo, hi, minflag = info[0], info[1], info[2]
        if int(minflag) not in (0, 1):
            raise ValueError(
                f"objective '{name}' minimize flag must be 0 or 1, got {minflag!r}"
            )
        objs.append([str(name), _fmt(lo), _fmt(hi), int(minflag)])

    return {
        "v": FRAME_SCHEMA_VERSION,
        "d": len(parameter_names),
        "M": len(objective_names),
        "params": params,
        "objs": objs,
    }


def frame_digest(frame):
    """Short stable digest of a frame, for logging and manifest identity."""
    blob = json.dumps(frame, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def frame_differences(expected, actual):
    """Field-level differences between two frames; empty list means compatible.

    Returns human-readable strings rather than a bool so the operator is told exactly which
    field diverged -- 'objective 2 minimize flag 0 != 1' is actionable, 'frame mismatch' is not.
    """
    diffs = []
    if not isinstance(actual, dict):
        return [f"frame is missing or not an object (got {type(actual).__name__})"]

    exp_v, act_v = expected.get("v"), actual.get("v")
    if exp_v != act_v:
        diffs.append(f"frame schema version {act_v!r} != expected {exp_v!r}")
        # A different schema version makes field-by-field comparison meaningless.
        return diffs

    for key, label in (("d", "parameter count"), ("M", "objective count")):
        if expected.get(key) != actual.get(key):
            diffs.append(f"{label} {actual.get(key)!r} != expected {expected.get(key)!r}")
    if diffs:
        # Comparing per-entry lists of different length adds noise, not information.
        return diffs

    for i, (exp_p, act_p) in enumerate(zip(expected.get("params", []), actual.get("params", []))):
        if exp_p[0] != act_p[0]:
            diffs.append(f"parameter {i} name {act_p[0]!r} != expected {exp_p[0]!r}")
        elif exp_p[1:] != act_p[1:]:
            diffs.append(
                f"parameter {i} ({exp_p[0]}) bounds [{act_p[1]}, {act_p[2]}] "
                f"!= expected [{exp_p[1]}, {exp_p[2]}]"
            )

    for i, (exp_o, act_o) in enumerate(zip(expected.get("objs", []), actual.get("objs", []))):
        if exp_o[0] != act_o[0]:
            diffs.append(f"objective {i} name {act_o[0]!r} != expected {exp_o[0]!r}")
            continue
        if exp_o[1:3] != act_o[1:3]:
            diffs.append(
                f"objective {i} ({exp_o[0]}) bounds [{act_o[1]}, {act_o[2]}] "
                f"!= expected [{exp_o[1]}, {exp_o[2]}]"
            )
        if exp_o[3] != act_o[3]:
            # Called out separately: this is the inverted-surface case.
            diffs.append(
                f"objective {i} ({exp_o[0]}) minimize flag {act_o[3]} != expected {exp_o[3]} "
                "(transferring this source would invert its response surface)"
            )

    return diffs


def frames_compatible(expected, actual):
    """True when `actual` may be transferred into `expected`."""
    return not frame_differences(expected, actual)
