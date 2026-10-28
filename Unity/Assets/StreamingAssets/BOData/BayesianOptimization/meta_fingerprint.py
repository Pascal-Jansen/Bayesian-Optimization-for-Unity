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
# unit-testable in the numpy+pandas-only CI job. artifact_sha256 is the file identity both
# sides compare (population.json, the trajectory stamp).

import hashlib
import json

FRAME_SCHEMA_VERSION = 1


def _fmt(v):
    """Format a bound as a stable string.

    Bounds are compared as formatted strings rather than floats so that a value which
    round-trips through JSON, C#, and different platforms cannot drift in the last bit and
    turn a matching frame into a mismatched one. 12 significant digits is well inside double
    precision while staying insensitive to repr differences. Negative zero is written as
    "0": -0.0 == 0.0 as a bound, but formats as "-0" (e.g. a C# float negated to -0).
    """
    v = float(v)
    if v == 0.0:
        v = 0.0  # drops the sign of -0.0
    return format(v, ".12g")


def _canonical_bound(value):
    """A stored bound (string or number) in _fmt's form; None when it is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return _fmt(number)


def _entry_problem(entry, n_fields, label):
    """Why a stored params/objs entry is malformed (None when it is well-formed)."""
    if not isinstance(entry, (list, tuple)) or len(entry) != n_fields:
        return f"{label} is not a list of {n_fields} fields (got {entry!r})"
    if not isinstance(entry[0], str):
        return f"{label} name is not a string (got {entry[0]!r})"
    for bound in entry[1:3]:
        if _canonical_bound(bound) is None:
            return f"{label} bound {bound!r} is not a finite number"
    if n_fields == 4 and (isinstance(entry[3], bool) or entry[3] not in (0, 1)):
        return f"{label} minimize flag {entry[3]!r} is not 0 or 1"
    return None


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

    # The lists must hold exactly d / M well-formed entries. zip() alone would stop at the
    # shorter list, so a frame with its params/objs missing or cut short (hand-edited, or a
    # truncated copy) compared as compatible without a single entry being checked.
    entries = {}
    for key, count_key, n_fields, label in (("params", "d", 3, "parameter"),
                                            ("objs", "M", 4, "objective")):
        exp_list = expected.get(key) or []
        act_list = actual.get(key)
        if not isinstance(act_list, list) or len(act_list) != len(exp_list):
            got = len(act_list) if isinstance(act_list, list) else type(act_list).__name__
            diffs.append(
                f"'{key}' must list {len(exp_list)} {label}(s) (= {count_key}), got {got}"
            )
            continue
        problems = [_entry_problem(entry, n_fields, f"{label} {i}") for i, entry in enumerate(act_list)]
        problems = [p for p in problems if p]
        if problems:
            diffs.extend(problems)
            continue
        entries[key] = (exp_list, act_list)
    if diffs:
        return diffs

    def bounds(entry):
        # Stored strings are re-canonicalized, so an artifact written before the "-0"
        # normalization still matches a live "0".
        return [_canonical_bound(entry[1]), _canonical_bound(entry[2])]

    exp_params, act_params = entries["params"]
    for i, (exp_p, act_p) in enumerate(zip(exp_params, act_params)):
        if exp_p[0] != act_p[0]:
            diffs.append(f"parameter {i} name {act_p[0]!r} != expected {exp_p[0]!r}")
        elif bounds(exp_p) != bounds(act_p):
            diffs.append(
                f"parameter {i} ({exp_p[0]}) bounds [{act_p[1]}, {act_p[2]}] "
                f"!= expected [{exp_p[1]}, {exp_p[2]}]"
            )

    exp_objs, act_objs = entries["objs"]
    for i, (exp_o, act_o) in enumerate(zip(exp_objs, act_objs)):
        if exp_o[0] != act_o[0]:
            diffs.append(f"objective {i} name {act_o[0]!r} != expected {exp_o[0]!r}")
            continue
        if bounds(exp_o) != bounds(act_o):
            diffs.append(
                f"objective {i} ({exp_o[0]}) bounds [{act_o[1]}, {act_o[2]}] "
                f"!= expected [{exp_o[1]}, {exp_o[2]}]"
            )
        if int(exp_o[3]) != int(act_o[3]):
            # Called out separately: this is the inverted-surface case.
            diffs.append(
                f"objective {i} ({exp_o[0]}) minimize flag {act_o[3]} != expected {exp_o[3]} "
                "(transferring this source would invert its response surface)"
            )

    return diffs


def frames_compatible(expected, actual):
    """True when `actual` may be transferred into `expected`."""
    return not frame_differences(expected, actual)


def artifact_sha256(path):
    """SHA-256 of a JSON artifact file, with CRLF line endings read as LF.

    The identity of a source file in population.json, in the gp_state's trajectory stamp and
    in MetaRunState.json. JSON cannot hold a raw line break inside a value, so the conversion
    changes nothing a loader reads; but git (core.autocrlf, the Git for Windows default) and
    some sync tools convert line endings on checkout, and a hash of the raw bytes then reported
    an unchanged population as replaced on every machine except the one that built it. Files
    written with LF line endings (as meta_train.py writes them) hash like the raw bytes.
    """
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
