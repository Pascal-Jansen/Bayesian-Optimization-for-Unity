# meta_mobo_runtime.py — multi-objective Meta-BO backend (TAF-EHVI, NDJSON protocol).
#
# Speaks the same wire protocol as mobo.py (TCP server on port 56001, newline-delimited
# JSON: one init message, then a parameters -> objectives loop with tempCoverage /
# coverage / optimization_finished updates) and writes the same CSV family, but the
# optimizer is openbo's MOTAFSequentialOptimizer: BoTorch qLogNEHVI blended with
# hypervolume-improvement terms from "source" models built offline from PRIOR runs
# (population models in the sense of Liao et al., CHI '24). With zero usable sources the
# acquisition would degenerate exactly to plain qLogNEHVI, i.e. behave like a mobo.py
# run -- which would silently turn a MetaTAF study condition into the no-transfer control.
# The backend therefore refuses to start when no source is actually LOADED by openbo (not
# merely staged: openbo warns and skips artifacts it cannot use), unless 'Meta Require
# Sources' is explicitly disabled in Unity. Every run records what it used, with library
# versions and the exact optimizer configuration, in MetaRunState.json, which is rewritten
# after every evaluation with the run's progress and, at the end, why it finished. When the
# source folder carries a population manifest (population.json, written by meta_train.py),
# the loaded population must match it exactly, or the run aborts before the first trial.
#
# Deliberate scope limits (validated, not silently ignored):
#   - multi-objective only (nObjectives >= 2),
#   - no warm start (population models are the transfer mechanism here),
#   - no contextual optimization (the LCE-M context pipeline stays BoTorch-only).
#
# openbo (https://github.com/M-Colley/openbo) is imported lazily inside the run so this
# module can be imported -- and its protocol/CSV logic tested -- without the heavy stack.

import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
import warnings

import numpy as np

# Sibling module imports must work both when running this file as a script and
# when loading it from another working directory (e.g. the test suite).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import bo_normalize
import meta_fingerprint
# Socket, NDJSON, init parsing and log-writing plumbing shared by every backend.
from bo_protocol import (
    SOCKET_ACCEPT_TIMEOUT_SEC, SOCKET_TIMEOUT_SEC, StopRequested,
    accept_unity_connection, append_csv_rows, close_connection, configure_listener_socket,
    create_csv_file, flush_pending_logs, get_cfg_bool, get_cfg_float, get_cfg_int,
    get_unique_folder, log_exists, parse_obj_init, parse_param_init, parse_user_ids,
    read_observation_log, receive_init_message, recv_objectives_blocking, send_json_line,
    validate_objective_bounds, validate_parameter_bounds, validate_raw_samples,
    write_csv_rows, write_data_to_csv, write_dataframe_csv, write_log_text,
)

# -------------------- defaults (overwritten by Unity init) --------------------
N_INITIAL = 5
N_ITERATIONS = 10
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 1024
MC_SAMPLES = 128  # matches Unity's default (and mobo.py's fallback)
SEED = 3

PROBLEM_DIM = None
NUM_OBJS = None  # must be >= 2

# Meta-TAF configuration (overwritten from the init message's meta* fields).
META_SOURCE_DIR = "MetaSources"
META_REQUIRE_SOURCES = True  # abort (not silently degrade) when no valid source remains
META_WEIGHT_MODE = "taf_r"
META_RHO = 1.0
META_TARGET_WEIGHT = 1.0
META_WARMUP_ITERS = 1
META_DECAY_START_ITER = 2
META_DECAY_RATE = 0.3

# paths/state
PROJECT_PATH = ""

# study info
USER_ID = ""
CONDITION_ID = ""
GROUP_ID = ""
USER_LOG_ID = ""
CONDITION_LOG_ID = ""

# names and meta parsed from init
parameter_names = []
objective_names = []
parameters_info = []   # [(lo, hi)]
objectives_info = []   # [(lo, hi, minimizeFlag)]
FRAME = None           # canonical frame of THIS study (meta_fingerprint.canonical_frame)
INIT_CONFIG = {}       # the init message's 'config' block as received (MetaRunState.json)

# Objectives live in [-1, 1] maximization. The hypervolume reference point lies slightly beyond
# the worst value in every objective (shared with mobo.py), so a Pareto-optimal design rated
# worst on one objective still counts; it is passed to openbo for qLogNEHVI, the source terms
# and the logged hypervolume alike.
REF_POINT_VALUE = bo_normalize.HYPERVOLUME_REFERENCE_VALUE

# Leading columns of ObservationsPerEvaluation.csv. Parameter/objective keys must not reuse
# them: pandas would mangle the duplicated header ("Phase.1") and the run would abort with
# a column mismatch right after the sampling phase, i.e. after the participant's first trials.
LOG_FIXED_COLUMNS = ('UserID', 'ConditionID', 'GroupID', 'Timestamp', 'Iteration', 'Phase', 'IsPareto')

# Leading columns of MetaWeightsPerEvaluation.csv (one column per loaded source follows).
# Iteration is the global evaluation index of ObservationsPerEvaluation.csv; OptimizationStep
# counts the optimization iterations only (1..numOptimizationIterations).
WEIGHTS_LOG_COLUMNS = ("Iteration", "OptimizationStep", "TargetWeight", "DecayFactor")

RUN_STATE_FILENAME = "MetaRunState.json"
RUN_STATE_SCHEMA_VERSION = 2
# Written by meta_train.py next to gp_states/ and trajectories/: the frozen population.
MANIFEST_FILENAME = "population.json"
# Distributions whose versions decide the numbers a run produces (MetaRunState.json).
RUN_STATE_DISTRIBUTIONS = (
    "torch", "botorch", "gpytorch", "linear_operator", "open-bo",
    "numpy", "scipy", "pandas", "moocore",
)
# openbo modules on the optimization path; their source hashes pin the exact code even for
# an editable install (open-bo keeps version 0.1.0 across behaviour changes).
OPENBO_MODULES = (
    "openbo.optimizers.mobo_taf",
    "openbo.optimizers.mobo_botorch",
    "openbo.acquisition.taf_mo_ehvi",
    "openbo.acquisition.taf",
)

# -------------------- IO utils --------------------
# Sockets, NDJSON and the CSV log writers live in bo_protocol (same contract in every backend).
def write_json_atomic(path, payload):
    """Write JSON atomically (temp file + fsync + os.replace) through bo_protocol's log writer.

    A file held open by another program is retried and otherwise kept in memory until the
    next write or flush_pending_logs() at shutdown, so rewriting MetaRunState.json after an
    evaluation can never end a study. Returns True when the file on disk is up to date.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return write_log_text(path, json.dumps(payload, indent=1, ensure_ascii=False, default=str))


def is_non_dominated_mask(values):
    """Boolean mask of Pareto-optimal rows (maximization) with mobo.py's IsPareto semantics.

    Of several IDENTICAL objective vectors only the first is flagged, exactly as in mobo.py
    (which combines moocore keep_weakly=True with a first-duplicate mask; keep_weakly=False
    is the same thing in one call). Flagging every duplicate instead inflated Pareto counts
    -- and FinalDesignSelector's candidate pool -- in the MetaTAF condition whenever ratings
    tie (Likert scales), biasing a comparison against the BoTorch backend. Non-finite rows
    are never Pareto-optimal.
    """
    import moocore  # lazy: the module stays importable with numpy + pandas only

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mask = np.zeros(arr.shape[0], dtype=bool)
    finite_rows = np.all(np.isfinite(arr), axis=1)
    if np.any(finite_rows):
        mask[finite_rows] = np.asarray(
            moocore.is_nondominated(arr[finite_rows], maximise=True, keep_weakly=False),
            dtype=bool,
        )
    return mask

# -------------------- openbo import (lazy, fail-fast) --------------------
def _pip_command():
    """pip for the interpreter running this script: Unity runs the backends in its own private
    environment, so a bare `python -m pip` would install into a different Python."""
    exe = sys.executable or "python"
    return (f'"{exe}"' if " " in exe else exe) + " -m pip"


def _import_openbo():
    """Import the openbo pieces this backend needs, with an actionable error."""
    # BoTorch's acquisition functions try to JIT-compile a fused C++ kernel on first use.
    # torch serializes those builds with a lock file in a SHARED cache directory -- and a
    # backend process that gets killed mid-run (Unity stop, crash, task manager) leaves
    # the lock behind, after which every later run hangs forever waiting on it. Giving
    # each process its own extensions directory makes the attempt fail fast in isolation
    # (the compile is a ~3x-speedup nicety, not a requirement). Users who really compile
    # kernels can pre-set TORCH_EXTENSIONS_DIR themselves; we only fill it when unset.
    if "TORCH_EXTENSIONS_DIR" not in os.environ:
        os.environ["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(prefix="bo_torch_ext_")
    try:
        from openbo.optimizers.mobo_taf import MOTAFConfig, MOTAFSequentialOptimizer
    except ImportError as e:
        raise RuntimeError(
            "The Meta-TAF backend needs the 'openbo' package (M-Colley fork with the "
            "multi-objective optimizers), which is not installed for this Python. Install it with:\n"
            f"    {_pip_command()} install \"open-bo @ git+https://github.com/M-Colley/openbo@main\"\n"
            "or, from a local clone:\n"
            f"    {_pip_command()} install -e path/to/openbo\n"
            "See docs/meta-taf-student-guide.md for details."
        ) from e
    # openbo's 2026-08 TAF-R rework changed what the mode token "taf_r" COMPUTES
    # (objective-wise pairwise ranking agreement instead of Pareto-dominance agreement,
    # which moved to "taf_r_pareto") without renaming the token. On an older install this
    # backend would therefore run a different similarity measure than the one configured
    # and logged -- silent variant drift across participants of one study. The function
    # below exists exactly since that rework, so its absence identifies a stale install;
    # refuse to start rather than guess (openbo kept version 0.1.0, hence the
    # --force-reinstall: plain '--upgrade' considers the install already satisfied).
    try:
        import openbo.acquisition.taf_mo_ehvi as taf_mo_ehvi
        if not hasattr(taf_mo_ehvi, "compute_taf_r_ranking_weights"):
            raise ImportError("openbo.acquisition.taf_mo_ehvi has no compute_taf_r_ranking_weights")
    except ImportError as e:
        raise RuntimeError(
            "The installed 'openbo' predates the TAF-R rework (objective-wise pairwise "
            "ranking agreement; taf_r_pareto ablation mode) and would compute outdated "
            "source weights. Upgrade it for this Python with:\n"
            f"    {_pip_command()} install --force-reinstall --no-deps "
            "\"open-bo @ git+https://github.com/M-Colley/openbo@main\"\n"
            "See docs/meta-taf-student-guide.md for details."
        ) from e
    return MOTAFConfig, MOTAFSequentialOptimizer

# -------------------- source artifact validation --------------------
def resolve_meta_source_dir():
    """Resolve the configured source directory to an absolute path."""
    raw = str(META_SOURCE_DIR or "").strip() or "MetaSources"
    if os.path.isabs(raw):
        return raw
    root = os.environ.get("BO_META_ROOT") or os.getcwd()
    return os.path.join(root, raw)


def validate_and_stage_sources(source_dir, staging_dir):
    """Copy frame-compatible artifact pairs into ``staging_dir``.

    Returns ``(staged_names, rejected_reasons)``; ``rejected_reasons`` holds one
    human-readable line per skipped candidate, which meta_execute folds into the
    fail-fast error when metaRequireSources is set and nothing is loaded. Staged is not
    loaded: openbo still warns and skips artifacts it cannot use (unreadable trajectory,
    malformed hyperparameters, a front that never dominates the reference point), so
    meta_execute reconciles this list with the optimizer's (reconcile_loaded_sources).

    Every source artifact must carry the canonical frame it was generated from
    (parameter names + bounds, objective names + bounds + minimize flags). Artifacts are
    stored already normalized, so a frame mismatch is invisible to shape checks -- a prior
    study with the same d and M but different bounds, or a flipped minimize flag, would
    silently transfer a rescaled or exactly INVERTED response surface. Mismatches are
    therefore skipped loudly, field by field. Unframed artifacts are skipped too unless
    BO_META_ALLOW_UNFRAMED=1 (escape hatch for hand-built fixtures).

    The staged copies live inside the run folder, which doubles as an audit trail of
    exactly which population models influenced this run.

    Each pair is COPIED first and validated on the copy, so what was validated is exactly
    what openbo loads (validating the shared folder and copying it afterwards read each file
    twice: a sync client replacing it in between staged content nobody had checked). A
    candidate that cannot be copied or read -- a locked file or a cloud placeholder
    (OSError), bytes that are not UTF-8 JSON, JSON that is not an object, macOS AppleDouble
    '._<name>.json' files from exFAT/SMB copies -- is skipped with its reason like any other
    rejected source instead of aborting the start with a traceback. A gp_state stamped with
    its trajectory's SHA-256 (meta_train.py) must match the staged trajectory: a mismatch
    means a half-replaced pair (new data next to old hyperparameters).
    """
    gp_dir = os.path.join(source_dir, "gp_states")
    traj_dir = os.path.join(source_dir, "trajectories")
    staged_gp_dir = os.path.join(staging_dir, "gp_states")
    staged_traj_dir = os.path.join(staging_dir, "trajectories")
    os.makedirs(staged_gp_dir, exist_ok=True)
    os.makedirs(staged_traj_dir, exist_ok=True)

    if not os.path.isdir(gp_dir):
        reason = f"source directory has no gp_states folder: {gp_dir}"
        print(f"Meta-TAF: {reason}", flush=True)
        return [], [reason]

    allow_unframed = os.environ.get("BO_META_ALLOW_UNFRAMED", "0") == "1"
    kept = []
    rejected = []

    def reject(name, reason, detail_lines=()):
        print(f"Meta-TAF: source '{name}' skipped: {reason}", flush=True)
        for line in detail_lines:
            print(f"    - {line}", flush=True)
        rejected.append(f"'{name}': {reason}" + (f" ({'; '.join(detail_lines)})" if detail_lines else ""))
        for staged in (os.path.join(staged_gp_dir, f"{name}.json"),
                       os.path.join(staged_traj_dir, f"{name}.json")):
            try:
                os.remove(staged)
            except OSError:
                pass

    for fname in sorted(os.listdir(gp_dir)):
        if not fname.endswith(".json"):
            continue
        name = fname[:-len(".json")]
        if fname.startswith("._"):
            reject(name, "macOS AppleDouble metadata file (created when the folder was copied "
                         "via exFAT/FAT32 or a network share), not a source")
            continue
        if fname.startswith("."):
            reject(name, "hidden file, not a source")
            continue
        if name in WEIGHTS_LOG_COLUMNS:
            reject(name, f"the name is a column of MetaWeightsPerEvaluation.csv "
                         f"{list(WEIGHTS_LOG_COLUMNS)}; rename both files")
            continue
        traj_path = os.path.join(traj_dir, fname)
        if not os.path.exists(traj_path):
            reject(name, "no trajectory file")
            continue
        staged_gp = os.path.join(staged_gp_dir, fname)
        staged_traj = os.path.join(staged_traj_dir, fname)
        try:
            shutil.copyfile(os.path.join(gp_dir, fname), staged_gp)
            shutil.copyfile(traj_path, staged_traj)
        except OSError as e:
            reject(name, f"could not be copied ({e}); a locked file or a cloud placeholder "
                         "that is not downloaded?")
            continue
        try:
            with open(staged_gp, "r", encoding="utf-8") as f:
                gp_payload = json.load(f)
        except (OSError, ValueError) as e:  # ValueError covers JSONDecodeError and UnicodeDecodeError
            reject(name, f"gp_states unreadable ({type(e).__name__}: {e})")
            continue
        if not isinstance(gp_payload, dict):
            reject(name, f"gp_states is not a JSON object (got {type(gp_payload).__name__})")
            continue

        frame = gp_payload.get("frame")
        if frame is None:
            if not allow_unframed:
                reject(name, "no frame block (predates frame stamping?); regenerate it with "
                             "meta_train.py, or set BO_META_ALLOW_UNFRAMED=1 if you really "
                             "know the frames match")
                continue
        else:
            diffs = meta_fingerprint.frame_differences(FRAME, frame)
            if diffs:
                reject(name, "built for a different study frame", diffs)
                continue

        provenance = gp_payload.get("provenance")
        if provenance is not None and not isinstance(provenance, dict):
            reject(name, f"provenance is not a JSON object (got {type(provenance).__name__})")
            continue
        stamp = (provenance or {}).get("trajectory_sha256")
        if stamp is not None:
            try:
                actual = meta_fingerprint.artifact_sha256(staged_traj)
            except OSError as e:
                reject(name, f"staged trajectory unreadable ({e})")
                continue
            if actual != stamp:
                reject(name, "its trajectory is not the one its gp_state was built with (a "
                             "half-replaced artifact pair, e.g. a failed meta_train.py rebuild "
                             "or an interrupted sync); rebuild it with meta_train.py --force")
                continue
        kept.append(name)

    return kept, rejected


_OPENBO_SOURCE_WARNING = re.compile(r"MO TAF source '([^']+)'")


def reconcile_loaded_sources(staged, loaded, caught_warnings, staging_dir):
    """Drop staged sources that openbo did not load; return ``[(name, reason), ...]``.

    openbo's loader and optimizer constructor warn-and-skip sources they cannot use, and
    those warnings name the source ("MO TAF source '<name>': ..."), which supplies the
    reason. Dropped pairs are removed from MetaSourcesUsed/ so the audit trail holds exactly
    the population models the optimizer used. Warnings that concern no dropped source are
    re-emitted unchanged.
    """
    loaded_set = set(loaded)
    reasons = {}
    for w in caught_warnings:
        match = _OPENBO_SOURCE_WARNING.search(str(w.message))
        name = match.group(1) if match else None
        if name is not None and name not in loaded_set:
            reasons.setdefault(name, []).append(str(w.message))
        else:
            warnings.showwarning(w.message, w.category, w.filename, w.lineno)

    dropped = []
    for name in staged:
        if name in loaded_set:
            continue
        reason = " | ".join(reasons.get(name, [])) or "not loaded by openbo (no reason reported)"
        print(
            f"Meta-TAF: source '{name}' passed frame validation but openbo did not load it; "
            f"removing it from MetaSourcesUsed: {reason}",
            flush=True,
        )
        for sub in ("gp_states", "trajectories"):
            try:
                os.remove(os.path.join(staging_dir, sub, f"{name}.json"))
            except FileNotFoundError:
                pass
        dropped.append((name, reason))
    return dropped


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions():
    """Installed versions of the distributions that decide a run's numbers (None = absent)."""
    versions = {}
    for dist in RUN_STATE_DISTRIBUTIONS:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def openbo_install_record():
    """How openbo is installed (PEP 610 direct_url.json: VCS commit or editable path) plus
    SHA-256 hashes of the openbo modules this process actually imported."""
    record = {"direct_url": None, "module_sha256": {}}
    try:
        text = importlib.metadata.distribution("open-bo").read_text("direct_url.json")
        if text:
            record["direct_url"] = json.loads(text)
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
        pass
    for module_name in OPENBO_MODULES:
        module_file = getattr(sys.modules.get(module_name), "__file__", None)
        if module_file and os.path.isfile(module_file):
            record["module_sha256"][module_name] = _sha256_file(module_file)
    return record


def config_record(config):
    """The optimizer configuration exactly as passed (dataclasses.asdict on openbo's config)."""
    if config is None:
        return None
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return dict(vars(config))


def source_record(staging_dir, name):
    """Identity and provenance of one loaded source, read from its staged copy."""
    gp_path = os.path.join(staging_dir, "gp_states", f"{name}.json")
    traj_path = os.path.join(staging_dir, "trajectories", f"{name}.json")
    record = {"name": name, "gp_state_sha256": None, "trajectory_sha256": None,
              "frame_digest": None, "provenance": None}
    # Line-ending-insensitive (meta_fingerprint.artifact_sha256), like population.json's hashes.
    if os.path.isfile(gp_path):
        record["gp_state_sha256"] = meta_fingerprint.artifact_sha256(gp_path)
        try:
            with open(gp_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload.get("frame"), dict):
                record["frame_digest"] = meta_fingerprint.frame_digest(payload["frame"])
            record["provenance"] = payload.get("provenance")
        except (OSError, ValueError, AttributeError):
            pass
    if os.path.isfile(traj_path):
        record["trajectory_sha256"] = meta_fingerprint.artifact_sha256(traj_path)
    return record


def duplicate_trajectories(records):
    """Groups of loaded sources whose trajectories are byte-identical (same SHA-256).

    Usually the same run published under two names (e.g. by meta_train.py versions that
    numbered artifacts by command-line position): that participant then counts twice in
    every TAF weighting, which the frame check cannot notice.
    """
    by_hash = {}
    for record in records:
        if record.get("trajectory_sha256"):
            by_hash.setdefault(record["trajectory_sha256"], []).append(record["name"])
    groups = [sorted(names) for names in by_hash.values() if len(names) > 1]
    for group in groups:
        print(
            f"Meta-TAF: WARNING: sources {group} have identical trajectories (the same run "
            "under several names?); this run counts several times in the population. Delete "
            "all but one from the source folder (and rebuild its population.json).",
            flush=True,
        )
    return groups


def check_population_manifest(source_dir, records, rejected, dropped):
    """Compare the loaded sources with ``source_dir``/population.json.

    Returns ``(summary, problems)``: ``summary`` is None when there is no manifest (the
    previous behaviour: whatever loads is used); ``problems`` lists every difference. The
    manifest is meta_train.py's record of the frozen population -- names, the SHA-256 of
    both files of every pair (meta_fingerprint.artifact_sha256: a git checkout that converts
    line endings is not a change), and the frame digest -- so a source added, removed,
    replaced or no longer loadable since the population was frozen is caught before the first
    trial instead of silently giving later participants a different treatment.
    """
    path = os.path.join(source_dir, MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None, []
    summary = {"path": path, "sha256": None, "frame_digest": None, "sources": None}
    try:
        summary["sha256"] = _sha256_file(path)
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), list):
            raise ValueError("not an object with a 'sources' list")
        expected = {}
        for entry in manifest["sources"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError(f"malformed source entry {entry!r}")
            expected[entry["name"]] = entry
    except (OSError, ValueError) as e:
        return summary, [f"{MANIFEST_FILENAME} is unreadable ({type(e).__name__}: {e})"]

    summary["frame_digest"] = manifest.get("frame_digest")
    summary["sources"] = sorted(expected)
    problems = []
    live_digest = meta_fingerprint.frame_digest(FRAME)
    if manifest.get("frame_digest") != live_digest:
        problems.append(
            f"it was written for frame digest {manifest.get('frame_digest')!r}, this study's "
            f"frame is {live_digest!r}"
        )
    reasons = {}
    for line in rejected:
        match = re.match(r"'([^']+)': (.*)", line)
        if match:
            reasons[match.group(1)] = match.group(2)
    for name, reason in dropped:
        reasons[name] = f"not loaded by openbo ({reason})"
    loaded = {record["name"]: record for record in records}
    for name in sorted(set(expected) - set(loaded)):
        problems.append(f"'{name}' is listed but was not loaded: "
                        f"{reasons.get(name, 'not in the source folder')}")
    for name in sorted(set(loaded) - set(expected)):
        problems.append(f"'{name}' was loaded but is not listed")
    for name in sorted(set(expected) & set(loaded)):
        changed = [label for key, label in (("trajectory_sha256", "trajectory"),
                                            ("gp_state_sha256", "gp_state"))
                   if expected[name].get(key) != loaded[name].get(key)]
        if changed:
            problems.append(f"'{name}' differs from the listed {' and '.join(changed)} (SHA-256)")
    return summary, problems


def build_run_state(config, seed, source_dir, records, dropped, rejected, abort_reason=None,
                    duplicates=(), manifest=None, planned=(0, 0)):
    """Everything needed to say later exactly what produced this run (MetaRunState.json).

    ``progress`` and the finish fields are kept current by RunStateLog during the run.
    """
    sampling, optimization = planned
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "generator": "Assets/StreamingAssets/BOData/BayesianOptimization/meta_mobo_runtime.py",
        "created_utc": now,
        "updated_utc": now,
        "abort_reason": abort_reason,
        "finished": abort_reason is not None,
        "finish_reason": None if abort_reason is None else "startup_abort",
        "finish_detail": abort_reason,
        "progress": {
            "sampling_iterations": sampling,
            "optimization_iterations": optimization,
            "planned_evaluations": sampling + optimization,
            "evaluations_completed": 0,
            "last_iteration": None,
            "last_phase": None,
            "latest_hypervolume": None,
            "latest_weights": None,
        },
        "seed": seed,
        "user": {"userId": USER_ID, "conditionId": CONDITION_ID, "groupId": GROUP_ID},
        "init_config": INIT_CONFIG,
        "frame": FRAME,
        "frame_digest": meta_fingerprint.frame_digest(FRAME) if FRAME is not None else None,
        "hypervolume_reference_point": [REF_POINT_VALUE] * len(objective_names),
        "motaf_config": config_record(config),
        "library_versions": library_versions(),
        "openbo_install": openbo_install_record(),
        "python": sys.version,
        "platform": platform.platform(),
        "sources": {
            "source_dir": source_dir,
            "loaded": list(records),
            "dropped": [{"name": name, "reason": reason} for name, reason in dropped],
            "rejected": list(rejected),
            "duplicate_trajectories": [list(group) for group in duplicates],
            "population_manifest": manifest,
        },
    }


class RunStateLog:
    """MetaRunState.json, rewritten atomically after every evaluation.

    Like DboRunState.json it follows the run: evaluations completed, the latest hypervolume
    and source weights, and at the end whether and why the run finished ('completed',
    'stop_requested' with Unity's reason, 'error'). A file still saying "finished": false
    belongs to a process that was killed mid-run.
    """

    def __init__(self, path, state):
        self.path = path
        self.state = state

    def write(self):
        self.state["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return write_json_atomic(self.path, self.state)

    def record_evaluation(self, iteration, phase, hypervolume, weights=None):
        progress = self.state["progress"]
        progress["evaluations_completed"] += 1
        progress["last_iteration"] = int(iteration)
        progress["last_phase"] = phase
        progress["latest_hypervolume"] = float(hypervolume)
        if weights is not None:
            progress["latest_weights"] = weights
        self.write()

    def finish(self, reason, detail=None):
        self.state["finished"] = True
        self.state["finish_reason"] = reason
        self.state["finish_detail"] = detail
        self.write()


class StartupAbort(RuntimeError):
    """The run refuses to start; MetaRunState.json records ``str(exc)`` as abort_reason."""


def create_run_folder():
    """LogData/<user>/<condition>/run[_k] for this session (BO_LOG_ROOT overrides LogData)."""
    base = os.environ.get("BO_LOG_ROOT") or os.path.join(os.getcwd(), "LogData")
    condition_base = os.path.join(base, USER_LOG_ID, CONDITION_LOG_ID)
    os.makedirs(condition_base, exist_ok=True)
    return get_unique_folder(condition_base, "run")


def reject_at_startup(message, seed):
    """Refuse an init that passed parsing but must not run: record why, then raise ValueError.

    The abort record goes to a fresh run folder like every other aborted start, so the
    reason is found where the run's logs would have been.
    """
    global PROJECT_PATH
    try:
        PROJECT_PATH = create_run_folder()
        write_json_atomic(
            os.path.join(PROJECT_PATH, RUN_STATE_FILENAME),
            build_run_state(None, seed, resolve_meta_source_dir(), [], [], [],
                            abort_reason=message, planned=(N_INITIAL, N_ITERATIONS)),
        )
    except OSError as e:
        print(f"Warning: could not write {RUN_STATE_FILENAME} for the aborted start: {e}", flush=True)
    raise ValueError(message)


def negative_seed_message(seed):
    return (
        f"Seed must be >= 0 for the Meta-TAF backend, got {seed}. openbo seeds numpy's "
        "generator with it (np.random.default_rng), which rejects negative seeds. It is not "
        "mapped to another value, because that would also change the initial Sobol design, "
        "which must equal the BoTorch condition's for the same Seed. Set a non-negative "
        "Random Seed in the BoForUnityManager inspector (the same in every condition)."
    )


# -------------------- objective evaluation over the socket --------------------
def objective_function(conn, x_unit, iteration):
    """Send one design (unit cube) to Unity; return its normalized objective row.

    ``iteration`` is the Iteration the design is logged under in ObservationsPerEvaluation.csv;
    Unity receives it with the parameters.
    """
    x = np.asarray(x_unit, dtype=np.float64).reshape(-1)
    values = {}
    for i, name in enumerate(parameter_names):
        lo, hi = parameters_info[i]
        # Keep full precision for optimizer-proposed points sent to Unity.
        values[name] = bo_normalize.denormalize_to_original_param(x[i], lo, hi, decimals=None)

    payload = {"type": "parameters", "values": values, "iteration": int(iteration)}
    print("Send parameters:", payload, flush=True)
    send_json_line(conn, payload)

    resp = recv_objectives_blocking(conn)  # a dict, or None once Unity disconnected
    if resp is None:
        raise RuntimeError("No objectives received from Unity.")

    missing = [name for name in objective_names if name not in resp]
    if missing:
        raise KeyError(f"Unity objectives missing required key(s): {missing}")
    unexpected = sorted([k for k in resp.keys() if k not in set(objective_names)])
    if unexpected:
        raise KeyError(f"Unity objectives payload contains unexpected key(s): {unexpected}")

    fs = []
    for i, name in enumerate(objective_names):
        lo, hi, minflag = objectives_info[i]
        fs.append(bo_normalize.normalize_objective_value(resp[name], lo, hi, minflag, name=name))
    return np.asarray(fs, dtype=np.float64)

# -------------------- logging --------------------
def expected_observation_columns():
    return list(LOG_FIXED_COLUMNS) + objective_names + parameter_names


def append_observation_row(iteration, phase, y_norm_row, x_unit_row):
    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    if not log_exists(obs_csv):
        write_csv_rows(obs_csv, expected_observation_columns(), [])

    x_den = [
        bo_normalize.denormalize_to_original_param(x_unit_row[j], parameters_info[j][0], parameters_info[j][1])
        for j in range(PROBLEM_DIM)
    ]
    y_den = [
        bo_normalize.denormalize_to_original_obj(
            y_norm_row[j], objectives_info[j][0], objectives_info[j][1], objectives_info[j][2]
        )
        for j in range(NUM_OBJS)
    ]
    row = [USER_ID, CONDITION_ID, GROUP_ID,
           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
           iteration, phase, 'FALSE', *y_den, *x_den]
    append_csv_rows(obs_csv, [row])


def rewrite_pareto_flags(y_all_norm):
    """Recompute IsPareto over every logged row of this run (max sense).

    The log is read back as plain strings (dtype=str, keep_default_na=False) so only the
    IsPareto column changes: with pandas' type inference every rewrite turned ID tokens into
    numbers or NaN ("007" -> "7", "01" -> "1", "NA" -> ""), after which FinalDesignSelector's
    exact ID match found no rows for the participant.
    """
    obs_csv = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    if not log_exists(obs_csv):
        return
    flags = ['TRUE' if b else 'FALSE' for b in is_non_dominated_mask(y_all_norm).tolist()]
    df = read_observation_log(obs_csv)
    expected_cols = expected_observation_columns()
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"ObservationsPerEvaluation.csv columns mismatch. "
            f"Expected {expected_cols}, got {list(df.columns)}"
        )
    if len(df) != len(flags):
        raise ValueError(
            f"ObservationsPerEvaluation.csv row count {len(df)} does not match "
            f"observation count {len(flags)}"
        )
    df['IsPareto'] = flags
    write_dataframe_csv(obs_csv, df)


def settle_pareto_flags(logged_rows):
    """IsPareto over the rows logged so far, for a run that ends early; never raises.

    Sampling rows are logged with a provisional FALSE and flagged only after the last of them,
    so a stop during the sampling phase (Unity's perfect-rating stop can come in the initial
    rounds) or an error left every logged row FALSE, and FinalDesignSelector found no Pareto
    design for the participant. The run is ending for another reason, which a failure here
    must not hide.
    """
    if not logged_rows:
        return
    try:
        rewrite_pareto_flags(np.vstack(logged_rows))
    except Exception as e:
        print(f"Warning: could not update IsPareto in ObservationsPerEvaluation.csv: "
              f"{type(e).__name__}: {e}", flush=True)


def save_hypervolume_to_file(hvs, iteration, ref_point):
    append_csv_rows(
        os.path.join(PROJECT_PATH, "HypervolumePerEvaluation.csv"),
        [[
            hvs[-1],
            iteration,
            "normalized maximize-space [-1,1] per objective",
            "[" + ",".join(str(float(value)) for value in ref_point) + "]",
        ]],
        header=["Hypervolume", "Iteration", "Scale", "ReferencePoint"],
    )


def append_meta_weights_row(weights_csv, iteration, step, optimizer, source_names):
    """Log the weights openbo used for the suggestion evaluated at ``iteration``; return them.

    ``iteration`` is the global evaluation index, i.e. the Iteration of the same design's row
    in ObservationsPerEvaluation.csv and HypervolumePerEvaluation.csv (the file used to log
    the optimization step 1..N here, so joins on Iteration paired weights with the evaluation
    numSamplingIterations rows earlier); ``step`` is that optimization step.
    """
    # Direct attribute access on purpose (present in every openbo passing the TAF-R probe):
    # defaults here would silently log TargetWeight 1.0 / zero weights / no decay if openbo
    # ever renamed them, i.e. a weights file that contradicts what the optimizer did.
    target_w = float(optimizer.last_target_weight)
    weights = np.asarray(optimizer.last_source_weights, dtype=np.float64).reshape(-1)
    decay = float(optimizer._decay_factor())
    if weights.size not in (0, len(source_names)):
        raise ValueError(
            f"openbo reported {weights.size} source weights for {len(source_names)} loaded sources."
        )
    padded = [float(v) for v in weights] + [0.0] * (len(source_names) - weights.size)
    append_csv_rows(weights_csv, [[iteration, step, target_w, decay, *padded]])
    return {
        "iteration": int(iteration),
        "optimization_step": int(step),
        "target_weight": target_w,
        "decay_factor": decay,
        "source_weights": dict(zip(source_names, padded)),
    }


def latest_hypervolume(optimizer, n_evaluations):
    """openbo's hypervolume after the latest evaluation.

    MOBoTorchSequentialOptimizer.observe() appends compute_hypervolume(all observations so
    far, config.ref_point) once per observed row -- exactly what this backend used to compute
    a second time for its log -- so the runtime reads that value instead. One entry per
    evaluation is checked, so a change in openbo's bookkeeping fails loudly instead of
    logging another quantity.
    """
    history = optimizer.hypervolume_history
    if len(history) != n_evaluations:
        raise RuntimeError(
            f"openbo's hypervolume_history has {len(history)} entries after {n_evaluations} "
            "evaluations; this backend logs it as the per-evaluation hypervolume and needs one "
            "entry per evaluation."
        )
    return float(history[-1])

# -------------------- sampling --------------------
def draw_initial_unit_samples(n_samples, d, seed):
    """Scrambled Sobol design in [0,1]^d -- the exact call the BoTorch backends make.

    bo.py, mobo.py and the DBO backend draw their initial design with this same call, so for
    the same Unity seed every backend starts from identical points and a MetaTAF-vs-BoTorch
    comparison differs in the optimizer only, not in where it started. (scipy's Sobol, used
    before, gave a different design for the same seed: a systematic per-condition offset
    under the shared default seed.) n=n_samples, q=1 draws n points of the d-dimensional
    sequence; n=1, q=n_samples would draw ONE point of an (n*d)-dimensional sequence instead.
    """
    import torch
    from botorch.utils.sampling import draw_sobol_samples

    unit_bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
    x = draw_sobol_samples(bounds=unit_bounds, n=n_samples, q=1, seed=seed).squeeze(1)
    return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)

# -------------------- main loop --------------------
def meta_execute(conn, seed, iterations, initial_samples):
    global PROJECT_PATH
    MOTAFConfig, MOTAFSequentialOptimizer = _import_openbo()

    PROJECT_PATH = create_run_folder()
    run_state_path = os.path.join(PROJECT_PATH, RUN_STATE_FILENAME)
    exec_csv = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(exec_csv, ['Optimization', 'Execution_Time'])

    source_dir = resolve_meta_source_dir()
    staging_dir = os.path.join(PROJECT_PATH, "MetaSourcesUsed")
    ref_point = [REF_POINT_VALUE] * NUM_OBJS
    config = None
    records, dropped, rejected, duplicates, manifest = [], [], [], [], None

    # Startup, up to the first trial. Any failure here is recorded as the run's abort reason
    # in MetaRunState.json (not only the missing-sources abort) before it propagates.
    try:
        if seed < 0:  # main() refuses this at init; kept for direct callers
            raise StartupAbort(negative_seed_message(seed))
        if initial_samples < 1:
            raise ValueError("numSamplingIterations must be >= 1 for the Meta-TAF backend.")

        # Stage frame-validated sources into the run folder (also the audit trail).
        staged, rejected = validate_and_stage_sources(source_dir, staging_dir)

        config = MOTAFConfig(
            bounds=[(0.0, 1.0)] * PROBLEM_DIM,
            ref_point=ref_point,
            taf_run_dir=staging_dir,
            n_init=0,
            n_iter=iterations,  # informational only: the ask/tell loop below drives the iterations
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            mc_samples=MC_SAMPLES,
            seed=seed,
            rho=META_RHO,
            taf_weight_mode=META_WEIGHT_MODE,
            target_weight=META_TARGET_WEIGHT,
            source_only_warmup_iters=META_WARMUP_ITERS,
            decay_start_iter=META_DECAY_START_ITER,
            decay_rate=META_DECAY_RATE,
            # Pinned to openbo's current defaults (identical in every openbo that passes the
            # TAF-R probe in _import_openbo), so an upstream default change -- e.g. to the newer
            # source_reference_mode "target_incumbent" -- cannot silently alter a running study.
            source_reference_mode="front",
            source_reference_quantile=0.9,
            min_informative_pairs=1,
            source_meta_features=None,
            target_meta_features=None,
        )
        # openbo warns and SKIPS sources it cannot use (unreadable trajectory, malformed
        # hyperparameters, a front that never dominates the reference point). The staged list
        # is therefore only a candidate list: the fail-fast below, the reported count and the
        # MetaSourcesUsed audit trail must follow what the optimizer actually LOADED.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer = MOTAFSequentialOptimizer(config)
        source_names = [s.name for s in optimizer.source_surrogates]
        dropped = reconcile_loaded_sources(staged, source_names, caught, staging_dir)
        records = [source_record(staging_dir, name) for name in source_names]
        duplicates = duplicate_trajectories(records)

        manifest, manifest_problems = check_population_manifest(source_dir, records, rejected, dropped)
        if manifest_problems:
            raise StartupAbort(
                f"Meta-TAF: the population loaded from '{source_dir}' differs from its frozen "
                f"population manifest {MANIFEST_FILENAME} (written by meta_train.py):\n"
                + "\n".join(f"    - {problem}" for problem in manifest_problems)
                + "\nEvery participant of a MetaTAF condition must get the identical population "
                "(student guide, 'Freeze the population'). Restore the source folder to the "
                "frozen state, or -- only before the study starts -- rewrite the manifest with "
                "meta_train.py --frame frame.json --out <source folder> --manifest-only."
            )
        if source_names:
            print(f"Meta-TAF: using {len(source_names)} population model(s): {source_names}", flush=True)
        elif META_REQUIRE_SOURCES:
            reasons = list(rejected) + [f"'{name}': not loaded by openbo ({reason})" for name, reason in dropped]
            detail = ""
            if reasons:
                detail = "\nRejected candidate(s):\n" + "\n".join(f"    - {r}" for r in reasons)
            raise StartupAbort(
                f"Meta-TAF: no valid population model found under '{source_dir}' and this run "
                "requires sources (metaRequireSources). Without sources the run would silently "
                "degrade to plain qLogNEHVI, turning a MetaTAF study condition into the "
                "no-transfer control. Fix 'Meta Source Dir' or regenerate the sources with "
                "meta_train.py against the CURRENT study frame; only if a source-less run is "
                "genuinely intended, disable 'Meta Require Sources' in the BoForUnityManager "
                "Inspector." + detail
            )
        else:
            print(
                "Meta-TAF: no valid population models found; running plain multi-objective "
                "BO (qLogNEHVI) because 'Meta Require Sources' is disabled.",
                flush=True,
            )
    except Exception as e:
        reason = str(e) if isinstance(e, StartupAbort) else f"{type(e).__name__}: {e}"
        write_json_atomic(run_state_path, build_run_state(
            config, seed, source_dir, records, dropped, rejected, abort_reason=reason,
            duplicates=duplicates, manifest=manifest, planned=(initial_samples, iterations),
        ))
        raise

    state_log = RunStateLog(run_state_path, build_run_state(
        config, seed, source_dir, records, dropped, rejected,
        duplicates=duplicates, manifest=manifest, planned=(initial_samples, iterations),
    ))
    state_log.write()

    weights_csv = os.path.join(PROJECT_PATH, "MetaWeightsPerEvaluation.csv")
    write_csv_rows(weights_csv, [*WEIGHTS_LOG_COLUMNS, *source_names], [])

    logged = []  # objective rows in ObservationsPerEvaluation.csv (settle_pareto_flags)
    try:
        # ---- sampling phase (Sobol, evaluated one-by-one through Unity) ----
        x_init = draw_initial_unit_samples(initial_samples, PROBLEM_DIM, seed)
        print("Initial Sobol X in [0,1]:", x_init, flush=True)

        y_rows = []
        hvs = []
        for i in range(initial_samples):
            print(f"---- Initial Sample {i+1}", flush=True)
            y_row = objective_function(conn, x_init[i], i + 1)
            y_rows.append(y_row)
            append_observation_row(i + 1, 'sampling', y_row, x_init[i])
            logged.append(y_row)
            send_json_line(conn, {"type": "tempCoverage", "value": float(i + 1) / float(max(1, initial_samples))})
            # Observed one by one (the same final state as one batched observe), so openbo's
            # hypervolume after this evaluation is available for the log.
            optimizer.observe(x_init[i:i + 1], y_row.reshape(1, -1))
            hvs.append(latest_hypervolume(optimizer, i + 1))
            save_hypervolume_to_file(hvs, i + 1, ref_point)
            send_json_line(conn, {"type": "coverage", "value": float(hvs[-1])})
            state_log.record_evaluation(i + 1, "sampling", hvs[-1])

        x_all = np.asarray(x_init, dtype=np.float64)
        y_all = np.vstack(y_rows)
        rewrite_pareto_flags(y_all)

        # ---- optimization phase (ask/tell against openbo) ----
        for it in range(1, iterations + 1):
            iteration = initial_samples + it  # global Iteration, as in every other log
            t0 = time.time()
            x_next = optimizer.suggest()  # (1, d) in the unit cube
            t_elapsed = time.time() - t0
            write_data_to_csv(exec_csv, ['Optimization', 'Execution_Time'],
                              [{'Optimization': it, 'Execution_Time': t_elapsed}])
            weights = append_meta_weights_row(weights_csv, iteration, it, optimizer, source_names)

            y_row = objective_function(conn, x_next[0], iteration)
            optimizer.observe(x_next, y_row.reshape(1, -1))
            x_all = np.vstack([x_all, x_next])
            y_all = np.vstack([y_all, y_row.reshape(1, -1)])

            append_observation_row(iteration, 'optimization', y_row, x_next[0])
            logged.append(y_row)
            rewrite_pareto_flags(y_all)
            hvs.append(latest_hypervolume(optimizer, iteration))
            save_hypervolume_to_file(hvs, iteration, ref_point)
            send_json_line(conn, {"type": "coverage", "value": float(hvs[-1])})
            state_log.record_evaluation(iteration, "optimization", hvs[-1], weights)

        send_json_line(conn, {"type": "optimization_finished"})
    except StopRequested as e:
        settle_pareto_flags(logged)
        state_log.finish("stop_requested", str(e))
        raise
    except BaseException as e:
        if isinstance(e, Exception):
            settle_pareto_flags(logged)
        state_log.finish("error", f"{type(e).__name__}: {e}")
        raise
    state_log.finish("completed")
    return hvs, x_all, y_all

# -------------------- boot --------------------
def main():
    global N_INITIAL, N_ITERATIONS, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, SEED
    global PROBLEM_DIM, NUM_OBJS
    global META_SOURCE_DIR, META_REQUIRE_SOURCES, META_WEIGHT_MODE, META_RHO, META_TARGET_WEIGHT
    global META_WARMUP_ITERS, META_DECAY_START_ITER, META_DECAY_RATE
    global USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID
    global parameter_names, objective_names, parameters_info, objectives_info, FRAME, INIT_CONFIG

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    configure_listener_socket(s)
    conn = None
    try:
        conn = accept_unity_connection(s, SOCKET_ACCEPT_TIMEOUT_SEC)
        init_msg = receive_init_message(conn, SOCKET_TIMEOUT_SEC)

        cfg = init_msg.get("config", {}) or {}
        INIT_CONFIG = dict(cfg)

        backend = str(cfg.get("optimizerBackend") or "").strip().lower()
        if backend and backend != "meta-taf":
            raise ValueError(
                f"meta_mobo_runtime was launched for optimizerBackend='{backend}'; expected 'meta-taf'."
            )

        N_INITIAL      = get_cfg_int(cfg, "numSamplingIterations", default=N_INITIAL)
        N_ITERATIONS   = get_cfg_int(cfg, "numOptimizationIterations", default=N_ITERATIONS)
        BATCH_SIZE     = get_cfg_int(cfg, "batchSize", default=BATCH_SIZE)
        NUM_RESTARTS   = get_cfg_int(cfg, "numRestarts", default=NUM_RESTARTS)
        RAW_SAMPLES    = get_cfg_int(cfg, "rawSamples", default=RAW_SAMPLES)
        MC_SAMPLES     = get_cfg_int(cfg, "mcSamples", default=MC_SAMPLES)
        SEED           = get_cfg_int(cfg, "seed", default=SEED)
        PROBLEM_DIM    = get_cfg_int(cfg, "nParameters", required=True)
        NUM_OBJS       = get_cfg_int(cfg, "nObjectives", required=True)

        META_SOURCE_DIR       = str(cfg.get("metaSourceDir") or META_SOURCE_DIR)
        META_REQUIRE_SOURCES  = get_cfg_bool(cfg, "metaRequireSources", default=META_REQUIRE_SOURCES)
        META_WEIGHT_MODE      = str(cfg.get("metaWeightMode") or META_WEIGHT_MODE).strip().lower()
        META_RHO              = get_cfg_float(cfg, "metaRho", default=META_RHO)
        META_TARGET_WEIGHT    = get_cfg_float(cfg, "metaTargetWeight", default=META_TARGET_WEIGHT)
        META_WARMUP_ITERS     = get_cfg_int(cfg, "metaWarmupIters", default=META_WARMUP_ITERS)
        META_DECAY_START_ITER = get_cfg_int(cfg, "metaDecayStartIter", default=META_DECAY_START_ITER)
        META_DECAY_RATE       = get_cfg_float(cfg, "metaDecayRate", default=META_DECAY_RATE)

        if PROBLEM_DIM < 1:
            raise ValueError(f"nParameters must be >= 1, got {PROBLEM_DIM}")
        if NUM_OBJS < 2:
            raise ValueError(f"meta_mobo_runtime expects at least 2 objectives, got {NUM_OBJS}")
        if N_INITIAL < 1 or N_ITERATIONS < 0:
            raise ValueError(
                f"Iteration counts invalid: sampling={N_INITIAL} (must be >= 1), "
                f"optimization={N_ITERATIONS} (must be >= 0)"
            )
        if NUM_RESTARTS < 1 or RAW_SAMPLES < 1 or MC_SAMPLES < 1:
            raise ValueError(
                f"numRestarts/rawSamples/mcSamples must be >=1, got {NUM_RESTARTS}/{RAW_SAMPLES}/{MC_SAMPLES}"
            )
        # openbo optimizes the acquisition with BoTorch's optimize_acqf as well.
        validate_raw_samples(NUM_RESTARTS, RAW_SAMPLES)
        if BATCH_SIZE != 1:
            print(f"Warning: batchSize={BATCH_SIZE} is not supported in this HITL loop; forcing batchSize=1.", flush=True)
            BATCH_SIZE = 1
        if META_WEIGHT_MODE not in ("taf_m", "taf_r", "taf_r_pareto"):
            raise ValueError(
                f"metaWeightMode must be 'taf_m', 'taf_r', or 'taf_r_pareto', got '{META_WEIGHT_MODE}'"
            )
        if META_RHO <= 0:
            raise ValueError(f"metaRho must be > 0, got {META_RHO}")
        if META_TARGET_WEIGHT <= 0:
            raise ValueError(f"metaTargetWeight must be > 0, got {META_TARGET_WEIGHT}")
        if META_WARMUP_ITERS < 0 or META_DECAY_START_ITER < 0:
            raise ValueError("metaWarmupIters and metaDecayStartIter must be >= 0")
        if not (0.0 <= META_DECAY_RATE <= 1.0):
            raise ValueError(f"metaDecayRate must be in [0, 1], got {META_DECAY_RATE}")

        # Explicit scope guards: fail fast instead of silently ignoring configuration.
        if bool(cfg.get("warmStart", False)):
            raise ValueError(
                "The Meta-TAF backend does not support warm start: population models are "
                "its transfer mechanism. Disable Warm Start or use the BoTorch backend."
            )
        context_cfg = init_msg.get("context") or {}
        if isinstance(context_cfg, dict) and bool(context_cfg.get("enabled", False)):
            raise ValueError(
                "Contextual optimization (LCE-M GP) is only supported with the BoTorch "
                "backend, not with Meta-TAF."
            )

        USER_ID, CONDITION_ID, GROUP_ID, USER_LOG_ID, CONDITION_LOG_ID = parse_user_ids(init_msg)

        parameters = init_msg.get("parameters", []) or []
        objectives = init_msg.get("objectives", []) or []
        parameter_names = [p.get("key") for p in parameters]
        objective_names = [o.get("key") for o in objectives]

        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("Duplicate parameter keys detected in init message.")
        if len(set(objective_names)) != len(objective_names):
            raise ValueError("Duplicate objective keys detected in init message.")
        overlap = sorted(set(parameter_names).intersection(set(objective_names)))
        if overlap:
            raise ValueError(f"Parameter and objective keys must be distinct. Overlap: {overlap}")
        reserved = sorted(set(LOG_FIXED_COLUMNS).intersection(parameter_names + objective_names))
        if reserved:
            raise ValueError(
                f"Parameter/objective key(s) {reserved} collide with the fixed columns of "
                f"ObservationsPerEvaluation.csv {list(LOG_FIXED_COLUMNS)}; rename them "
                "(the duplicated column would abort the run after the sampling phase)."
            )
        if len(parameter_names) != PROBLEM_DIM:
            raise ValueError(f"parameter_names len {len(parameter_names)} != nParameters {PROBLEM_DIM}")
        if len(objective_names) != NUM_OBJS:
            raise ValueError(f"objective_names len {len(objective_names)} != nObjectives {NUM_OBJS}")

        parameters_info = [parse_param_init(p.get("init")) for p in parameters]
        objectives_info = [parse_obj_init(o.get("init")) for o in objectives]
        validate_parameter_bounds(parameter_names, parameters_info, "MetaTAF")
        validate_objective_bounds(objective_names, objectives_info)

        FRAME = meta_fingerprint.canonical_frame(
            parameter_names, parameters_info, objective_names, objectives_info
        )
        if SEED < 0:
            # Refused here, before any source is staged: openbo's np.random.default_rng(seed)
            # would otherwise crash the start after staging, without an abort record.
            reject_at_startup(negative_seed_message(SEED), SEED)

        print("Init OK:", dict(
            BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS, RAW_SAMPLES=RAW_SAMPLES,
            N_ITERATIONS=N_ITERATIONS, MC_SAMPLES=MC_SAMPLES,
            N_INITIAL=N_INITIAL, SEED=SEED, PROBLEM_DIM=PROBLEM_DIM, NUM_OBJS=NUM_OBJS,
            META_WEIGHT_MODE=META_WEIGHT_MODE, META_SOURCE_DIR=META_SOURCE_DIR,
            META_REQUIRE_SOURCES=META_REQUIRE_SOURCES,
            META_WARMUP_ITERS=META_WARMUP_ITERS,
            META_DECAY=(META_DECAY_START_ITER, META_DECAY_RATE),
            FRAME_DIGEST=meta_fingerprint.frame_digest(FRAME),
        ), flush=True)

        meta_execute(conn, SEED, N_ITERATIONS, N_INITIAL)
    except StopRequested:
        pass  # Unity ended the study on purpose; every completed evaluation is logged
    finally:
        close_connection(conn)
        s.close()
        flush_pending_logs()


if __name__ == "__main__":
    main()
