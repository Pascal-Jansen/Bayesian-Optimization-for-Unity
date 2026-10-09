"""Tier-1 tests for meta_mobo_runtime.py — the Meta-TAF (TAF-EHVI) Unity backend.

Runs in the numpy+pandas-only CI environment: torch/botorch/moocore come from
tests/_stubs.py and openbo comes from install_openbo_stub(). What is verified here is the
backend's OWN contract — NDJSON protocol, config validation, frame-based source staging,
and the CSV family — not the optimizer math (that lives in the openbo test suite).
"""

import contextlib
import csv
import dataclasses
import importlib
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
import uuid
from unittest import mock

import numpy as np
import pandas as pd

# Support both `discover tests` (tests/ on sys.path) and direct module runs.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import (  # noqa: E402
    FakeConn as _FakeConn,
    FakeServerSocket as _FakeServerSocket,
    FakeTensor as _FakeTensor,
    assert_hardened_listener,
    install_openbo_stub,
    install_stub_modules,
    json_line as _json_line,
    run_main_recording_listener,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BO_DIR = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization"
RUNTIME_PATH = BO_DIR / "meta_mobo_runtime.py"
FINGERPRINT_PATH = BO_DIR / "meta_fingerprint.py"
TRAIN_PATH = BO_DIR / "meta_train.py"


def _load(path, prefix):
    name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime():
    install_stub_modules()
    install_openbo_stub()
    return _load(RUNTIME_PATH, "meta_runtime_test")


def load_fingerprint():
    return _load(FINGERPRINT_PATH, "meta_fp_for_runtime_test")


def load_meta_train():
    # Module level needs numpy + pandas only; the torch/openbo stack is imported lazily.
    return _load(TRAIN_PATH, "meta_train_test")


PARAMS = [
    {"key": "p0", "init": {"low": 0.0, "high": 1.0}},
    {"key": "p1", "init": {"low": 2.0, "high": 6.0}},
]
OBJECTIVES = [
    {"key": "o0", "init": {"low": 0.0, "high": 10.0, "minimize": 1}},
    {"key": "o1", "init": {"low": 0.0, "high": 10.0, "minimize": 0}},
]


def base_init_message(**config_overrides):
    config = {
        "numSamplingIterations": 2,
        "numOptimizationIterations": 2,
        "batchSize": 1,
        "numRestarts": 3,
        "rawSamples": 16,
        "mcSamples": 8,
        "seed": 7,
        "nParameters": 2,
        "nObjectives": 2,
        "warmStart": False,
        "optimizerBackend": "meta-taf",
        "metaSourceDir": "MetaSources",
        "metaWeightMode": "taf_r",
        "metaRho": 1.0,
        "metaTargetWeight": 1.0,
        "metaWarmupIters": 0,
        "metaDecayStartIter": 2,
        "metaDecayRate": 0.3,
    }
    config.update(config_overrides)
    return {
        "type": "init",
        "config": config,
        "parameters": [dict(p) for p in PARAMS],
        "objectives": [dict(o) for o in OBJECTIVES],
        "user": {"userId": "u1", "conditionId": "c1", "groupId": "g1"},
    }


def make_frame(fp_module, flip_minimize=False):
    objectives_info = [(0.0, 10.0, 1), (0.0, 10.0, 0)]
    if flip_minimize:
        objectives_info = [(0.0, 10.0, 0), (0.0, 10.0, 0)]
    return fp_module.canonical_frame(
        ["p0", "p1"], [(0.0, 1.0), (2.0, 6.0)], ["o0", "o1"], objectives_info
    )


def write_source(meta_dir, name, frame=None, unframed=False, truncate_trajectory=False, front=None):
    """Write one artifact pair. ``truncate_trajectory`` mimics a half-synced cloud copy and
    ``front`` overrides the stored Pareto front: both pass the runtime's frame validation
    but are skipped by openbo's loader/constructor (warn-and-skip)."""
    gp_dir = meta_dir / "gp_states"
    traj_dir = meta_dir / "trajectories"
    gp_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    x = [[0.1, 0.2], [0.5, 0.6], [0.9, 0.4]]
    y = [[0.2, 0.1], [-0.3, 0.5], [0.4, -0.2]]
    trajectory = json.dumps({"x_values": x, "y_values": y, "pareto_front": y if front is None else front})
    if truncate_trajectory:
        trajectory = trajectory[: len(trajectory) // 2]
    (traj_dir / f"{name}.json").write_text(trajectory, encoding="utf-8")
    payload = {"gp_state": {"objectives": [
        {"kernel_type": "matern52", "lengthscale": [0.3, 0.3], "variance": 1.0, "noise": 1e-4},
        {"kernel_type": "matern52", "lengthscale": [0.3, 0.3], "variance": 1.0, "noise": 1e-4},
    ]}}
    if not unframed:
        payload["frame"] = frame
    (gp_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def sent_messages(conn):
    return [json.loads(line) for line in b"".join(conn.sent).decode("utf-8").splitlines()]


RESPONSES = [
    {"o0": 2.0, "o1": 7.0},
    {"o0": 4.0, "o1": 6.0},
    {"o0": 1.0, "o1": 8.0},
    {"o0": 3.0, "o1": 9.0},
]


def _import_real_moocore():
    """The installed moocore (None when absent, as in CI), leaving the stub in sys.modules."""
    saved = sys.modules.pop("moocore", None)
    try:
        return importlib.import_module("moocore")
    except ImportError:
        return None
    finally:
        if saved is not None:
            sys.modules["moocore"] = saved


class MetaRuntimeModuleTests(unittest.TestCase):
    def test_module_imports_without_openbo_installed(self):
        """The lazy-import contract: module load must not need openbo."""
        install_stub_modules()
        saved = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "openbo" or k.startswith("openbo.")}
        try:
            module = _load(RUNTIME_PATH, "meta_runtime_no_openbo")
            with self.assertRaises(RuntimeError) as ctx:
                module._import_openbo()
            self.assertIn("open-bo", str(ctx.exception))
        finally:
            sys.modules.update(saved)

    def test_outdated_openbo_fails_fast_with_upgrade_command(self):
        """An openbo predating the TAF-R rework must be refused: on such an install the
        mode token 'taf_r' computes Pareto-dominance agreement instead of the configured
        objective-wise ranking agreement — silent variant drift within one study."""
        install_stub_modules()
        install_openbo_stub()
        module = _load(RUNTIME_PATH, "meta_runtime_stale_openbo")
        taf_mo_ehvi = sys.modules["openbo.acquisition.taf_mo_ehvi"]
        saved_probe = taf_mo_ehvi.compute_taf_r_ranking_weights
        del taf_mo_ehvi.compute_taf_r_ranking_weights
        try:
            with self.assertRaises(RuntimeError) as ctx:
                module._import_openbo()
            msg = str(ctx.exception)
            self.assertIn("--force-reinstall", msg)  # pip '--upgrade' is a no-op at same version
            self.assertNotIn("not installed", msg)   # distinct from the missing-package error
        finally:
            taf_mo_ehvi.compute_taf_r_ranking_weights = saved_probe


class SourceStagingTests(unittest.TestCase):
    def setUp(self):
        self.runtime = load_runtime()
        self.fp = load_fingerprint()
        self.runtime.FRAME = make_frame(self.fp)

    def test_stages_only_frame_compatible_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            src = tmp / "MetaSources"
            write_source(src, "good_a", frame=make_frame(self.fp))
            write_source(src, "good_b", frame=make_frame(self.fp))
            write_source(src, "flipped", frame=make_frame(self.fp, flip_minimize=True))
            write_source(src, "unframed", unframed=True)
            staging = tmp / "staged"
            kept, rejected = self.runtime.validate_and_stage_sources(str(src), str(staging))
            self.assertEqual(kept, ["good_a", "good_b"])
            self.assertEqual(len(rejected), 2)
            self.assertTrue(any(r.startswith("'flipped'") for r in rejected), rejected)
            self.assertTrue(any(r.startswith("'unframed'") for r in rejected), rejected)
            staged = sorted(p.stem for p in (staging / "gp_states").glob("*.json"))
            self.assertEqual(staged, ["good_a", "good_b"])

    def test_unframed_sources_allowed_with_escape_hatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            src = tmp / "MetaSources"
            write_source(src, "unframed", unframed=True)
            with mock.patch.dict(os.environ, {"BO_META_ALLOW_UNFRAMED": "1"}):
                kept, rejected = self.runtime.validate_and_stage_sources(str(src), str(tmp / "s"))
            self.assertEqual(kept, ["unframed"])
            self.assertEqual(rejected, [])

    def test_missing_source_dir_yields_no_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            kept, rejected = self.runtime.validate_and_stage_sources(
                os.path.join(tmp, "nope"), os.path.join(tmp, "staged")
            )
            self.assertEqual(kept, [])
            self.assertEqual(len(rejected), 1)
            self.assertIn("gp_states", rejected[0])


class _RunMainMixin:
    """Drives runtime.main() over a scripted fake socket; self.last_conn keeps the sent bytes."""

    def _run_main(self, runtime, init_msg, objective_responses, env):
        chunks = [_json_line(init_msg)]
        chunks += [_json_line({"type": "objectives", "values": v}) for v in objective_responses]
        conn = _FakeConn(chunks)
        self.last_conn = conn  # inspectable even when main() raises
        fake_server = _FakeServerSocket(conn)
        original_socket_ctor = runtime.socket.socket
        try:
            runtime.socket.socket = lambda *args, **kwargs: fake_server
            with mock.patch.dict(os.environ, env):
                runtime.main()
        finally:
            runtime.socket.socket = original_socket_ctor
        return conn


class MetaRuntimeProtocolTests(_RunMainMixin, unittest.TestCase):
    def test_main_listens_on_loopback_only_and_stops_after_connect(self):
        runtime = load_runtime()
        server, listening = run_main_recording_listener(runtime, base_init_message(), "meta_execute")
        assert_hardened_listener(self, runtime.socket, server, listening)

    def test_full_protocol_run_with_sources(self):
        runtime = load_runtime()
        fp = load_fingerprint()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            src = tmp / "MetaSources"
            write_source(src, "srcA", frame=make_frame(fp))
            write_source(src, "srcB", frame=make_frame(fp))
            write_source(src, "wrong", frame=make_frame(fp, flip_minimize=True))
            log_root = tmp / "LogData"

            responses = [
                {"o0": 2.0, "o1": 7.0},
                {"o0": 4.0, "o1": 6.0},
                {"o0": 1.0, "o1": 8.0},
                {"o0": 3.0, "o1": 9.0},
            ]
            conn = self._run_main(
                runtime,
                base_init_message(),
                responses,
                {"BO_LOG_ROOT": str(log_root), "BO_META_ROOT": str(tmp)},
            )

            sent = sent_messages(conn)
            types = [m["type"] for m in sent]
            self.assertEqual(types.count("parameters"), 4)
            self.assertEqual(types.count("tempCoverage"), 2)
            self.assertEqual(types.count("coverage"), 4)  # after every evaluation
            self.assertEqual(types.count("optimization_finished"), 1)
            self.assertEqual(types[-1], "optimization_finished")

            # Parameters must be raw-unit values inside the configured bounds.
            first_params = next(m for m in sent if m["type"] == "parameters")["values"]
            self.assertEqual(set(first_params), {"p0", "p1"})
            self.assertGreaterEqual(first_params["p1"], 2.0)
            self.assertLessEqual(first_params["p1"], 6.0)

            run_dir = log_root / "u1" / "c1" / "run"
            self.assertTrue(run_dir.is_dir())

            # Staged audit trail holds exactly the frame-compatible sources.
            staged = sorted(p.stem for p in (run_dir / "MetaSourcesUsed" / "gp_states").glob("*.json"))
            self.assertEqual(staged, ["srcA", "srcB"])

            # ObservationsPerEvaluation.csv: schema, phases, raw units, Pareto flags.
            with open(run_dir / "ObservationsPerEvaluation.csv", newline="") as f:
                rows = list(csv.reader(f, delimiter=";"))
            header, data = rows[0], rows[1:]
            self.assertEqual(
                header,
                ["UserID", "ConditionID", "GroupID", "Timestamp", "Iteration", "Phase",
                 "IsPareto", "o0", "o1", "p0", "p1"],
            )
            self.assertEqual(len(data), 4)
            self.assertEqual([r[5] for r in data],
                             ["sampling", "sampling", "optimization", "optimization"])
            self.assertEqual([r[4] for r in data], ["1", "2", "3", "4"])
            for r in data:
                self.assertIn(r[6], ("TRUE", "FALSE"))
                self.assertGreaterEqual(float(r[10]), 2.0)  # p1 back in raw units
                self.assertLessEqual(float(r[10]), 6.0)
            # Raw objective values must round-trip (o0 responses were 2,4,1,3).
            self.assertAlmostEqual(float(data[0][7]), 2.0, places=3)
            self.assertAlmostEqual(float(data[3][7]), 3.0, places=3)

            with open(run_dir / "HypervolumePerEvaluation.csv", newline="") as f:
                hv_rows = list(csv.reader(f, delimiter=";"))
            self.assertEqual(hv_rows[0], ["Hypervolume", "Iteration", "Scale", "ReferencePoint"])
            self.assertEqual([r[1] for r in hv_rows[1:]], ["1", "2", "3", "4"])
            self.assertTrue(all(r[2] == "normalized maximize-space [-1,1] per objective" for r in hv_rows[1:]))
            self.assertTrue(all(r[3] == "[-1.0,-1.0]" for r in hv_rows[1:]))

            with open(run_dir / "ExecutionTimes.csv", newline="") as f:
                exec_rows = list(csv.reader(f, delimiter=";"))
            self.assertEqual(exec_rows[0], ["Optimization", "Execution_Time"])
            self.assertEqual(len(exec_rows[1:]), 2)

            with open(run_dir / "MetaWeightsPerEvaluation.csv", newline="") as f:
                w_rows = list(csv.reader(f, delimiter=";"))
            self.assertEqual(w_rows[0], ["Iteration", "TargetWeight", "DecayFactor", "srcA", "srcB"])
            self.assertEqual(len(w_rows[1:]), 2)
            for row in w_rows[1:]:
                self.assertAlmostEqual(float(row[3]) + float(row[4]), 1.0, places=6)

    def test_taf_r_pareto_ablation_mode_reaches_the_optimizer(self):
        """The dominance ablation mode must pass init validation and arrive verbatim in
        MOTAFConfig (openbo dispatches on the exact token)."""
        runtime = load_runtime()
        fp = load_fingerprint()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            write_source(tmp / "MetaSources", "srcA", frame=make_frame(fp))
            responses = [{"o0": 2.0, "o1": 7.0}, {"o0": 4.0, "o1": 6.0}, {"o0": 1.0, "o1": 8.0}]
            conn = self._run_main(
                runtime,
                base_init_message(numSamplingIterations=2, numOptimizationIterations=1,
                                  metaWeightMode="taf_r_pareto"),
                responses,
                {"BO_LOG_ROOT": str(tmp / "LogData"), "BO_META_ROOT": str(tmp)},
            )
            types = [m["type"] for m in sent_messages(conn)]
            self.assertEqual(types.count("optimization_finished"), 1)
            optimizer = sys.modules["openbo.optimizers.mobo_taf"].MOTAFSequentialOptimizer.instances[-1]
            self.assertEqual(optimizer.config.taf_weight_mode, "taf_r_pareto")

    def test_zero_sources_falls_back_to_plain_mobo_run(self):
        """Source-less runs stay possible, but only by explicit opt-out."""
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            responses = [{"o0": 2.0, "o1": 7.0}, {"o0": 4.0, "o1": 6.0}, {"o0": 1.0, "o1": 8.0}]
            conn = self._run_main(
                runtime,
                base_init_message(numSamplingIterations=2, numOptimizationIterations=1,
                                  metaRequireSources=False),
                responses,
                {"BO_LOG_ROOT": str(tmp / "LogData"), "BO_META_ROOT": str(tmp)},
            )
            types = [m["type"] for m in sent_messages(conn)]
            self.assertEqual(types.count("optimization_finished"), 1)
            weights_csv = tmp / "LogData" / "u1" / "c1" / "run" / "MetaWeightsPerEvaluation.csv"
            with open(weights_csv, newline="") as f:
                w_rows = list(csv.reader(f, delimiter=";"))
            self.assertEqual(w_rows[0], ["Iteration", "TargetWeight", "DecayFactor"])

    def test_zero_sources_fails_fast_by_default(self):
        """metaRequireSources defaults to true: a source-less MetaTAF run must abort
        instead of silently becoming the no-transfer control."""
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            with self.assertRaises(RuntimeError) as ctx:
                self._run_main(
                    runtime,
                    base_init_message(),  # no metaRequireSources field -> default true
                    [],
                    {"BO_LOG_ROOT": str(tmp / "LogData"), "BO_META_ROOT": str(tmp)},
                )
            self.assertIn("metaRequireSources", str(ctx.exception))

    def test_all_sources_rejected_fails_fast_with_reasons(self):
        """A stale MetaSources dir (all frame-rejected) must abort and say why."""
        runtime = load_runtime()
        fp = load_fingerprint()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            src = tmp / "MetaSources"
            write_source(src, "stale", frame=make_frame(fp, flip_minimize=True))
            with self.assertRaises(RuntimeError) as ctx:
                self._run_main(
                    runtime,
                    base_init_message(),
                    [],
                    {"BO_LOG_ROOT": str(tmp / "LogData"), "BO_META_ROOT": str(tmp)},
                )
            msg = str(ctx.exception)
            self.assertIn("'stale'", msg)
            self.assertIn("different study frame", msg)

    def _assert_init_rejected(self, init_msg, expected_snippet):
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((ValueError, RuntimeError)) as ctx:
                self._run_main(runtime, init_msg, [],
                               {"BO_LOG_ROOT": tmp, "BO_META_ROOT": tmp})
            self.assertIn(expected_snippet, str(ctx.exception))

    def test_warm_start_is_rejected(self):
        self._assert_init_rejected(base_init_message(warmStart=True), "warm start")

    def test_contextual_optimization_is_rejected(self):
        msg = base_init_message()
        msg["context"] = {"enabled": True, "currentContext": "a", "contexts": [{"key": "a"}]}
        self._assert_init_rejected(msg, "Contextual optimization")

    def test_single_objective_is_rejected(self):
        msg = base_init_message(nObjectives=1)
        msg["objectives"] = [dict(OBJECTIVES[0])]
        self._assert_init_rejected(msg, "at least 2 objectives")

    def test_wrong_backend_token_is_rejected(self):
        self._assert_init_rejected(base_init_message(optimizerBackend="cabop"), "meta-taf")

    def test_invalid_weight_mode_is_rejected(self):
        self._assert_init_rejected(base_init_message(metaWeightMode="bogus"), "metaWeightMode")

    def test_keys_colliding_with_log_columns_are_rejected_at_init(self):
        """A parameter named like a fixed log column ('Phase') used to abort the run with a
        CSV column mismatch AFTER the sampling phase; it must be refused before any trial."""
        msg = base_init_message()
        msg["parameters"][0] = {"key": "Phase", "init": {"low": 0.0, "high": 1.0}}
        self._assert_init_rejected(msg, "collide with the fixed columns")
        self.assertNotIn("parameters", [m["type"] for m in sent_messages(self.last_conn)])


class LoadedSourceTests(_RunMainMixin, unittest.TestCase):
    """Fail-fast, reported count and audit trail follow the sources openbo LOADED."""

    def _env(self, tmp):
        return {"BO_LOG_ROOT": str(tmp / "LogData"), "BO_META_ROOT": str(tmp)}

    def test_source_openbo_drops_is_reported_and_removed_from_audit_trail(self):
        runtime = load_runtime()
        fp = load_fingerprint()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            src = tmp / "MetaSources"
            write_source(src, "srcA", frame=make_frame(fp))
            write_source(src, "srcB", frame=make_frame(fp), truncate_trajectory=True)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self._run_main(runtime, base_init_message(numOptimizationIterations=1),
                               RESPONSES[:3], self._env(tmp))
            log = out.getvalue()
            self.assertIn("using 1 population model(s): ['srcA']", log)
            self.assertIn("'srcB' passed frame validation but openbo did not load it", log)

            run_dir = tmp / "LogData" / "u1" / "c1" / "run"
            used = run_dir / "MetaSourcesUsed"
            self.assertEqual(sorted(p.stem for p in (used / "gp_states").glob("*.json")), ["srcA"])
            self.assertEqual(sorted(p.stem for p in (used / "trajectories").glob("*.json")), ["srcA"])
            with open(run_dir / "MetaWeightsPerEvaluation.csv", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f, delimiter=";"))
            self.assertEqual(header, ["Iteration", "TargetWeight", "DecayFactor", "srcA"])

            state = json.loads((run_dir / "MetaRunState.json").read_text(encoding="utf-8"))
            self.assertIsNone(state["abort_reason"])
            self.assertEqual([s["name"] for s in state["sources"]["loaded"]], ["srcA"])
            self.assertEqual([d["name"] for d in state["sources"]["dropped"]], ["srcB"])
            self.assertIn("malformed artifacts", state["sources"]["dropped"][0]["reason"])

    def test_all_staged_sources_dropped_fails_fast_before_any_evaluation(self):
        """The silent-degrade scenario: every frame-valid source is unusable for openbo (a
        half-synced trajectory, a front never dominating the reference point). With
        metaRequireSources the run must abort before the first trial instead of quietly
        running as plain qLogNEHVI."""
        runtime = load_runtime()
        fp = load_fingerprint()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            src = tmp / "MetaSources"
            write_source(src, "srcA", frame=make_frame(fp), truncate_trajectory=True)
            write_source(src, "srcB", frame=make_frame(fp), front=[[-1.0, 0.5]])
            with self.assertRaises(RuntimeError) as ctx:
                self._run_main(runtime, base_init_message(), [], self._env(tmp))
            msg = str(ctx.exception)
            self.assertIn("metaRequireSources", msg)
            self.assertIn("'srcA': not loaded by openbo", msg)
            self.assertIn("'srcB': not loaded by openbo", msg)
            self.assertNotIn("parameters", [m["type"] for m in sent_messages(self.last_conn)])

            run_dir = tmp / "LogData" / "u1" / "c1" / "run"
            self.assertEqual(list((run_dir / "MetaSourcesUsed" / "gp_states").glob("*.json")), [])
            state = json.loads((run_dir / "MetaRunState.json").read_text(encoding="utf-8"))
            self.assertIn("metaRequireSources", state["abort_reason"])
            self.assertEqual(sorted(d["name"] for d in state["sources"]["dropped"]), ["srcA", "srcB"])

    def test_run_state_records_versions_config_frame_and_sources(self):
        runtime = load_runtime()
        fp = load_fingerprint()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            src = tmp / "MetaSources"
            write_source(src, "srcA", frame=make_frame(fp))
            write_source(src, "srcB", frame=make_frame(fp))
            write_source(src, "wrong", frame=make_frame(fp, flip_minimize=True))
            self._run_main(runtime, base_init_message(), RESPONSES, self._env(tmp))
            run_dir = tmp / "LogData" / "u1" / "c1" / "run"
            state = json.loads((run_dir / "MetaRunState.json").read_text(encoding="utf-8"))
            self.assertEqual([p.name for p in run_dir.iterdir() if p.name.startswith(".tmp_")], [])

        for key in ("created_utc", "seed", "init_config", "frame", "frame_digest", "motaf_config",
                    "library_versions", "openbo_install", "python", "platform", "sources"):
            self.assertIn(key, state)
        for dist in ("torch", "botorch", "gpytorch", "open-bo", "numpy"):
            self.assertIn(dist, state["library_versions"])
        self.assertIn("direct_url", state["openbo_install"])
        self.assertEqual(state["seed"], 7)
        self.assertEqual(state["init_config"]["metaWeightMode"], "taf_r")
        self.assertEqual(state["frame"], make_frame(fp))
        self.assertEqual(state["frame_digest"], fp.frame_digest(make_frame(fp)))
        cfg = state["motaf_config"]
        self.assertEqual(
            (cfg["source_reference_mode"], cfg["source_reference_quantile"], cfg["min_informative_pairs"]),
            ("front", 0.9, 1),
        )
        self.assertEqual((cfg["taf_weight_mode"], cfg["decay_start_iter"], cfg["decay_rate"], cfg["seed"]),
                         ("taf_r", 2, 0.3, 7))
        loaded = state["sources"]["loaded"]
        self.assertEqual([s["name"] for s in loaded], ["srcA", "srcB"])
        self.assertTrue(all(len(s["gp_state_sha256"]) == 64 and len(s["trajectory_sha256"]) == 64
                            for s in loaded))
        self.assertTrue(all(s["frame_digest"] == state["frame_digest"] for s in loaded))
        self.assertEqual(state["sources"]["dropped"], [])
        self.assertTrue(any(r.startswith("'wrong'") for r in state["sources"]["rejected"]))

    def test_pinned_config_fields_survive_an_openbo_default_change(self):
        """An upstream default change (say source_reference_mode -> 'target_incumbent') must
        not silently alter a study: the runtime passes those fields explicitly."""
        runtime = load_runtime()
        fp = load_fingerprint()
        mobo_taf = sys.modules["openbo.optimizers.mobo_taf"]

        @dataclasses.dataclass
        class DriftedConfig(mobo_taf.MOTAFConfig):
            source_reference_mode: str = "target_incumbent"
            source_reference_quantile: float = 0.5
            min_informative_pairs: int = 7
            source_meta_features: object = dataclasses.field(default_factory=lambda: {"srcA": [1.0]})
            target_meta_features: object = dataclasses.field(default_factory=lambda: [1.0])

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(mobo_taf, "MOTAFConfig", DriftedConfig):
            tmp = pathlib.Path(tmp)
            write_source(tmp / "MetaSources", "srcA", frame=make_frame(fp))
            self._run_main(runtime, base_init_message(numOptimizationIterations=1),
                           RESPONSES[:3], self._env(tmp))
        config = mobo_taf.MOTAFSequentialOptimizer.instances[-1].config
        self.assertIsInstance(config, DriftedConfig)
        self.assertEqual(config.source_reference_mode, "front")
        self.assertEqual(config.source_reference_quantile, 0.9)
        self.assertEqual(config.min_informative_pairs, 1)
        self.assertIsNone(config.source_meta_features)
        self.assertIsNone(config.target_meta_features)


class LoggingTests(_RunMainMixin, unittest.TestCase):
    """Initial design, IDs and non-ASCII keys as written to the run folder."""

    def test_initial_design_is_the_botorch_backends_sobol_call(self):
        """Same seed -> same initial design as bo.py/mobo.py: the design must come from
        draw_sobol_samples(bounds=unit, n=N, q=1, seed=SEED).squeeze(1)."""
        runtime = load_runtime()
        fp = load_fingerprint()
        calls = []
        design = np.array([[0.1, 0.2], [0.3, 0.4]])

        def fake_draw_sobol_samples(bounds, n, q, seed=None, **kwargs):
            calls.append({"bounds": np.asarray(bounds.arr), "n": n, "q": q, "seed": seed, **kwargs})
            return _FakeTensor(design[:n].reshape(n, q, -1))

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                sys.modules["botorch.utils.sampling"], "draw_sobol_samples", fake_draw_sobol_samples):
            tmp = pathlib.Path(tmp)
            write_source(tmp / "MetaSources", "srcA", frame=make_frame(fp))
            self._run_main(runtime, base_init_message(numOptimizationIterations=0), RESPONSES[:2],
                           {"BO_LOG_ROOT": str(tmp / "LogData"), "BO_META_ROOT": str(tmp)})
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["n"], calls[0]["q"], calls[0]["seed"]), (2, 1, 7))
        np.testing.assert_array_equal(calls[0]["bounds"], [[0.0, 0.0], [1.0, 1.0]])
        params = [m["values"] for m in sent_messages(self.last_conn) if m["type"] == "parameters"]
        self.assertEqual(len(params), 2)
        self.assertAlmostEqual(params[0]["p0"], 0.1)
        self.assertAlmostEqual(params[0]["p1"], 2.8)  # 2 + 0.2 * (6 - 2)
        self.assertAlmostEqual(params[1]["p0"], 0.3)
        self.assertAlmostEqual(params[1]["p1"], 3.6)

    def test_id_tokens_survive_the_pareto_rewrite(self):
        """IDs are logged verbatim: the per-iteration IsPareto rewrite must not re-type them
        ('007' -> '7', '01' -> '1', 'NA' -> ''), FinalDesignSelector matches them exactly."""
        runtime = load_runtime()
        fp = load_fingerprint()
        msg = base_init_message()
        msg["user"] = {"userId": "007", "conditionId": "01", "groupId": "NA"}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            write_source(tmp / "MetaSources", "srcA", frame=make_frame(fp))
            self._run_main(runtime, msg, RESPONSES,
                           {"BO_LOG_ROOT": str(tmp / "LogData"), "BO_META_ROOT": str(tmp)})
            obs = tmp / "LogData" / "007" / "01" / "run" / "ObservationsPerEvaluation.csv"
            with open(obs, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f, delimiter=";"))[1:]
        self.assertEqual(len(rows), 4)
        self.assertEqual({tuple(r[:3]) for r in rows}, {("007", "01", "NA")})

    def test_non_ascii_keys_round_trip_through_the_logs(self):
        """UTF-8 everywhere: 'Größe' / 'Übersicht' must survive the CSV write and the pandas
        read-back of the IsPareto rewrite (the cp1252 default on Windows aborted the run)."""
        runtime = load_runtime()
        msg = base_init_message(numOptimizationIterations=1, metaRequireSources=False)
        msg["parameters"][0]["key"] = "Größe"
        msg["objectives"][0]["key"] = "Übersicht"
        responses = [{"Übersicht": r["o0"], "o1": r["o1"]} for r in RESPONSES[:3]]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            self._run_main(runtime, msg, responses,
                           {"BO_LOG_ROOT": str(tmp / "LogData"), "BO_META_ROOT": str(tmp)})
            obs = tmp / "LogData" / "u1" / "c1" / "run" / "ObservationsPerEvaluation.csv"
            with open(obs, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f, delimiter=";"))
        self.assertEqual(rows[0][7:], ["Übersicht", "o1", "Größe", "p1"])
        self.assertEqual(len(rows) - 1, 3)


class SocketFramingTests(unittest.TestCase):
    def test_utf8_character_split_across_recv_chunks_is_preserved(self):
        runtime = load_runtime()
        line = (json.dumps({"type": "objectives", "values": {"Übersicht": 3.0, "Dauer": 2.0}},
                           ensure_ascii=False) + "\n").encode("utf-8")
        cut = line.index("Ü".encode("utf-8")) + 1  # inside the two-byte 'Ü'
        msg = runtime.recv_json_message(_FakeConn([line[:cut], line[cut:]]))
        self.assertEqual(set(msg["values"]), {"Übersicht", "Dauer"})


class ParetoFlagTests(unittest.TestCase):
    """IsPareto must mean the same as in mobo.py: of identical vectors only the first."""

    Y = np.array([[0.5, 0.2], [0.5, 0.2], [0.1, 0.6], [0.0, 0.0]])

    def test_moocore_is_asked_for_mobo_semantics(self):
        runtime = load_runtime()
        runtime.is_non_dominated_mask(self.Y)
        name, _, maximise, keep_weakly = sys.modules["moocore"].calls[-1]
        self.assertEqual((name, maximise, keep_weakly), ("is_nondominated", True, False))

    def test_only_the_first_duplicate_is_pareto_like_mobo(self):
        real = _import_real_moocore()
        if real is None:
            self.skipTest("moocore is not installed (CI stubs it)")
        runtime = load_runtime()
        rng = np.random.default_rng(0)
        with mock.patch.dict(sys.modules, {"moocore": real}):
            self.assertEqual(runtime.is_non_dominated_mask(self.Y).tolist(), [True, False, True, False])
            for _ in range(50):  # tie-heavy Likert-like data vs mobo.py's own formula
                y = rng.integers(0, 4, size=(12, 2)).astype(np.float64)
                first = np.zeros(len(y), dtype=bool)
                first[np.unique(y, axis=0, return_index=True)[1]] = True
                expected = real.is_nondominated(y, maximise=True, keep_weakly=True) & first
                np.testing.assert_array_equal(runtime.is_non_dominated_mask(y), expected)


class MetaTrainTests(unittest.TestCase):
    FRAME_ARGS = (["p0", "p1"], [(0.2, 0.8), (2.0, 6.0)], ["o0", "o1"], [(0.0, 10.0, 1), (0.0, 10.0, 0)])

    def setUp(self):
        self.mt = load_meta_train()

    @staticmethod
    def _df(p0_values):
        n = len(p0_values)
        return pd.DataFrame({"p0": p0_values, "p1": [4.0] * n, "o0": [5.0] * n, "o1": [5.0] * n})

    def test_parameter_values_outside_frame_bounds_are_rejected(self):
        """No silent 'already normalized' fallback: a pilot recorded on [0, 1] fed against a
        frame of [0.2, 0.8] must be refused, naming the column and the offending values."""
        with self.assertRaises(ValueError) as ctx:
            self.mt.extract_normalized_xy(self._df([0.05, 0.5, 0.95]), *self.FRAME_ARGS)
        msg = str(ctx.exception)
        for snippet in ("'p0'", "0.05", "0.95", "[0.2, 0.8]"):
            self.assertIn(snippet, msg)

    def test_parameter_values_within_frame_bounds_are_normalized(self):
        x, _ = self.mt.extract_normalized_xy(self._df([0.2, 0.5, 0.8]), *self.FRAME_ARGS)
        np.testing.assert_allclose(x[:, 0], [0.0, 0.5, 1.0])

    def test_non_finite_restart_is_never_selected(self):
        entry = {"variance": 1.0, "noise": 1e-3, "mean_constant": 0.0, "lengthscale": [0.3, 0.4]}
        self.assertTrue(self.mt.is_finite_restart(-1.5, entry))
        self.assertFalse(self.mt.is_finite_restart(float("nan"), entry))
        self.assertFalse(self.mt.is_finite_restart(-1.5, dict(entry, lengthscale=[0.3, float("nan")])))
        self.assertFalse(self.mt.is_finite_restart(-1.5, dict(entry, mean_constant=float("inf"))))

    # write_artifact only uses the stack's pareto_front and loader; fake both.
    @staticmethod
    def _stack(loader):
        return (None,) * 7 + (lambda y: np.asarray(y, dtype=np.float64), loader)

    def _write(self, out_dir, loader):
        fp = load_fingerprint()
        entry = {"kernel_type": "matern52", "lengthscale": [0.3, 0.3], "variance": 1.0, "noise": 1e-4,
                 "mean_constant": 0.0}
        return self.mt.write_artifact(
            str(out_dir), "src", np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.4]]),
            np.array([[0.2, 0.1], [-0.3, 0.5], [0.4, -0.2]]), [entry, entry], make_frame(fp),
            "run", "synthetic", "generated", self._stack(loader),
        )

    @staticmethod
    def _files(out_dir):
        return sorted(str(p.relative_to(out_dir)) for p in pathlib.Path(out_dir).rglob("*") if p.is_file())

    def test_failed_self_check_leaves_no_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                self._write(tmp, lambda directory, expected_m=None, expected_d=None: [])
            self.assertEqual(self._files(tmp), [])
            self.assertEqual(os.listdir(tmp), [])  # scratch folder removed as well

    def test_non_finite_replay_fails_the_self_check(self):
        class _NaNSource:
            name = "src"

            def posterior_mean(self, x):
                return np.full((len(x), 2), np.nan)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                self._write(tmp, lambda directory, expected_m=None, expected_d=None: [_NaNSource()])
            self.assertIn("non-finite", str(ctx.exception))
            self.assertEqual(self._files(tmp), [])

    def test_passing_self_check_publishes_the_pair_with_its_residual(self):
        checked_dirs = []

        def loader(directory, expected_m=None, expected_d=None):
            checked_dirs.append(os.path.abspath(directory))
            traj = json.loads((pathlib.Path(directory) / "trajectories" / "src.json").read_text(encoding="utf-8"))

            class _Source:
                name = "src"

                def posterior_mean(self, x):
                    return np.asarray(traj["y_values"], dtype=np.float64)

            return [_Source()]

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._write(tmp, loader), 0.0)
            self.assertEqual(self._files(tmp), [os.path.join("gp_states", "src.json"),
                                                os.path.join("trajectories", "src.json")])
            payload = json.loads((pathlib.Path(tmp) / "gp_states" / "src.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["provenance"]["fit_residual"], 0.0)
            self.assertNotEqual(checked_dirs[0], os.path.abspath(tmp))  # checked on the scratch copy


if __name__ == "__main__":
    unittest.main()
