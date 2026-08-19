"""Tier-1 tests for meta_mobo_runtime.py — the Meta-TAF (TAF-EHVI) Unity backend.

Runs in the numpy+pandas-only CI environment: torch/botorch/moocore come from
tests/_stubs.py and openbo comes from install_openbo_stub(). What is verified here is the
backend's OWN contract — NDJSON protocol, config validation, frame-based source staging,
and the CSV family — not the optimizer math (that lives in the openbo test suite).
"""

import csv
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
import uuid
from unittest import mock

import numpy as np

# Support both `discover tests` (tests/ on sys.path) and direct module runs.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import (  # noqa: E402
    FakeConn as _FakeConn,
    FakeServerSocket as _FakeServerSocket,
    install_openbo_stub,
    install_stub_modules,
    json_line as _json_line,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BO_DIR = REPO_ROOT / "Assets/StreamingAssets/BOData/BayesianOptimization"
RUNTIME_PATH = BO_DIR / "meta_mobo_runtime.py"
FINGERPRINT_PATH = BO_DIR / "meta_fingerprint.py"


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


def write_source(meta_dir, name, frame=None, unframed=False):
    gp_dir = meta_dir / "gp_states"
    traj_dir = meta_dir / "trajectories"
    gp_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    x = [[0.1, 0.2], [0.5, 0.6], [0.9, 0.4]]
    y = [[0.2, 0.1], [-0.3, 0.5], [0.4, -0.2]]
    (traj_dir / f"{name}.json").write_text(
        json.dumps({"x_values": x, "y_values": y, "pareto_front": y}), encoding="utf-8"
    )
    payload = {"gp_state": {"objectives": [
        {"kernel_type": "matern52", "lengthscale": [0.3, 0.3], "variance": 1.0, "noise": 1e-4},
        {"kernel_type": "matern52", "lengthscale": [0.3, 0.3], "variance": 1.0, "noise": 1e-4},
    ]}}
    if not unframed:
        payload["frame"] = frame
    (gp_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def sent_messages(conn):
    return [json.loads(line) for line in b"".join(conn.sent).decode("utf-8").splitlines()]


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


class MetaRuntimeProtocolTests(unittest.TestCase):
    def _run_main(self, runtime, init_msg, objective_responses, env):
        chunks = [_json_line(init_msg)]
        chunks += [_json_line({"type": "objectives", "values": v}) for v in objective_responses]
        conn = _FakeConn(chunks)
        fake_server = _FakeServerSocket(conn)
        original_socket_ctor = runtime.socket.socket
        try:
            runtime.socket.socket = lambda *args, **kwargs: fake_server
            with mock.patch.dict(os.environ, env):
                runtime.main()
        finally:
            runtime.socket.socket = original_socket_ctor
        return conn

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


if __name__ == "__main__":
    unittest.main()
