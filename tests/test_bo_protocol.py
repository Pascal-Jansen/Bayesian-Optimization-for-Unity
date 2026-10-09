"""Tests for bo_protocol.py, the plumbing all five backends share (stub suite, runs on CI).

Covers the NDJSON reader, the objectives wait (stop requests, malformed lines, non-dict
values), the receive limit, the atomic lock-tolerant log writer and the init validation
helpers; and, through stub harnesses for bo.py, mobo.py, dbo_runtime.py, cabop_runtime.py and
meta_mobo_runtime.py, that every 'parameters' message carries the Iteration its design is
logged under, that a stop request ends each backend cleanly, and that the shared init checks
run before the first design is sent.
"""

import contextlib
import csv
import importlib.util
import io
import itertools
import json
import os
import pathlib
import sys
import tempfile
import types
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
    BACKEND_DIR,
    FakeConn,
    FakeServerSocket,
    FakeTensor,
    install_openbo_stub,
    install_stub_modules,
    json_line,
    protocol_module,
    reset_protocol_state,
)


def load_protocol_copy(env):
    """A fresh bo_protocol module, so module-level settings are read from ``env``."""
    name = f"bo_protocol_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BACKEND_DIR / "bo_protocol.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, env):
        for key in ("BO_MAX_RECV_BUF_BYTES", "BO_SOCKET_TIMEOUT_SEC", "BO_ACCEPT_TIMEOUT_SEC"):
            if key not in env:
                os.environ.pop(key, None)
        spec.loader.exec_module(module)
    return module


class FakeClock:
    """Stands in for bo_protocol's ``time``: sleeping advances the clock instantly."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class _ProtocolTestCase(unittest.TestCase):
    def setUp(self):
        self.p = reset_protocol_state()
        self.addCleanup(self.p.discard_unsaved_logs)
        self.addCleanup(self.p.reset_receive_state)

    def quietly(self):
        """Capture stdout; the returned buffer holds what the code printed."""
        buf = io.StringIO()
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(buf))
        self.addCleanup(stack.close)
        return buf


# -------------------- NDJSON reader --------------------
class ReceiveTests(_ProtocolTestCase):
    def test_several_messages_in_one_chunk(self):
        conn = FakeConn([b'{"type":"a"}\n{"type":"b"}\n'])
        self.assertEqual(self.p.recv_json_message(conn)["type"], "a")
        self.assertEqual(self.p.recv_json_message(conn)["type"], "b")

    def test_utf8_character_split_across_reads_is_reassembled(self):
        payload = '{"type":"objectives","values":{"Größe":1.0,"Preis €":2.0}}\n'.encode("utf-8")
        for char in ("ö", "€"):
            start = payload.index(char.encode("utf-8"))
            for cut in range(start + 1, start + len(char.encode("utf-8"))):
                self.p.reset_receive_state()
                msg = self.p.recv_json_message(FakeConn([payload[:cut], payload[cut:]]))
                self.assertEqual(list(msg["values"]), ["Größe", "Preis €"], (char, cut))

    def test_crlf_endings_and_blank_lines(self):
        conn = FakeConn([b'\r\n\n{"type":"a"}\r\n  \n{"type":"b"}\n'])
        self.assertEqual([m["type"] for m in self.p.ndjson_reader(conn)], ["a", "b"])

    def test_malformed_line_is_skipped_outside_the_objectives_wait(self):
        out = self.quietly()
        msg = self.p.recv_json_message(FakeConn([b'{"type":bad}\n{"type":"ok"}\n']))
        self.assertEqual(msg["type"], "ok")
        self.assertIn("Warning: skipping malformed JSON line from Unity", out.getvalue())

    def test_unterminated_tail_is_discarded_when_unity_closes(self):
        out = self.quietly()
        conn = FakeConn([b'{"type":"init"}\n{"type":"partial"', b""])
        self.assertEqual(self.p.recv_json_message(conn)["type"], "init")
        self.assertEqual(self.p.pending_receive_buffer(), '{"type":"partial"')
        self.assertIsNone(self.p.recv_json_message(conn))
        self.assertEqual(self.p.pending_receive_buffer(), "")
        self.assertIn('Warning: discarding trailing unterminated socket data: {"type":"partial"', out.getvalue())

    def test_pending_buffer_keeps_unread_lines_across_calls(self):
        conn = FakeConn([b'{"type":"a"}\n{"type":"b"}\n{"ty'])
        self.p.recv_json_message(conn)
        self.assertEqual(self.p.pending_receive_buffer(), '{"type":"b"}\n{"ty')
        self.p.reset_receive_state()
        self.assertEqual(self.p.pending_receive_buffer(), "")

    def test_timeout_names_the_limit(self):
        conn = FakeConn([self.p.socket.timeout("timeout")])
        with mock.patch.object(self.p, "SOCKET_TIMEOUT_SEC", 7):
            with self.assertRaisesRegex(TimeoutError, "after 7 seconds"):
                self.p.recv_json_message(conn)

    def test_init_wait_skips_noise_and_arms_the_timeout(self):
        conn = FakeConn([b'["noise"]\n' + json_line({"type": "log"}) + json_line({"type": "init", "n": 1})])
        msg = self.p.receive_init_message(conn, 42.0)
        self.assertEqual((msg["type"], msg["n"], conn.timeout), ("init", 1, 42.0))
        with self.assertRaisesRegex(RuntimeError, "Did not receive init message"):
            self.p.receive_init_message(FakeConn([]), 42.0)
        with self.assertRaisesRegex(ValueError, "BO_SOCKET_TIMEOUT_SEC"):
            self.p.receive_init_message(FakeConn([]), 0)

    def test_stop_in_place_of_the_init_raises_stop_requested(self):
        out = self.quietly()
        conn = FakeConn([json_line({"type": "stop", "reason": "invalid_configuration"})])
        with self.assertRaises(self.p.StopRequested):
            self.p.receive_init_message(conn, 5.0)
        self.assertIn("Stop requested by Unity: invalid_configuration", out.getvalue())


# -------------------- objectives wait --------------------
class ObjectivesWaitTests(_ProtocolTestCase):
    def test_other_types_and_non_dict_lines_are_skipped(self):
        conn = FakeConn([
            json_line({"type": "coverage", "value": 1.0}),
            b'["not-a-dict"]\n',
            json_line({"type": "heartbeat"}),  # unknown types stay ignored
            json_line({"type": "objectives", "values": {"o0": 1.25}}),
        ])
        self.assertEqual(self.p.recv_objectives_blocking(conn), {"o0": 1.25})

    def test_malformed_line_raises_at_once_instead_of_waiting_for_the_timeout(self):
        conn = FakeConn([
            b'{"type":"objectives","values":{"o0":1.0}\n',  # truncated reply
            json_line({"type": "objectives", "values": {"o0": 2.0}}),
        ])
        with self.assertRaisesRegex(RuntimeError, "malformed JSON line from Unity while waiting for objectives"):
            self.p.recv_objectives_blocking(conn)
        self.assertEqual(len(conn._chunks), 1, "the backend must not block for further data")

    def test_stop_request_raises_stop_requested(self):
        out = self.quietly()
        conn = FakeConn([json_line({"type": "stop", "reason": "perfect rating"}),
                         json_line({"type": "objectives", "values": {"o0": 2.0}})])
        with self.assertRaises(self.p.StopRequested) as ctx:
            self.p.recv_objectives_blocking(conn)
        self.assertEqual(str(ctx.exception), "perfect rating")
        self.assertIn("Stop requested by Unity: perfect rating", out.getvalue())

    def test_stop_request_without_reason(self):
        out = self.quietly()
        with self.assertRaises(self.p.StopRequested):
            self.p.recv_objectives_blocking(FakeConn([json_line({"type": "stop"})]))
        self.assertIn("Stop requested by Unity: no reason given", out.getvalue())

    def test_values_must_be_a_dict(self):
        for msg in ({"type": "objectives", "values": [1.0]}, {"type": "objectives"}):
            self.p.reset_receive_state()
            with self.assertRaisesRegex(RuntimeError, "non-dict 'values'"):
                self.p.recv_objectives_blocking(FakeConn([json_line(msg)]))

    def test_closed_connection_returns_none(self):
        self.assertIsNone(self.p.recv_objectives_blocking(FakeConn([])))


# -------------------- receive limit --------------------
class ReceiveLimitTests(_ProtocolTestCase):
    def test_default_limit_is_64_mib_and_the_environment_overrides_it(self):
        self.assertEqual(load_protocol_copy({}).SOCKET_MAX_RECV_BUF_BYTES, 64 * 1024 * 1024)
        self.assertEqual(load_protocol_copy({"BO_MAX_RECV_BUF_BYTES": "1234"}).SOCKET_MAX_RECV_BUF_BYTES, 1234)

    def test_overflow_names_the_environment_variable_and_resets(self):
        with mock.patch.object(self.p, "SOCKET_MAX_RECV_BUF_BYTES", 8):
            with self.assertRaises(RuntimeError) as ctx:
                self.p.recv_json_message(FakeConn([b"123456789"]))
        self.assertIn("exceeded 8 bytes", str(ctx.exception))
        self.assertIn("BO_MAX_RECV_BUF_BYTES", str(ctx.exception))
        self.assertEqual(self.p.pending_receive_buffer(), "")

    def test_limit_applies_to_one_line_not_to_a_chunk_of_complete_lines(self):
        chunk = b'{"t":"aaa"}\n' * 4  # 48 bytes, every line 12 bytes long
        with mock.patch.object(self.p, "SOCKET_MAX_RECV_BUF_BYTES", 16):
            msgs = list(self.p.ndjson_reader(FakeConn([chunk])))
        self.assertEqual(len(msgs), 4)

    def test_limit_counts_bytes_of_the_unterminated_line(self):
        with mock.patch.object(self.p, "SOCKET_MAX_RECV_BUF_BYTES", 10):
            with self.assertRaisesRegex(RuntimeError, "exceeded 10 bytes"):
                self.p.recv_json_message(FakeConn([b'{"k":"', "ääää".encode("utf-8"), b'"}\n']))

    def test_init_with_large_context_embeddings_is_received(self):
        # 300 manual 1280-d embeddings: ~4 MB, four times the former 1 MiB limit.
        rng = np.random.default_rng(0)
        contexts = [{"key": f"c{i}", "embedding": rng.standard_normal(1280).tolist()} for i in range(300)]
        data = json_line({"type": "init", "context": {"enabled": True, "contexts": contexts}})
        self.assertGreater(len(data), 4 * 1024 * 1024)
        chunks = [data[i:i + 65536] for i in range(0, len(data), 65536)]
        msg = self.p.receive_init_message(FakeConn(chunks), 10.0)
        self.assertEqual(len(msg["context"]["contexts"]), 300)


# -------------------- log writer --------------------
class LogWriterTests(_ProtocolTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(self.dir, "ObservationsPerEvaluation.csv")
        self.copy = os.path.join(self.dir, "ObservationsPerEvaluation.unsaved.csv")
        self.clock = FakeClock()
        patcher = mock.patch.object(self.p, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def read(self, path=None):
        with open(path or self.path, newline="", encoding="utf-8") as f:
            return f.read()

    def locked_replace(self, locked_paths, failures=None):
        """Like a log open in Excel: ``locked_paths`` can be neither replaced nor written in place
        (for ``failures`` write attempts, or forever)."""
        real_replace, real_overwrite = os.replace, self.p._overwrite_file
        state = {"left": failures}

        def is_locked(path):
            return os.path.abspath(path) in locked_paths and (state["left"] is None or state["left"] > 0)

        def replace(src, dst):
            if is_locked(dst):
                raise PermissionError(13, "The process cannot access the file", dst)
            return real_replace(src, dst)

        def overwrite(path, text):  # the last step of every write attempt
            if is_locked(path):
                if state["left"] is not None:
                    state["left"] -= 1
                raise PermissionError(13, "The process cannot access the file", path)
            return real_overwrite(path, text)

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(self.p.os, "replace", replace))
        stack.enter_context(mock.patch.object(self.p, "_overwrite_file", overwrite))
        return stack

    def test_rewrite_replaces_the_file_and_leaves_no_temp_file(self):
        self.assertTrue(self.p.write_log_text(self.path, "a;b\r\n1;2\r\n"))
        self.assertTrue(self.p.write_log_text(self.path, "a;b\r\n3;4\r\n"))
        self.assertEqual(self.read(), "a;b\r\n3;4\r\n")
        self.assertEqual(os.listdir(self.dir), ["ObservationsPerEvaluation.csv"])

    def test_a_failure_during_the_rewrite_keeps_the_previous_file_intact(self):
        out = self.quietly()
        self.p.write_log_text(self.path, "old\n")
        with mock.patch.object(self.p.os, "fsync", side_effect=OSError(5, "I/O error")):
            self.assertFalse(self.p.write_log_text(self.path, "new\n"))
        self.assertEqual(self.read(), "old\n")
        self.assertEqual(os.listdir(self.dir), ["ObservationsPerEvaluation.csv"])
        self.assertEqual(self.p.read_log_text(self.path), "new\n")  # kept in memory
        self.assertIn("Warning: could not write", out.getvalue())

    def test_a_log_that_only_refuses_the_replace_is_rewritten_in_place(self):
        # Windows: a viewer or a script holding the log open without delete sharing blocks
        # os.replace but not writing; the rewrite before 1.8.1 (in place) worked there.
        out = self.quietly()
        self.p.write_log_text(self.path, "h\n1\n")
        refuse = PermissionError(13, "The process cannot access the file", self.path)
        with mock.patch.object(self.p.os, "replace", side_effect=refuse):
            self.assertTrue(self.p.write_log_text(self.path, "h\n1\n2\n"))
        self.assertEqual(self.read(), "h\n1\n2\n")
        self.assertEqual(os.listdir(self.dir), ["ObservationsPerEvaluation.csv"])
        self.assertEqual((self.p.unsaved_logs(), self.clock.sleeps, out.getvalue()), ([], [], ""))

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_a_rewritten_log_gets_the_permissions_of_any_new_file(self):
        reference = os.path.join(self.dir, "reference.csv")
        with open(reference, "w", encoding="utf-8"):
            pass
        self.p.write_log_text(self.path, "a\n")
        self.p.write_log_text(self.path, "b\n")
        # Was 0o600 (tempfile.mkstemp): other accounts could no longer read the study logs.
        self.assertEqual(os.stat(self.path).st_mode & 0o777, os.stat(reference).st_mode & 0o777)

    def test_a_brief_lock_is_retried_until_released(self):
        out = self.quietly()
        with self.locked_replace({self.path}, failures=2):
            self.assertTrue(self.p.write_log_text(self.path, "x\n"))
        self.assertEqual(self.read(), "x\n")
        self.assertEqual(len(self.clock.sleeps), 2)
        self.assertEqual(out.getvalue(), "")

    def test_a_lasting_lock_is_reported_once_and_the_next_write_catches_up(self):
        out = self.quietly()
        self.p.write_log_text(self.path, "h\n1\n")
        with self.locked_replace({self.path}):
            self.assertFalse(self.p.write_log_text(self.path, "h\n1\n2\n"))
            waited = sum(self.clock.sleeps)
            self.assertGreater(waited, 1.0)
            self.assertLessEqual(waited, self.p.LOG_WRITE_RETRY_SEC)
            self.assertFalse(self.p.write_log_text(self.path, "h\n1\n2\n3\n"))
            self.assertEqual(sum(self.clock.sleeps), waited, "a known lock must not stall every write")
            self.assertEqual(self.read(), "h\n1\n")  # the locked file is untouched
            self.assertEqual(self.p.read_log_text(self.path), "h\n1\n2\n3\n")
            self.assertEqual(self.p.unsaved_logs(), [os.path.abspath(self.path)])
            # Unity kills the backend of an aborted session: the rows must not be in memory only.
            self.assertEqual(self.read(self.copy), "h\n1\n2\n3\n")
        self.assertEqual(out.getvalue().count("Warning: could not write"), 1)
        self.assertIn(f"kept in memory and as {self.copy}", out.getvalue())
        self.assertTrue(self.p.write_log_text(self.path, "h\n1\n2\n3\n4\n"))
        self.assertEqual(self.read(), "h\n1\n2\n3\n4\n")
        self.assertEqual(self.p.unsaved_logs(), [])
        self.assertIn("again; the file is complete", out.getvalue())
        self.assertEqual(os.listdir(self.dir), ["ObservationsPerEvaluation.csv"])  # the stale copy is gone

    def test_appends_while_locked_are_kept_and_written_later(self):
        self.quietly()
        metric = os.path.join(self.dir, "BestObjectivePerEvaluation.csv")
        with mock.patch.object(self.p, "_append_file", side_effect=PermissionError(13, "locked")):
            self.assertFalse(self.p.append_csv_rows(metric, [[0.1, 1]], header=["BestObjective", "Iteration"]))
            self.assertFalse(self.p.append_csv_rows(metric, [[0.2, 2]], header=["BestObjective", "Iteration"]))
            self.assertTrue(self.p.log_exists(metric))
            self.assertFalse(os.path.exists(metric))
            copy = os.path.join(self.dir, "BestObjectivePerEvaluation.unsaved.csv")
            self.assertEqual(self.read(copy), "BestObjective;Iteration\r\n0.1;1\r\n0.2;2\r\n")
        self.p.append_csv_rows(metric, [[0.3, 3]], header=["BestObjective", "Iteration"])
        with open(metric, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter=";"))
        self.assertEqual(rows, [["BestObjective", "Iteration"], ["0.1", "1"], ["0.2", "2"], ["0.3", "3"]])
        self.assertFalse(os.path.exists(copy))

    def test_read_modify_write_sees_rows_that_are_not_on_disk_yet(self):
        self.quietly()
        self.p.write_csv_rows(self.path, ["UserID", "IsBest", "o0"], [["007", "TRUE", 0.5]])
        df = self.p.read_observation_log(self.path)
        df = pd.concat([df, pd.DataFrame([["007", "FALSE", "0.2"]], columns=df.columns)], ignore_index=True)
        with self.locked_replace({self.path}):
            self.p.write_dataframe_csv(self.path, df)
            again = self.p.read_observation_log(self.path)
        self.assertEqual(again["UserID"].tolist(), ["007", "007"])  # IDs stay text
        self.assertEqual(len(self.p.read_observation_log(self.path)), 2)

    def test_flush_saves_a_still_locked_log_next_to_it(self):
        out = self.quietly()
        self.p.write_log_text(self.path, "h\n1\n")
        with self.locked_replace({self.path}):
            self.p.write_log_text(self.path, "h\n1\n2\n")
            self.assertEqual(self.p.flush_pending_logs(), [])
        fallback = os.path.join(self.dir, "ObservationsPerEvaluation.unsaved.csv")
        self.assertEqual(self.read(fallback), "h\n1\n2\n")
        self.assertEqual(self.read(), "h\n1\n")
        self.assertIn("was saved as", out.getvalue())

    def test_flush_writes_a_log_whose_lock_is_gone(self):
        self.quietly()
        with self.locked_replace({self.path}):
            self.p.write_log_text(self.path, "late\n")
        self.assertEqual(self.p.flush_pending_logs(), [])
        self.assertEqual(self.read(), "late\n")

    def test_dataframe_rewrite_is_byte_identical_to_pandas(self):
        df = pd.DataFrame([["007", 'a;"b"', "Größe", 1.5], ["NA", "", "x", 2.0]], columns=["A", "B", "C", "D"])
        reference = os.path.join(self.dir, "reference.csv")
        df.to_csv(reference, sep=";", index=False)
        self.p.write_dataframe_csv(self.path, df)
        with open(reference, "rb") as a, open(self.path, "rb") as b:
            self.assertEqual(a.read(), b.read())

    def test_header_is_written_once(self):
        for i in range(3):
            self.p.append_csv_rows(self.path, [[i]], header=["I"])
        self.p.create_csv_file(self.path, ["I"])
        self.assertEqual(self.read(), "I\r\n0\r\n1\r\n2\r\n")


# -------------------- init helpers --------------------
class InitHelperTests(_ProtocolTestCase):
    def test_parameter_bounds(self):
        self.p.validate_parameter_bounds(["a", "b"], [(0.0, 1.0), (-2.0, 3.0, "group")], "BoTorch")
        cases = [
            ((float("nan"), 1.0), "bounds must be finite"),
            ((0.0, float("inf")), "bounds must be finite"),
            ((2.0, 1.0), "invalid bounds: low=2.0 > high=1.0"),
            ((1.0, 1.0), r"degenerate range \[1.0, 1.0\]. The MetaTAF backend needs a non-empty range"),
        ]
        for bounds, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                self.p.validate_parameter_bounds(["p"], [bounds], "MetaTAF")

    def test_objective_bounds(self):
        self.p.validate_objective_bounds(["o"], [(1.0, 1.0, 0)])  # a fixed objective scale is allowed
        for info, message in [((0.0, float("nan"), 0), "finite"), ((2.0, 1.0, 1), "invalid bounds"),
                              ((0.0, 1.0, 2), "minimize flag")]:
            with self.assertRaisesRegex(ValueError, message):
                self.p.validate_objective_bounds(["o"], [info])

    def test_raw_samples_must_cover_the_restarts(self):
        self.p.validate_raw_samples(10, 10)
        with self.assertRaisesRegex(ValueError, r"rawSamples \(5\) must be >= numRestarts \(10\)"):
            self.p.validate_raw_samples(10, 5)

    def test_user_ids_are_announced_when_the_folder_name_differs(self):
        out = self.quietly()
        ids = self.p.parse_user_ids({"user": {"userId": " a/b ", "conditionId": "c:1", "groupId": None}})
        self.assertEqual(ids, ("a/b", "c:1", "-1", "a_b", "c_1"))
        self.assertIn("Warning: userId 'a/b' was normalized to safe log-folder token 'a_b'.", out.getvalue())
        self.assertIn("Warning: conditionId 'c:1' was normalized to safe log-folder token 'c_1'.", out.getvalue())
        out.truncate(0)
        self.assertEqual(self.p.parse_user_ids({}), ("-1", "-1", "-1", "-1", "-1"))
        self.assertEqual(out.getvalue(), "")

    def test_config_getters(self):
        cfg = {"i": "3", "f": "0.5", "b": "TRUE", "s": " RBF ", "n": None}
        self.assertEqual(self.p.get_cfg_int(cfg, "i"), 3)
        self.assertEqual(self.p.get_cfg_float(cfg, "f"), 0.5)
        self.assertIs(self.p.get_cfg_bool(cfg, "b"), True)
        self.assertEqual(self.p.get_cfg_str(cfg, "s"), "rbf")
        self.assertEqual(self.p.get_cfg_int(cfg, "n", default=7), 7)
        with self.assertRaisesRegex(ValueError, "Missing required config field 'x'"):
            self.p.get_cfg_int(cfg, "x", required=True)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.p.get_cfg_bool({"b": "maybe"}, "b")


# -------------------- backends (stub harness) --------------------
@contextlib.contextmanager
def swapped_modules(stubs):
    """Put ``stubs`` into sys.modules while a backend is imported, then restore."""
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_backend(filename, stubs=None):
    name = f"{pathlib.Path(filename).stem}_protocol_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BACKEND_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    with swapped_modules(stubs or {}):
        spec.loader.exec_module(module)
    reset_protocol_state()
    return module


def dbo_torch_stub():
    """DynamicBO stand-in: replays seed points, then proposes the box centre."""
    mod = types.ModuleType("dbo_torch")
    mod.__version__ = "stub"
    mod.DBOModelConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    mod.DBOConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)

    class DynamicBO:
        def __init__(self, bounds, config):
            self.dim = len(bounds)
            self.seeds = list(config.seed_points or [])
            self.observations = []
            self.alpha = None

        @property
        def num_observations(self):
            return len(self.observations)

        @property
        def next_iteration(self):
            return self.num_observations + 1

        def is_validation_iteration(self):
            return False

        def suggest(self):
            if self.num_observations < len(self.seeds):
                return list(self.seeds[self.num_observations])
            return [0.5] * self.dim

        def observe(self, x, y):
            obs = types.SimpleNamespace(iteration=self.next_iteration, y=y, is_validation=False,
                                        predicted_y=None, predicted_sd=None)
            self.observations.append(obs)
            return obs

        def save(self, path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"observations": len(self.observations)}, f)

    mod.DynamicBO = DynamicBO
    return {"dbo_torch": mod}


def cabop_stub():
    """BayesOpt stand-in: walks through the box, realizes every candidate unchanged."""
    package = types.ModuleType("cabop")
    package.__path__ = []
    bayesopt = types.ModuleType("cabop.bayesopt")

    class BOSpace:
        def __init__(self, parameters):
            self.bounds = np.array([p["bound"] for p in parameters["parameters"].values()], dtype=float)

    class BayesOpt:
        def __init__(self, space, ifCost=True, random_state=None):
            self.space = space
            self.current_best = {}
            self.asked = 0

        def ask(self, n_init=0):
            self.asked += 1
            lo, hi = self.space.bounds[:, 0], self.space.bounds[:, 1]
            return lo + (hi - lo) * min(0.9, 0.2 * self.asked), None

        def select_sample(self, x, prefab=None):
            return np.zeros(1), np.asarray(x, dtype=float)

        def tell(self, x, y, x_intended, update_rule="actual"):
            pass

    bayesopt.BOSpace = BOSpace
    bayesopt.BayesOpt = BayesOpt
    package.bayesopt = bayesopt
    return {"cabop": package, "cabop.bayesopt": bayesopt}


def _init(n_objectives, **config):
    msg = {
        "type": "init",
        "config": {
            "numSamplingIterations": 2, "numOptimizationIterations": 2, "batchSize": 1,
            "numRestarts": 3, "rawSamples": 16, "mcSamples": 8, "seed": 5,
            "nParameters": 1, "nObjectives": n_objectives, "warmStart": False,
            "initialParametersDataPath": "", "initialObjectivesDataPath": "",
        },
        "parameters": [{"key": "p0", "init": {"low": 0.0, "high": 10.0}, "group": "default"}],
        "objectives": [{"key": f"o{j}", "init": {"low": 0.0, "high": 10.0, "minimize": j % 2}}
                       for j in range(n_objectives)],
        "user": {"userId": "u", "conditionId": "c", "groupId": "g"},
    }
    msg["config"].update(config)
    return msg


def _replies(n_objectives, count):
    return [{f"o{j}": float(1 + (i + j) % 9) for j in range(n_objectives)} for i in range(count)]


def load_bo():
    install_stub_modules()
    module = _load_backend("bo.py")
    module.SobolQMCNormalSampler = lambda sample_shape, seed: None
    module.draw_sobol_samples = lambda bounds, n, q, seed: FakeTensor(np.linspace(0.1, 0.9, n).reshape(n, q, 1))
    module.optimize_candidates = lambda model, sampler, X_baseline: FakeTensor([[0.4]])
    return module


def load_mobo():
    install_stub_modules()
    module = _load_backend("mobo.py")
    module.SobolQMCNormalSampler = lambda sample_shape, seed: None
    module.draw_sobol_samples = lambda bounds, n, q, seed: FakeTensor(np.linspace(0.1, 0.9, n).reshape(n, q, 1))
    module.optimize_qnehvi = lambda model, sampler, X_baseline: FakeTensor([[0.4]])
    return module


def load_dbo():
    install_stub_modules()
    return _load_backend("dbo_runtime.py", dbo_torch_stub())


def load_cabop():
    return _load_backend("cabop_runtime.py", cabop_stub())


def load_meta():
    install_stub_modules()
    install_openbo_stub()
    return _load_backend("meta_mobo_runtime.py")


# name -> (loader, objectives, extra init config, main() args, run folder below LogData/u/c)
BACKENDS = {
    "bo": (load_bo, 1, {}, (), "run"),
    "mobo": (load_mobo, 2, {}, (), "run"),
    "dbo": (load_dbo, 1, {}, (), "run"),
    "cabop": (load_cabop, 1, {"optimizerBackend": "cabop"}, ("single",), "CABOP/single/run"),
    "meta": (load_meta, 2, {"optimizerBackend": "meta-taf", "metaRequireSources": False}, (), "run"),
}


class BackendProtocolTests(unittest.TestCase):
    def run_main(self, module, init_msg, chunks, main_args=(), env=None):
        """main() against a scripted Unity; returns (conn, stdout, sent messages)."""
        conn = FakeConn([json_line(init_msg)] + chunks)
        self.last_conn = conn  # inspectable when main() raises
        server = FakeServerSocket(conn)
        out = io.StringIO()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_root = pathlib.Path(tmp.name) / "LogData"
        environ = {"BO_LOG_ROOT": str(self.log_root), "BO_META_ROOT": tmp.name,
                   "TORCH_EXTENSIONS_DIR": tmp.name}
        environ.update(env or {})
        with mock.patch.object(module.socket, "socket", lambda *a, **k: server), \
                mock.patch.dict(os.environ, environ), contextlib.redirect_stdout(out):
            module.main(*main_args)
        sent = [json.loads(line) for line in b"".join(conn.sent).decode("utf-8").splitlines()]
        return conn, out.getvalue(), sent

    def observation_iterations(self, folder):
        path = self.log_root / "u" / "c" / folder / "ObservationsPerEvaluation.csv"
        with open(path, newline="", encoding="utf-8") as f:
            return [int(row["Iteration"]) for row in csv.DictReader(f, delimiter=";")]

    def test_parameters_carry_the_iteration_they_are_logged_under(self):
        for name, (loader, n_obj, config, args, folder) in BACKENDS.items():
            with self.subTest(backend=name):
                module = loader()
                chunks = [json_line({"type": "objectives", "values": v}) for v in _replies(n_obj, 4)]
                _, _, sent = self.run_main(module, _init(n_obj, **config), chunks, args)
                iterations = [m["iteration"] for m in sent if m["type"] == "parameters"]
                self.assertEqual(iterations, [1, 2, 3, 4])
                self.assertEqual(self.observation_iterations(folder), iterations)
                self.assertEqual(sent[-1]["type"], "optimization_finished")

    def test_warm_start_rows_count_towards_the_iteration(self):
        for name in ("bo", "mobo", "dbo", "cabop"):
            loader, n_obj, config, args, folder = BACKENDS[name]
            with self.subTest(backend=name):
                with tempfile.TemporaryDirectory() as init_root:
                    pd.DataFrame({"p0": [2.0, 7.0]}).to_csv(os.path.join(init_root, "x.csv"), sep=";", index=False)
                    pd.DataFrame({f"o{j}": [3.0, 6.0] for j in range(n_obj)}).to_csv(
                        os.path.join(init_root, "y.csv"), sep=";", index=False)
                    chunks = [json_line({"type": "objectives", "values": v}) for v in _replies(n_obj, 2)]
                    init = _init(n_obj, warmStart=True, initialParametersDataPath="x.csv",
                                 initialObjectivesDataPath="y.csv", **config)
                    _, _, sent = self.run_main(loader(), init, chunks, args, env={"BO_INIT_ROOT": init_root})
                iterations = [m["iteration"] for m in sent if m["type"] == "parameters"]
                self.assertEqual(iterations, [3, 4])
                self.assertEqual(self.observation_iterations(folder), iterations)

    def test_stop_request_ends_every_backend_cleanly(self):
        for name, (loader, n_obj, config, args, folder) in BACKENDS.items():
            with self.subTest(backend=name):
                module = loader()
                chunks = [json_line({"type": "objectives", "values": v}) for v in _replies(n_obj, 3)]
                chunks.append(json_line({"type": "stop", "reason": "perfect rating"}))
                conn, out, sent = self.run_main(module, _init(n_obj, **config), chunks, args)  # no exception
                self.assertIn("Stop requested by Unity: perfect rating", out)
                self.assertNotIn("optimization_finished", [m["type"] for m in sent])
                self.assertEqual(len([m for m in sent if m["type"] == "parameters"]), 4)
                self.assertEqual(self.observation_iterations(folder), [1, 2, 3])
                self.assertTrue(conn.closed)

    def test_a_log_held_open_all_session_neither_ends_the_study_nor_loses_rows(self):
        """Like Excel: once the observation log exists it can be read but not written."""
        protocol = protocol_module()
        real_replace, real_append, real_overwrite = os.replace, protocol._append_file, protocol._overwrite_file

        def locked(path):
            return os.path.basename(path) == "ObservationsPerEvaluation.csv" and os.path.exists(path)

        def replace(src, dst):
            if locked(dst):
                raise PermissionError(13, "The process cannot access the file", dst)
            return real_replace(src, dst)

        def append(path, text):
            if locked(path):
                raise PermissionError(13, "The process cannot access the file", path)
            return real_append(path, text)

        def overwrite(path, text):
            if locked(path):
                raise PermissionError(13, "The process cannot access the file", path)
            return real_overwrite(path, text)

        for (name, (loader, n_obj, config, args, folder)), killed in itertools.product(
                BACKENDS.items(), (False, True)):
            with self.subTest(backend=name, killed=killed):
                module = loader()
                chunks = [json_line({"type": "objectives", "values": v}) for v in _replies(n_obj, 4)]
                # killed: Unity terminates the backend of an aborted session, no shutdown flush.
                flush = (lambda: None) if killed else protocol.flush_pending_logs
                with mock.patch.object(protocol, "time", FakeClock()), \
                        mock.patch.object(protocol.os, "replace", replace), \
                        mock.patch.object(protocol, "_append_file", append), \
                        mock.patch.object(protocol, "_overwrite_file", overwrite), \
                        mock.patch.object(module, "flush_pending_logs", flush):
                    _, out, sent = self.run_main(module, _init(n_obj, **config), chunks, args)
                self.assertEqual(sent[-1]["type"], "optimization_finished")
                self.assertEqual(out.count("Warning: could not write"), 1, out)
                self.assertEqual(protocol.unsaved_logs() != [], killed)
                protocol.discard_unsaved_logs()
                # Every evaluation reached the copy kept next to the locked log.
                saved = self.log_root / "u" / "c" / folder / "ObservationsPerEvaluation.unsaved.csv"
                with open(saved, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f, delimiter=";"))
                self.assertEqual([int(r["Iteration"]) for r in rows], [1, 2, 3, 4])

    def test_stop_in_place_of_the_init_ends_every_backend_cleanly(self):
        for name, (loader, _, _, args, _) in BACKENDS.items():
            with self.subTest(backend=name):
                conn = FakeConn([json_line({"type": "stop", "reason": "invalid_configuration"})])
                out = io.StringIO()
                module = loader()
                with mock.patch.object(module.socket, "socket", lambda *a, **k: FakeServerSocket(conn)), \
                        contextlib.redirect_stdout(out):
                    module.main(*args)  # returns: exit code 0
                self.assertIn("Stop requested by Unity: invalid_configuration", out.getvalue())
                self.assertEqual(conn.sent, [])
                self.assertTrue(conn.closed)

    def test_init_checks_run_before_the_first_design(self):
        cases = [
            ("raw samples", {"rawSamples": 4, "numRestarts": 10}, None, "rawSamples \\(4\\) must be >= numRestarts",
             ("bo", "mobo", "dbo", "meta")),
            ("degenerate range", {}, {"low": 3.0, "high": 3.0}, "degenerate range",
             ("bo", "mobo", "dbo", "cabop", "meta")),
            ("infinite bound", {}, {"low": 0.0, "high": "inf"}, "bounds must be finite",
             ("bo", "mobo", "dbo", "cabop", "meta")),
        ]
        for label, config, bounds, message, backends in cases:
            for name in backends:
                loader, n_obj, extra, args, _ = BACKENDS[name]
                with self.subTest(case=label, backend=name):
                    init = _init(n_obj, **{**extra, **config})
                    if bounds is not None:
                        init["parameters"][0]["init"] = bounds
                    with self.assertRaisesRegex(ValueError, message):
                        self.run_main(loader(), init, [], args)
                    self.assertEqual(self.last_conn.sent, [], "no design may reach Unity")
                    self.assertTrue(self.last_conn.closed)

    def test_rewritten_ids_are_announced_by_every_backend(self):
        for name, (loader, n_obj, config, args, _) in BACKENDS.items():
            with self.subTest(backend=name):
                init = _init(n_obj, numOptimizationIterations=0, **config)
                init["user"]["userId"] = "a/b"
                chunks = [json_line({"type": "objectives", "values": v}) for v in _replies(n_obj, 2)]
                _, out, _ = self.run_main(loader(), init, chunks, args)
                self.assertIn("Warning: userId 'a/b' was normalized to safe log-folder token 'a_b'.", out)


if __name__ == "__main__":
    unittest.main()
