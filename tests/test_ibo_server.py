"""Tests for ibo_server.py, the single entry point of the iBO server (stub suite, runs on CI).

Covers the backend choice from the init message (the rules PythonStarter.cs applies when Unity
picks a script itself), the hand-over of the accepted connection and its init message through
bo_protocol, and the server's main(): it runs the chosen backend on the same connection, ends
quietly on a stop request and closes the connection on an unusable init.
"""

import contextlib
import importlib.util
import io
import os
import sys
import types
import unittest
import uuid
from unittest import mock

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _stubs import (  # noqa: E402
    BACKEND_DIR,
    FakeConn,
    FakeServerSocket,
    json_line,
    reset_protocol_state,
)


def load_server():
    reset_protocol_state()  # puts BACKEND_DIR on sys.path, so the server finds bo_protocol
    name = f"ibo_server_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, BACKEND_DIR / "ibo_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def init_message(backend=None, objectives=2, **config):
    cfg = dict(config)
    if backend is not None:
        cfg["optimizerBackend"] = backend
    return {
        "type": "init",
        "config": cfg,
        "parameters": [{"key": "x", "init": {"low": 0, "high": 1}}],
        "objectives": [{"key": f"o{i}", "init": {"low": 0, "high": 1, "minimize": 0}} for i in range(objectives)],
    }


class _ServerTestCase(unittest.TestCase):
    def setUp(self):
        self.server = load_server()
        self.p = self.server.bo_protocol
        self.addCleanup(setattr, self.p, "_HANDOVER", None)
        self.addCleanup(self.p.reset_receive_state)

    def quietly(self):
        buf = io.StringIO()
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(buf))
        self.addCleanup(stack.close)
        return buf


class SelectBackendTests(_ServerTestCase):
    def test_botorch_picks_bo_or_mobo_by_objective_count(self):
        self.assertEqual(self.server.select_backend(init_message("botorch", 1)), ("bo", {}))
        self.assertEqual(self.server.select_backend(init_message("botorch", 3)), ("mobo", {}))

    def test_botorch_is_the_default(self):
        self.assertEqual(self.server.select_backend(init_message(None, 2)), ("mobo", {}))
        self.assertEqual(self.server.select_backend(init_message("", 1)), ("bo", {}))

    def test_n_objectives_from_the_config_wins_over_the_list(self):
        msg = init_message("botorch", 1, nObjectives=2)
        self.assertEqual(self.server.select_backend(msg), ("mobo", {}))

    def test_other_backends_and_case(self):
        self.assertEqual(self.server.select_backend(init_message("CABOP", 2)), ("cabop_runtime", {"forced_mode": None}))
        self.assertEqual(self.server.select_backend(init_message("meta-taf", 2)), ("meta_mobo_runtime", {}))
        self.assertEqual(self.server.select_backend(init_message("Dbo", 1)), ("dbo_runtime", {}))

    def test_unknown_backend_names_the_known_ones(self):
        with self.assertRaisesRegex(ValueError, "Unknown optimizerBackend 'hyperopt'.*botorch, cabop, dbo, meta-taf"):
            self.server.select_backend(init_message("hyperopt", 2))


class HandoverTests(_ServerTestCase):
    def test_backend_gets_the_handed_over_connection_and_init(self):
        init = init_message("botorch", 2)
        objectives = {"type": "objectives", "values": {"o0": 1.0, "o1": 2.0}}
        conn = FakeConn([json_line(init) + json_line(objectives)])
        self.quietly()
        self.assertEqual(self.p.receive_init_message(conn, 5.0), init)  # the server reads the init

        self.p.adopt_connection(conn, init)
        listener = FakeServerSocket(None)
        self.assertIs(self.p.accept_unity_connection(listener, 5.0), conn)
        self.assertIsNone(listener.bound)  # nothing listens a second time
        self.assertEqual(self.p.receive_init_message(conn, 9.0), init)
        self.assertEqual(conn.timeout, 9.0)
        # Bytes that arrived after the init are still there for the backend.
        self.assertEqual(self.p.recv_objectives_blocking(conn), objectives["values"])

    def test_hand_over_is_used_once(self):
        init = init_message("botorch", 2)
        conn = FakeConn([])
        self.p.adopt_connection(conn, init)
        self.assertEqual(self.p.receive_init_message(conn, 1.0), init)
        self.assertIsNone(self.p._HANDOVER)
        later = FakeConn([json_line({"type": "init", "n": 2})])
        self.assertEqual(self.p.receive_init_message(later, 1.0)["n"], 2)


class MainTests(_ServerTestCase):
    def run_main(self, conn, backend_main=None):
        listener = FakeServerSocket(conn)
        imported = []

        def fake_import(name):
            imported.append(name)
            return types.SimpleNamespace(main=backend_main or (lambda **kwargs: None))

        out = self.quietly()
        with mock.patch.object(self.server.socket, "socket", return_value=listener), \
                mock.patch.object(self.server.importlib, "import_module", side_effect=fake_import):
            self.server.main()
        return listener, imported, out.getvalue()

    def test_runs_the_chosen_backend_on_the_same_connection(self):
        init = init_message("botorch", 2)
        conn = FakeConn([json_line(init)])
        seen = {}

        def backend_main(**kwargs):
            seen["kwargs"] = kwargs
            seen["conn"] = self.p.accept_unity_connection(FakeServerSocket(None), 1.0)
            seen["init"] = self.p.receive_init_message(seen["conn"], 1.0)

        listener, imported, out = self.run_main(conn, backend_main)
        self.assertEqual(imported, ["mobo"])
        self.assertEqual((seen["kwargs"], seen["conn"], seen["init"]), ({}, conn, init))
        self.assertEqual(listener.bound, (self.p.HOST, self.p.PORT))
        self.assertTrue(listener.closed)
        self.assertEqual(out.count("Server starts, waiting for connection..."), 1)
        self.assertIn("iBO server: running backend 'mobo'.", out)

    def test_stop_in_place_of_the_init_ends_without_a_backend(self):
        conn = FakeConn([json_line({"type": "stop", "reason": "invalid_configuration"})])
        _, imported, _ = self.run_main(conn)
        self.assertEqual(imported, [])
        self.assertTrue(conn.closed)

    def test_unknown_backend_closes_the_connection(self):
        conn = FakeConn([json_line(init_message("hyperopt", 2))])
        with self.assertRaises(ValueError):
            self.run_main(conn)
        self.assertTrue(conn.closed)
        self.assertIsNone(self.p._HANDOVER)


if __name__ == "__main__":
    unittest.main()
