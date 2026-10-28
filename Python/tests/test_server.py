"""Tests for interactive_bo.server: launching ibo_server.py (a stand-in script; standard library only)."""

import json
import os
import pathlib
import sys
import tempfile
import textwrap
import unittest

_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from interactive_bo import IboClient, launch_server  # noqa: E402
from interactive_bo.server import READY_LINE  # noqa: E402

# Behaves like the real server on the wire: prints the ready line, serves one session, exits.
FAKE_SERVER = textwrap.dedent(f"""
    import json, os, socket
    print("env", json.dumps({{k: os.environ.get(k) for k in ("BO_PORT", "BO_LOG_ROOT", "BO_INIT_ROOT")}}), flush=True)
    print("cwd", os.getcwd(), flush=True)
    s = socket.socket()
    s.bind(("127.0.0.1", int(os.environ["BO_PORT"])))
    s.listen(1)
    print({READY_LINE!r}, flush=True)
    conn, _ = s.accept()
    reader = conn.makefile("r", encoding="utf-8")
    init = json.loads(reader.readline())
    conn.sendall((json.dumps({{"type": "parameters", "values": {{"x": 0.5}}, "iteration": 1}}) + "\\n").encode())
    reader.readline()
    conn.sendall(b'{{"type": "optimization_finished"}}\\n')
    conn.close()
""")


class LaunchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        self.backend_dir = self.root / "BOData" / "BayesianOptimization"
        self.backend_dir.mkdir(parents=True)

    def write_server(self, source):
        (self.backend_dir / "ibo_server.py").write_text(source, encoding="utf-8")

    def test_starts_serves_one_session_and_passes_the_folders(self):
        self.write_server(FAKE_SERVER)
        log_root = self.root / "logs"
        with launch_server(backend_dir=self.backend_dir, log_root=log_root, ready_timeout=30) as server:
            with IboClient(port=server.port, timeout=10) as client:
                client.start({"type": "init"})
                self.assertEqual(client.next_design().values, {"x": 0.5})
                client.report({"o": 1.0})
                self.assertIsNone(client.next_design())
            self.assertEqual(server.wait(timeout=30), 0)
        env_line = next(line for line in server.output if line.startswith("env "))
        env = json.loads(env_line[4:])
        self.assertEqual(env["BO_PORT"], str(server.port))
        self.assertEqual(pathlib.Path(env["BO_LOG_ROOT"]), log_root.resolve())
        self.assertEqual(pathlib.Path(env["BO_INIT_ROOT"]), (self.backend_dir.parent / "InitData").resolve())
        self.assertTrue(log_root.is_dir())
        cwd_line = next(line for line in server.output if line.startswith("cwd "))
        self.assertEqual(pathlib.Path(cwd_line[4:]).resolve(), self.backend_dir.parent.resolve())

    def test_a_server_that_fails_at_startup_reports_its_output(self):
        self.write_server('print("Traceback: no module named torch", flush=True)\nraise SystemExit(3)\n')
        with self.assertRaisesRegex(RuntimeError, r"(?s)did not start.*exit code 3.*no module named torch"):
            launch_server(backend_dir=self.backend_dir, log_root=self.root / "logs", ready_timeout=30)

    def test_a_server_that_never_gets_ready_is_stopped(self):
        self.write_server("import time\nprint('loading', flush=True)\ntime.sleep(60)\n")
        with self.assertRaisesRegex(RuntimeError, r"(?s)did not start within 1 s.*loading"):
            launch_server(backend_dir=self.backend_dir, log_root=self.root / "logs", ready_timeout=1)

    def test_missing_script(self):
        with self.assertRaisesRegex(FileNotFoundError, "ibo_server.py not found"):
            launch_server(backend_dir=self.root, log_root=self.root / "logs")


if __name__ == "__main__":
    unittest.main()
