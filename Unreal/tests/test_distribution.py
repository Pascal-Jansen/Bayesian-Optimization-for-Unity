"""Standalone distribution checks. Set BO_RUN_INTEGRATION_TESTS=1 for real optimizer runs."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("unreal_bo_launcher", ROOT / "launch_backend.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
refresh_spec = importlib.util.spec_from_file_location("unreal_bo_refresh", ROOT / "tools" / "refresh_backend.py")
refresher = importlib.util.module_from_spec(refresh_spec)
refresh_spec.loader.exec_module(refresher)


class DistributionTests(unittest.TestCase):
    def test_port_validation(self):
        for port in (1, 56001, 65535):
            self.assertEqual(launcher.validate_port(port), port)
        for port in (0, -1, 65536, 1.5, True, "56001"):
            with self.subTest(port=port), self.assertRaises(ValueError):
                launcher.validate_port(port)

    def test_line_endings_are_canonical(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.py"
            path.write_bytes(b"one\r\ntwo\r\n")
            crlf = refresher.canonical_bytes(path)
            path.write_bytes(b"one\ntwo\n")
            self.assertEqual(crlf, refresher.canonical_bytes(path))

    def test_launcher_rejects_mismatched_study(self):
        init = {"config": {"optimizerBackend": "botorch", "nObjectives": 1}}
        self.assertIs(launcher.validate_init_identity(init, "botorch", 1), init)
        for backend, objectives in (("dbo", 1), ("cabop", 1), ("botorch", 2)):
            with self.subTest(backend=backend, objectives=objectives):
                with self.assertRaises(ValueError):
                    launcher.validate_init_identity(init, backend, objectives)

    def test_backend_mapping(self):
        for backend, count, filename in (
            ("botorch", 1, "bo.py"), ("botorch", 2, "mobo.py"),
            ("cabop", 1, "cabop_bo.py"), ("cabop", 2, "cabop_mobo.py"),
            ("dbo", 1, "dbo.py"), ("meta-taf", 2, "meta_mobo.py"),
        ):
            with self.subTest(backend=backend, count=count):
                self.assertEqual(launcher.backend_entrypoint(backend, count), filename)
                self.assertTrue((ROOT / "Backend" / filename).is_file())
        for backend, count in (("dbo", 2), ("meta-taf", 1), ("botorch", 0), ("unknown", 1)):
            with self.subTest(backend=backend, count=count):
                with self.assertRaises(ValueError):
                    launcher.backend_entrypoint(backend, count)

    def test_copy_integrity(self):
        manifest = json.loads((ROOT / "backend-manifest.json").read_text(encoding="utf-8"))
        self.assertGreater(len(manifest), 20)
        for filename, expected in manifest.items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), expected)

    def test_plugin_descriptor_and_dependencies(self):
        plugin = ROOT / "Plugins" / "BayesianOptimization"
        descriptor = json.loads((plugin / "BayesianOptimization.uplugin").read_text(encoding="utf-8"))
        self.assertEqual(descriptor["Modules"][0]["Type"], "Runtime")
        self.assertFalse(descriptor["CanContainContent"])
        self.assertTrue((plugin / "Source" / "BayesianOptimization" / "BayesianOptimization.Build.cs").is_file())
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertTrue((ROOT / "Backend" / "dbo_torch" / "PROVENANCE.md").is_file())


@unittest.skipUnless(os.environ.get("BO_RUN_INTEGRATION_TESTS") == "1", "Set BO_RUN_INTEGRATION_TESTS=1 for real BO/MOBO optimization")
class BackendIntegrationTests(unittest.TestCase):
    def run_study(self, objective_count):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="unreal-bo-smoke-") as temporary:
            temporary = Path(temporary)
            output = temporary / "backend.log"
            env = dict(os.environ, BO_ACCEPT_TIMEOUT_SEC="30", BO_SOCKET_TIMEOUT_SEC="30")
            with output.open("wb") as log:
                process = subprocess.Popen([
                    sys.executable, str(ROOT / "launch_backend.py"),
                    "--backend", "botorch", "--objectives", str(objective_count),
                    "--port", str(port),
                    "--log-root", str(temporary / "logs"),
                ], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=temporary)
                connection = None
                try:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            self.fail(output.read_text(encoding="utf-8", errors="replace"))
                        try:
                            connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
                            break
                        except OSError:
                            time.sleep(0.05)
                    self.assertIsNotNone(connection, "Backend failed to listen within 30 seconds")
                    connection.settimeout(60)
                    objective_names = ["cost"] if objective_count == 1 else ["cost", "benefit"]
                    init = {
                        "type": "init",
                        "config": {
                            "optimizerBackend": "botorch", "nParameters": 1, "nObjectives": objective_count,
                            "numSamplingIterations": 2, "numOptimizationIterations": 1,
                            "batchSize": 1, "numRestarts": 2, "rawSamples": 16, "mcSamples": 16,
                            "seed": 19, "warmStart": False,
                        },
                        "parameters": [{"key": "size", "init": {"low": 10, "high": 100}, "optSeqOrder": 0}],
                        "objectives": [
                            {"key": name, "init": {"low": 0, "high": 1, "minimize": int(name == "cost")}, "optSeqOrder": index}
                            for index, name in enumerate(objective_names)
                        ],
                        "user": {"userId": "smoke", "conditionId": "unreal", "groupId": "0"},
                    }
                    # Deliberately fragment the init message across writes.
                    encoded = (json.dumps(init) + "\n").encode("utf-8")
                    connection.sendall(encoded[:17])
                    connection.sendall(encoded[17:])
                    iterations = []
                    with connection.makefile("rb") as reader:
                        while True:
                            line = reader.readline()
                            self.assertTrue(line, "Backend closed before optimization_finished")
                            message = json.loads(line)
                            if message["type"] == "parameters":
                                self.assertEqual(set(message["values"]), {"size"})
                                raw = message["values"]["size"]
                                self.assertTrue(10 <= raw <= 100)
                                x = (raw - 10) / 90
                                values = {"cost": (x - 0.25) ** 2}
                                if objective_count > 1:
                                    values["benefit"] = 1 - (x - 0.75) ** 2
                                self.assertEqual(set(values), set(objective_names))
                                connection.sendall((json.dumps({"type": "objectives", "values": values}) + "\n").encode())
                                iterations.append(message["iteration"])
                            elif message["type"] == "optimization_finished":
                                break
                            else:
                                self.assertIn(message["type"], ("coverage", "tempCoverage"))
                                self.assertTrue(math.isfinite(message["value"]))
                    self.assertEqual(iterations, [1, 2, 3])
                    connection.close()
                    connection = None
                    self.assertEqual(process.wait(timeout=20), 0, output.read_text(encoding="utf-8", errors="replace"))
                    observation_files = list((temporary / "logs").rglob("ObservationsPerEvaluation.csv"))
                    self.assertEqual(len(observation_files), 1)
                    with observation_files[0].open(newline="", encoding="utf-8-sig") as stream:
                        rows = list(csv.DictReader(stream, delimiter=";"))
                    self.assertEqual(len(rows), 3)
                    self.assertEqual([int(row["Iteration"]) for row in rows], [1, 2, 3])
                finally:
                    if connection is not None:
                        connection.close()
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)

    def test_single_objective(self):
        self.run_study(1)

    def test_multi_objective(self):
        self.run_study(2)


if __name__ == "__main__":
    unittest.main()
