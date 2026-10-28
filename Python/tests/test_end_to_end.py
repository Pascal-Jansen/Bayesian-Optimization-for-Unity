"""End-to-end: real studies through ibo_server.py and the Python client, without any engine.

Runs only when IBO_TEST_PYTHON names an interpreter with the optimizer stack (torch, botorch, ...),
e.g. the private environment Unity creates, or a CI job that installs requirements.txt.
"""

import csv
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from interactive_bo import Objective, Parameter, Study, launch_server, run_study  # noqa: E402
from interactive_bo.demo import STUDY_OBJECTIVES, STUDY_PARAMETERS, simulated_participant  # noqa: E402

PYTHON = os.environ.get("IBO_TEST_PYTHON")


def stack_available():
    if not PYTHON:
        return False
    probe = subprocess.run([PYTHON, "-c", "import torch, botorch, gpytorch, moocore, pandas"],
                           capture_output=True, timeout=300)
    return probe.returncode == 0


@unittest.skipUnless(stack_available(), "set IBO_TEST_PYTHON to an interpreter with the optimizer stack")
class EndToEndTests(unittest.TestCase):
    def run_and_check(self, study, evaluate, expected_designs):
        with tempfile.TemporaryDirectory() as tmp:
            log_root = pathlib.Path(tmp) / "LogData"
            with launch_server(python=PYTHON, log_root=log_root, ready_timeout=300) as server:
                history = run_study(study, evaluate, port=server.port, timeout=600)
                code = server.wait(timeout=120)
            output = "\n".join(server.output)
            self.assertEqual(code, 0, output)
            self.assertEqual([h["iteration"] for h in history], list(range(1, expected_designs + 1)), output)
            for row in history:
                for p in study.parameters:
                    self.assertTrue(p.low <= row["parameters"][p.key] <= p.high, (p.key, row))
            log = log_root / study.user_id / study.condition_id / "run" / "ObservationsPerEvaluation.csv"
            self.assertTrue(log.is_file(), output)
            with open(log, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), expected_designs, output)
            return history, output

    def test_multi_objective_botorch(self):
        study = Study(STUDY_PARAMETERS, STUDY_OBJECTIVES, sampling_iterations=3, optimization_iterations=2,
                      user_id="e2e", condition_id="mobo", config={"rawSamples": 128, "numRestarts": 4})
        _, output = self.run_and_check(study, simulated_participant(random.Random(1)), 5)
        self.assertIn("IBO server: running backend 'mobo'.", output)

    def test_single_objective_botorch(self):
        study = Study([Parameter("x", -2.0, 2.0)], [Objective("score", 0.0, 10.0)],
                      sampling_iterations=2, optimization_iterations=2, user_id="e2e", condition_id="bo",
                      config={"rawSamples": 128, "numRestarts": 4})
        _, output = self.run_and_check(study, lambda d: {"score": 10.0 - (d["x"] - 0.5) ** 2}, 4)
        self.assertIn("IBO server: running backend 'bo'.", output)

    def test_dynamic_bo_backend(self):
        study = Study([Parameter("stiffness", 0.0, 1.0)], [Objective("effort", 0.0, 5.0, minimize=True)],
                      backend="dbo", sampling_iterations=2, optimization_iterations=2,
                      user_id="e2e", condition_id="dbo", config={"dboInitialAlpha": 0.99})
        _, output = self.run_and_check(study, lambda d: {"effort": 1.0 + abs(d["stiffness"] - 0.3)}, 4)
        self.assertIn("IBO server: running backend 'dbo_runtime'.", output)


if __name__ == "__main__":
    unittest.main()
