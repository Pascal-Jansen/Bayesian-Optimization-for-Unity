"""protocol/ibo-protocol-v1.schema.json against real messages: the Python client's, the backends'
and the Unity integration's (needs jsonschema; skipped without it)."""

import json
import os
import pathlib
import sys
import unittest

_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from interactive_bo import IboClient, Objective, Parameter, Study  # noqa: E402

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

SCHEMA_PATH = pathlib.Path(_CORE_DIR).parent / "protocol" / "ibo-protocol-v1.schema.json"

# Shaped like SocketNetwork.SendInitInfo's output (nulls included, CABOP fields always present).
UNITY_INIT = {
    "type": "init",
    "config": {
        "batchSize": 1, "numRestarts": 10, "rawSamples": 1024, "numOptimizationIterations": 10, "mcSamples": 128,
        "numSamplingIterations": 6, "seed": 3, "nParameters": 2, "nObjectives": 2, "warmStart": False,
        "initialParametersDataPath": None, "initialObjectivesDataPath": "", "warmStartObjectiveFormat": "auto",
        "optimizerBackend": "botorch", "cabopObjectiveMode": "single", "cabopUpdateRule": "rule",
        "cabopUseCostAwareAcquisition": True, "cabopEnableCostBudget": False, "cabopMaxCumulativeCost": 0.0,
        "metaSourceDir": "MetaSources", "metaRequireSources": True, "metaWeightMode": "taf_r", "metaRho": 0.1,
        "metaTargetWeight": 1.0, "metaWarmupIters": 3, "metaDecayStartIter": 0, "metaDecayRate": 0.0,
        "dboSpatialKernel": "matern52", "dboAlphaParameterization": "logit", "dboInitialAlpha": 0.99,
        "dboAcquisitionTimeOffset": 1.0, "dboValidationEvery": 0, "dboValidationConfidence": 0.01,
        "dboValidationVisitedOnly": True, "dboStationaryBaseline": False, "explorationRatio": 0.0,
    },
    "parameters": [
        {"key": "ColorR", "init": {"low": 0.0, "high": 1.0}, "optSeqOrder": 0, "group": "default",
         "tolerance": 0.0, "prefabValues": []},
        {"key": "Size", "init": {"low": 0.5, "high": 2.0}, "optSeqOrder": 1, "group": "default",
         "tolerance": 0.1, "prefabValues": [1.0, 1.5]},
    ],
    "objectives": [
        {"key": "Trust", "init": {"low": 1.0, "high": 7.0, "minimize": 0}, "optSeqOrder": 0, "weight": 1.0},
        {"key": "Time", "init": {"low": 0.0, "high": 30.0, "minimize": 1}, "optSeqOrder": 1, "weight": 1.0},
    ],
    "cabopGroupCosts": [{"group": "default", "cost": {"unchanged": 0.0, "swapped": 1.0, "acquired": 2.0},
                         "actualCost": {"unchanged": 0.0, "swapped": 1.0, "acquired": 2.0}}],
    "context": None,
    "user": {"userId": "P01", "conditionId": "A", "groupId": None},
}

# As the backends send them (bo_protocol.send_json_line).
SERVER_MESSAGES = [
    {"type": "tempCoverage", "value": 0.25},
    {"type": "parameters", "values": {"ColorR": 0.4, "Size": 1.2}, "iteration": 3},
    {"type": "parameters", "values": {"ColorR": 0.4}},
    {"type": "coverage", "value": 12.5},
    {"type": "optimization_finished"},
]


class _RecordingSocket:
    def __init__(self):
        self.lines = []

    def sendall(self, data):
        self.lines.extend(json.loads(line) for line in data.decode("utf-8").splitlines())

    def settimeout(self, timeout):
        pass

    def close(self):
        pass


@unittest.skipIf(jsonschema is None, "jsonschema is not installed")
class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        cls.validator = jsonschema.Draft202012Validator(schema)

    def assertValid(self, message):
        errors = sorted(self.validator.iter_errors(message), key=lambda e: list(e.path))
        self.assertEqual(errors, [], "\n".join(e.message for e in errors))

    def assertInvalid(self, message):
        self.assertFalse(self.validator.is_valid(message), message)

    def test_python_client_messages(self):
        study = Study([Parameter("size", 0.2, 2.0)], [Objective("rating", 1, 7), Objective("time", 0, 9, minimize=True)],
                      backend="dbo", config={"dboInitialAlpha": 0.98},
                      extra={"context": {"enabled": True, "embeddingSource": "manual",
                                         "contexts": [{"key": "car", "embedding": [0.1, 0.2]}]}})
        client = IboClient()
        client._sock = _RecordingSocket()
        client.start(study)
        client._awaiting_report = True  # as after next_design()
        client.report({"rating": 5, "time": 2.5})
        client.stop("participant left")
        self.assertEqual([m["type"] for m in client._sock.lines], ["init", "objectives", "stop"])
        for message in client._sock.lines:
            with self.subTest(type=message["type"]):
                self.assertValid(message)

    def test_unity_init_and_backend_messages(self):
        self.assertValid(UNITY_INIT)
        legacy = dict(UNITY_INIT, parameters=[{"key": "x", "init": "0,1"}], objectives=[{"key": "o", "init": "0,1,0"}])
        self.assertValid(legacy)
        for message in SERVER_MESSAGES:
            with self.subTest(type=message["type"]):
                self.assertValid(message)

    def test_invalid_messages(self):
        no_counts = dict(UNITY_INIT, config={"optimizerBackend": "botorch"})
        bad_flag = dict(UNITY_INIT, objectives=[{"key": "o", "init": {"low": 0, "high": 1, "minimize": 2}}])
        for message in (
            no_counts,
            bad_flag,
            dict(UNITY_INIT, config=dict(UNITY_INIT["config"], optimizerBackend="hyperopt")),
            {"type": "objectives", "values": {"o": "high"}},
            {"type": "parameters", "values": {"x": 1}, "iteration": 0},
            {"type": "tempCoverage", "value": 1.5},
        ):
            with self.subTest(message=str(message)[:60]):
                self.assertInvalid(message)


if __name__ == "__main__":
    unittest.main()
