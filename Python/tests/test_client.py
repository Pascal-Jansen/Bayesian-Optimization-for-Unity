"""Tests for interactive_bo.client against a scripted IBO server on a real socket (standard library only)."""

import json
import math
import os
import socket
import sys
import threading
import unittest

_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from interactive_bo import (  # noqa: E402
    PROTOCOL_VERSION,
    IboClient,
    Objective,
    Parameter,
    ProtocolError,
    Study,
    run_study,
)


def make_study(**kwargs):
    return Study([Parameter("size", 0.2, 2.0), Parameter("contrast", 0.0, 1.0)],
                 [Objective("usability", 0, 100), Objective("time", 0, 10, minimize=True)], **kwargs)


class ScriptedServer:
    """Accepts one client; ``script(conn, receive)`` plays the server side and records what arrives."""

    def __init__(self, script):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.received = []
        self.error = None
        self._script = script
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            conn, _ = self.listener.accept()
            # The reader holds its own reference to the socket: close both, so the client sees the end.
            with conn, conn.makefile("r", encoding="utf-8") as reader:
                def receive():
                    line = reader.readline()
                    message = json.loads(line) if line else None
                    self.received.append(message)
                    return message

                def send(message, raw=None):
                    conn.sendall(raw if raw is not None else (json.dumps(message) + "\n").encode("utf-8"))

                self._script(send, receive)
        except Exception as e:  # surfaced by join()
            self.error = e
        finally:
            self.listener.close()

    def join(self):
        self._thread.join(timeout=10)
        if self.error is not None:
            raise self.error


def full_session(designs, finish=True):
    """A server that sends each design, waits for its objectives and then finishes."""
    def script(send, receive):
        receive()  # init
        for i, values in enumerate(designs, start=1):
            send({"type": "tempCoverage", "value": i / len(designs)})
            send({"type": "parameters", "values": values, "iteration": i})
            receive()  # objectives
            send({"type": "coverage", "value": float(i)})
        if finish:
            send({"type": "optimization_finished"})
    return script


class StudyTests(unittest.TestCase):
    def test_init_message_matches_the_protocol(self):
        msg = make_study(user_id="P01", condition_id="A", seed=7, optimization_iterations=12).init_message()
        self.assertEqual(msg["type"], "init")
        self.assertEqual(msg["protocolVersion"], PROTOCOL_VERSION)
        self.assertEqual(msg["parameters"][0], {"key": "size", "init": {"low": 0.2, "high": 2.0}})
        self.assertEqual(msg["objectives"][1], {"key": "time", "init": {"low": 0.0, "high": 10.0, "minimize": 1}})
        self.assertEqual(msg["user"], {"userId": "P01", "conditionId": "A", "groupId": "-1"})
        cfg = msg["config"]
        self.assertEqual((cfg["optimizerBackend"], cfg["nParameters"], cfg["nObjectives"]), ("botorch", 2, 2))
        self.assertEqual((cfg["numSamplingIterations"], cfg["numOptimizationIterations"], cfg["seed"]), (6, 12, 7))
        self.assertEqual((cfg["numRestarts"], cfg["rawSamples"], cfg["mcSamples"], cfg["batchSize"]), (10, 1024, 128, 1))

    def test_config_and_extra_pass_through(self):
        msg = make_study(backend="dbo", config={"dboInitialAlpha": 0.98, "rawSamples": 256},
                         extra={"context": {"enabled": True}}).init_message()
        self.assertEqual(msg["config"]["optimizerBackend"], "dbo")
        self.assertEqual((msg["config"]["dboInitialAlpha"], msg["config"]["rawSamples"]), (0.98, 256))
        self.assertEqual(msg["context"], {"enabled": True})

    def test_invalid_studies_are_refused_before_sending(self):
        cases = [
            (Study([], [Objective("o", 0, 1)]), "at least one parameter"),
            (Study([Parameter("x", 0, 1), Parameter("x", 0, 2)], [Objective("o", 0, 1)]), "unique"),
            (Study([Parameter("o", 0, 1)], [Objective("o", 0, 1)]), "unique"),
            (Study([Parameter(" ", 0, 1)], [Objective("o", 0, 1)]), "non-empty"),
            (Study([Parameter("x", 0, math.inf)], [Objective("o", 0, 1)]), "finite"),
            (Study([Parameter("x", 2, 1)], [Objective("o", 0, 1)]), "low 2 > high 1"),
        ]
        for study, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                study.init_message()


class ClientTests(unittest.TestCase):
    def test_full_loop_tracks_progress_and_finishes(self):
        server = ScriptedServer(full_session([{"size": 1.0, "contrast": 0.5}, {"size": 1.5, "contrast": 0.9}]))
        seen = []
        with IboClient(timeout=10, port=server.port, on_message=seen.append) as client:
            client.start(make_study())
            design = client.next_design()
            self.assertEqual((design.values, design.iteration, client.sampling_progress), ({"size": 1.0, "contrast": 0.5}, 1, 0.5))
            client.report({"usability": 80, "time": 2.5})
            design = client.next_design()
            self.assertEqual((design.iteration, client.coverage), (2, 1.0))
            client.report({"usability": 90, "time": 2.0})
            self.assertIsNone(client.next_design())
            self.assertTrue(client.finished)
            self.assertEqual(client.coverage, 2.0)
        server.join()
        init, first, second = server.received
        self.assertEqual(init["type"], "init")
        self.assertEqual(first, {"type": "objectives", "values": {"usability": 80.0, "time": 2.5}})
        self.assertEqual(second["values"], {"usability": 90.0, "time": 2.0})
        self.assertEqual([m["type"] for m in seen][-1], "optimization_finished")

    def test_unknown_messages_and_split_utf8_are_handled(self):
        def script(send, receive):
            receive()
            payload = (json.dumps({"type": "parameters", "values": {"Größe": 1.0}, "iteration": 1},
                                  ensure_ascii=False) + "\n").encode("utf-8")
            cut = payload.index("ö".encode("utf-8")) + 1  # split inside the two-byte character
            send(None, raw=b'{"type": "status", "text": "warming up"}\n\n' + payload[:cut])
            send(None, raw=payload[cut:])
            receive()
            send({"type": "optimization_finished"})
        server = ScriptedServer(script)
        with IboClient(timeout=10, port=server.port) as client:
            client.start({"type": "init"})
            self.assertEqual(client.next_design().values, {"Größe": 1.0})
            client.report({"o": 1})
            self.assertIsNone(client.next_design())
        server.join()

    def test_server_closing_early_is_an_error(self):
        server = ScriptedServer(full_session([{"size": 1.0}], finish=False))
        with IboClient(timeout=10, port=server.port) as client:
            client.start(make_study())
            client.next_design()
            client.report({"usability": 1, "time": 1})
            with self.assertRaisesRegex(ProtocolError, "closed the connection before the study finished"):
                client.next_design()
        server.join()

    def test_report_and_next_design_must_alternate(self):
        server = ScriptedServer(full_session([{"size": 1.0}]))
        with IboClient(timeout=10, port=server.port) as client:
            client.start(make_study())
            with self.assertRaisesRegex(RuntimeError, "call next_design"):
                client.report({"usability": 1, "time": 1})
            client.next_design()
            with self.assertRaisesRegex(RuntimeError, "Report the objectives"):
                client.next_design()
            with self.assertRaisesRegex(ValueError, "not a finite number"):
                client.report({"usability": float("nan"), "time": 1})
            client.report({"usability": 1, "time": 1})
            self.assertIsNone(client.next_design())
        server.join()

    def test_stop_is_sent(self):
        def script(send, receive):
            receive()
            send({"type": "parameters", "values": {"size": 1.0}, "iteration": 1})
            receive()
        server = ScriptedServer(script)
        with IboClient(timeout=10, port=server.port) as client:
            client.start(make_study())
            client.next_design()
            client.stop("participant left")
        server.join()
        self.assertEqual(server.received[-1], {"type": "stop", "reason": "participant left"})

    def test_run_study_returns_the_history(self):
        server = ScriptedServer(full_session([{"size": 1.0, "contrast": 0.1}, {"size": 2.0, "contrast": 0.2}]))
        history = run_study(make_study(), lambda d: {"usability": d["size"] * 10, "time": d["contrast"]}, port=server.port, timeout=10)
        server.join()
        self.assertEqual([h["iteration"] for h in history], [1, 2])
        self.assertEqual(history[1]["objectives"], {"usability": 20.0, "time": 0.2})


if __name__ == "__main__":
    unittest.main()
