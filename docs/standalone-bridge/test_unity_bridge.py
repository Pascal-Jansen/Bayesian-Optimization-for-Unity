"""Tests for the standalone TCP/JSON bridge.

These drive the server over a real socket rather than calling the handler
directly, because the framing and the error path are the parts most likely to
break an integration.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unity_bridge import DBOServer  # noqa: E402 - sibling module, not a package


@pytest.fixture
def client():
    """A running server plus a connected line-oriented client."""
    server = DBOServer("127.0.0.1", 0)  # port 0 lets the OS pick a free one
    host, port = server.server_address

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    connection = socket.create_connection((host, port), timeout=60)
    stream = connection.makefile("rw", encoding="utf-8", newline="\n")

    def call(**payload):
        stream.write(json.dumps(payload) + "\n")
        stream.flush()
        return json.loads(stream.readline())

    call.raw = stream
    try:
        yield call
    finally:
        stream.close()
        connection.close()
        server.shutdown()
        server.server_close()


def test_ping(client):
    assert client(cmd="ping") == {"ok": True, "pong": True}


def test_reset_then_suggest_and_observe(client):
    assert client(cmd="reset", bounds=[[-5.0, 9.0]], seed_points=[[5.0]])["ok"]

    suggestion = client(cmd="suggest")
    assert suggestion["ok"]
    assert suggestion["x"] == pytest.approx([5.0])
    assert suggestion["iteration"] == 1

    assert client(cmd="observe", x=suggestion["x"], y=1.0)["ok"]


def test_commands_before_reset_are_refused(client):
    reply = client(cmd="suggest")
    assert reply["ok"] is False
    assert "reset" in reply["error"].lower()


def test_unknown_command_is_reported_not_fatal(client):
    assert client(cmd="not_a_command")["ok"] is False
    # The connection must survive a bad command.
    assert client(cmd="ping")["ok"]


def test_malformed_json_is_reported_not_fatal(client):
    client.raw.write("{ this is not json\n")
    client.raw.flush()
    reply = json.loads(client.raw.readline())

    assert reply["ok"] is False
    assert "bad request" in reply["error"].lower()
    assert client(cmd="ping")["ok"]


def test_reset_requires_bounds(client):
    reply = client(cmd="reset")
    assert reply["ok"] is False
    assert "bounds" in reply["error"]


def test_full_run_reports_alpha_and_history(client):
    client(cmd="reset", bounds=[[-5.0, 9.0]], seed_points=[[5.0], [7.0], [3.0]],
           validation_every=5, seed=0)

    for i in range(10):
        suggestion = client(cmd="suggest")
        x = suggestion["x"][0]
        ideal = 5.0 * (1.0 - i / 9.0)
        assert client(cmd="observe", x=[x], y=abs(x - ideal))["ok"]

    state = client(cmd="state")
    assert state["iteration"] == 10
    assert 0.0 < state["alpha"] <= 1.0
    assert len(state["observations"]) == 10
    assert state["best"] is not None
    assert any(o["is_validation"] for o in state["observations"])


def test_predict_returns_mean_and_std(client):
    client(cmd="reset", bounds=[[-5.0, 9.0]], seed_points=[[5.0], [7.0], [3.0]], seed=0)
    for _ in range(4):
        x = client(cmd="suggest")["x"]
        client(cmd="observe", x=x, y=abs(x[0]))

    reply = client(cmd="predict", x=[0.0])
    assert reply["ok"]
    assert reply["std"] > 0


def test_stationary_flag_pins_alpha(client):
    client(cmd="reset", bounds=[[-5.0, 9.0]], seed_points=[[5.0], [7.0], [3.0]],
           stationary=True, seed=0)
    for _ in range(5):
        x = client(cmd="suggest")["x"]
        client(cmd="observe", x=x, y=abs(x[0]))

    assert client(cmd="state")["alpha"] == 1.0


def test_save_writes_the_run(client, tmp_path):
    client(cmd="reset", bounds=[[-5.0, 9.0]], seed_points=[[5.0], [7.0], [3.0]], seed=0)
    for _ in range(4):
        x = client(cmd="suggest")["x"]
        client(cmd="observe", x=x, y=abs(x[0]))

    target = tmp_path / "run.json"
    assert client(cmd="save", path=str(target))["ok"]

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert len(payload["observations"]) == 4


def test_refuses_remote_bind_without_opt_in():
    from unity_bridge import main

    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0"])
