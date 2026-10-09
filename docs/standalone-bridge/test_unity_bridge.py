"""Tests for the standalone TCP/JSON bridge.

These drive the server over a real socket rather than calling the handler
directly, because the framing and the error path are the parts most likely to
break an integration.

The transport tests (framing, port policy, save-path checks) need no torch. The
optimiser tests need ``dbo_torch`` and its torch/botorch stack: they use an
installed ``dbo_torch`` if there is one, otherwise BOforUnity's vendored copy in
``Assets/StreamingAssets/BOData/BayesianOptimization``, and are skipped when
neither imports.
"""

from __future__ import annotations

import importlib
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_VENDORED = _HERE.parents[1] / "Assets" / "StreamingAssets" / "BOData" / "BayesianOptimization"

sys.path.insert(0, str(_HERE))
from unity_bridge import DBOServer, main, resolve_save_path  # noqa: E402 - sibling module, not a package


def _import_dbo() -> bool:
    try:
        importlib.import_module("dbo_torch.optimizer")
        return True
    except ImportError:
        pass
    if not (_VENDORED / "dbo_torch").is_dir():
        return False
    # Never write __pycache__ into the Unity project.
    sys.dont_write_bytecode = True
    sys.path.append(str(_VENDORED))
    try:
        importlib.import_module("dbo_torch.optimizer")
        return True
    except ImportError:
        return False


HAVE_DBO = _import_dbo()
needs_dbo = pytest.mark.skipif(not HAVE_DBO, reason="dbo_torch (torch/botorch) is not importable")


@pytest.fixture
def client(tmp_path):
    """A running server plus a connected line-oriented client."""
    server = DBOServer("127.0.0.1", 0, save_dir=tmp_path)  # port 0 lets the OS pick a free one
    host, port = server.server_address[:2]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    connection = socket.create_connection((host, port), timeout=60)
    stream = connection.makefile("rw", encoding="utf-8", newline="\n")

    def call(**payload):
        stream.write(json.dumps(payload) + "\n")
        stream.flush()
        return json.loads(stream.readline())

    call.raw = stream
    call.save_dir = tmp_path
    try:
        yield call
    finally:
        stream.close()
        connection.close()
        server.shutdown()
        server.server_close()


def _run_four_iterations(client):
    client(cmd="reset", bounds=[[-5.0, 9.0]], seed_points=[[5.0], [7.0], [3.0]], seed=0)
    for _ in range(4):
        x = client(cmd="suggest")["x"]
        client(cmd="observe", x=x, y=abs(x[0]))


# --------------------------------------------------------------------------- transport


def test_ping(client):
    assert client(cmd="ping") == {"ok": True, "pong": True}


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


def test_refuses_remote_bind_without_opt_in():
    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0"])


def test_binds_loopback_by_default():
    server = DBOServer(port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_listener_never_shares_its_port():
    first = DBOServer("127.0.0.1", 0)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            assert first.socket.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE) != 0
        else:
            assert first.socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) != 0
        port = first.server_address[1]
        # A second server must fail instead of silently taking over the port.
        with pytest.raises(OSError):
            second = DBOServer("127.0.0.1", port)
            second.server_close()
    finally:
        first.server_close()


@pytest.mark.parametrize(
    "path",
    [
        "/etc/bo4u_escape.json",
        "C:\\Windows\\bo4u_escape.json",
        "C:bo4u_escape.json",
        "\\\\server\\share\\bo4u_escape.json",
        "../bo4u_escape.json",
        "..\\bo4u_escape.json",
        "runs/../../bo4u_escape.json",
        "",
        ".",
    ],
)
def test_save_refuses_paths_outside_the_save_dir(client, path):
    reply = client(cmd="save", path=path)
    assert reply["ok"] is False
    assert "path" in reply["error"]
    assert not (client.save_dir.parent / "bo4u_escape.json").exists()


def test_save_requires_a_path(client):
    reply = client(cmd="save")
    assert reply["ok"] is False
    assert "path" in reply["error"]


def test_save_refuses_symlink_escape(tmp_path):
    save_dir = tmp_path / "runs"
    outside = tmp_path / "outside"
    save_dir.mkdir()
    outside.mkdir()
    try:
        (save_dir / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    with pytest.raises(ValueError):
        resolve_save_path(save_dir, "link/run.json")


def test_save_path_resolves_inside_save_dir(tmp_path):
    assert resolve_save_path(tmp_path, "run.json") == (tmp_path / "run.json").resolve()
    assert resolve_save_path(tmp_path, "p01\\run.json") == (tmp_path / "p01" / "run.json").resolve()


# --------------------------------------------------------------------------- optimiser


@needs_dbo
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


def test_reset_requires_bounds(client):
    reply = client(cmd="reset")
    assert reply["ok"] is False
    assert "bounds" in reply["error"]


@needs_dbo
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


@needs_dbo
def test_predict_returns_mean_and_std(client):
    _run_four_iterations(client)

    reply = client(cmd="predict", x=[0.0])
    assert reply["ok"]
    assert reply["std"] > 0


@needs_dbo
def test_stationary_flag_pins_alpha(client):
    client(cmd="reset", bounds=[[-5.0, 9.0]], seed_points=[[5.0], [7.0], [3.0]],
           stationary=True, seed=0)
    for _ in range(5):
        x = client(cmd="suggest")["x"]
        client(cmd="observe", x=x, y=abs(x[0]))

    assert client(cmd="state")["alpha"] == 1.0


@needs_dbo
def test_save_writes_the_run(client):
    _run_four_iterations(client)

    reply = client(cmd="save", path="run.json")
    assert reply["ok"]

    target = client.save_dir / "run.json"
    assert Path(reply["path"]) == target.resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert len(payload["observations"]) == 4


@needs_dbo
def test_save_creates_subfolders_inside_the_save_dir(client):
    _run_four_iterations(client)

    assert client(cmd="save", path="p01/session1/run.json")["ok"]
    assert (client.save_dir / "p01" / "session1" / "run.json").is_file()
