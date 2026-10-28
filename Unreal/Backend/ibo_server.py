# ibo_server.py — the single entry point of the Interactive Bayesian Optimization (IBO) server.
#
# A client can start this script instead of choosing one of bo.py, mobo.py, cabop_bo.py,
# cabop_mobo.py, meta_mobo.py or dbo.py itself (Unity's PythonStarter and Unreal's
# launch_backend.py still choose; the Python client starts this server). It listens like
# every backend (127.0.0.1, BO_PORT or 56001; prints "Server starts, waiting for connection..."),
# reads the client's init message, picks the backend from it with the rules PythonStarter.cs uses,
# and runs that backend on the same connection (bo_protocol.adopt_connection). The wire protocol
# is unchanged: protocol/README.md (IBO protocol v1).
#
# Backend choice from init.config:
#   optimizerBackend "botorch" (default)  -> bo.py with one objective, mobo.py with more
#   optimizerBackend "cabop"              -> cabop_runtime (cabopObjectiveMode "single" or "multi")
#   optimizerBackend "meta-taf"           -> meta_mobo_runtime
#   optimizerBackend "dbo"                -> dbo_runtime
# Each backend validates the rest of the init message (objective counts, bounds, settings).

import importlib
import os
import socket
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import bo_protocol  # noqa: E402

# optimizerBackend value -> backend module and the keyword arguments of its main()
_BACKENDS = {
    "cabop": ("cabop_runtime", {"forced_mode": None}),
    "meta-taf": ("meta_mobo_runtime", {}),
    "dbo": ("dbo_runtime", {}),
}


def objective_count(init_msg):
    cfg = init_msg.get("config") or {}
    if cfg.get("nObjectives") is not None:
        return bo_protocol.get_cfg_int(cfg, "nObjectives")
    return len(init_msg.get("objectives") or [])


def select_backend(init_msg):
    """(module name, main() keyword arguments) of the backend the init message asks for."""
    cfg = init_msg.get("config") or {}
    name = bo_protocol.get_cfg_str(cfg, "optimizerBackend", default="botorch")
    if name == "botorch":
        return ("mobo" if objective_count(init_msg) > 1 else "bo"), {}
    if name in _BACKENDS:
        module, kwargs = _BACKENDS[name]
        return module, dict(kwargs)
    known = ", ".join(["botorch"] + sorted(_BACKENDS))
    raise ValueError(f"Unknown optimizerBackend '{name}' in the init message (known: {known}).")


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bo_protocol.configure_listener_socket(s)
    conn = None
    try:
        conn = bo_protocol.accept_unity_connection(s, bo_protocol.SOCKET_ACCEPT_TIMEOUT_SEC)
        init_msg = bo_protocol.receive_init_message(conn, bo_protocol.SOCKET_TIMEOUT_SEC)
        module_name, kwargs = select_backend(init_msg)
    except bo_protocol.StopRequested:
        bo_protocol.close_connection(conn)
        return
    except BaseException:
        bo_protocol.close_connection(conn)
        raise
    finally:
        s.close()

    print(f"IBO server: running backend '{module_name}'.", flush=True)
    bo_protocol.adopt_connection(conn, init_msg)
    backend = importlib.import_module(module_name)
    backend.main(**kwargs)  # closes the connection when the study ends


if __name__ == "__main__":
    main()
