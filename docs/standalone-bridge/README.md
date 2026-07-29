# Standalone DBO bridge (not part of the BOforUnity asset)

A minimal client/server pair for driving a `dbo_torch.DynamicBO` optimiser from
a Unity project that does **not** use BOforUnity. Kept here for reference; it is
deliberately outside `Assets/` so Unity never imports it.

**Do not mix this with the BOforUnity DBO backend.** The two speak incompatible
protocols with opposite topologies:

| | BOforUnity backend | This bridge |
|---|---|---|
| Server | Python (`dbo_runtime.py`), port 56001 | Python (`unity_bridge.py`), port 8756 |
| Driver | Python sends `parameters`, blocks for `objectives` | Unity sends explicit commands (`reset`, `suggest`, `observe`, …) |
| Client | BOforUnity's `SocketNetwork.cs` | `DboClient.cs` (drop into any Unity project) |

If you are inside this repository, you want the BOforUnity backend — see
[docs/dbo-backend.md](../dbo-backend.md). Use this bridge only for a bare Unity
project where adopting BOforUnity is not an option.

Run the server (needs `dbo-torch` and its dependencies on the Python side):

```bash
python unity_bridge.py --host 127.0.0.1 --port 8756
```

`test_unity_bridge.py` is its pytest suite (11 tests over a real socket); run it
with `dbo_torch` importable, e.g. from a dbo-torch checkout:

```bash
pytest docs/standalone-bridge/test_unity_bridge.py
```

The server binds to loopback and has no authentication; `--allow-remote` is
required to bind anywhere else, so it cannot happen by accident.
