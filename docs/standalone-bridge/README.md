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
[shared backend architecture](../architecture.md). Use this bridge only for a bare Unity
project where adopting BOforUnity is not an option.

Run the server (needs `dbo_torch` importable: a dbo-torch checkout or installation, or this
repository's vendored copy on `PYTHONPATH`):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Backend \
    python docs/standalone-bridge/unity_bridge.py --host 127.0.0.1 --port 8756 --save-dir runs
```

(`PYTHONDONTWRITEBYTECODE=1` keeps `__pycache__` folders out of the Unity project.)

`save` requests write only inside `--save-dir` (default: the server's working directory);
absolute paths and `..` are refused, because the bridge has no authentication.

`test_unity_bridge.py` is its pytest suite (26 tests over a real socket). The optimiser
tests use an installed `dbo_torch`, or else the vendored copy, and are skipped without
torch; the transport tests (framing, port policy, save-path checks) always run:

```bash
pytest docs/standalone-bridge -p no:cacheprovider
```

The server binds to loopback and has no authentication; `--allow-remote` is
required to bind anywhere else, so it cannot happen by accident. Like the
BOforUnity backends it never shares its port (`SO_EXCLUSIVEADDRUSE` on Windows).

`DboClient.cs` sends calls in the order they are made (one worker thread, FIFO
queue), so `Reset(); Suggest();` is safe without waiting for the first callback.

`DboClient.cs` requires the Unity Registry package `com.unity.nuget.newtonsoft-json` (tested with 3.2.2). Add that package through Package Manager before copying the client into your project. Assemblies that override precompiled references must include `Newtonsoft.Json.dll`. Replies are parsed with Json.NET. Numeric arrays must contain finite numbers in the float range. A malformed reply invokes the error callback before any success callback is queued.

`DboClientEditModeTests.cs` contains Unity NUnit tests for JSON escape decoding, complete coordinate arrays, invalid payloads, and error callback delivery through a local TCP server. To run them in this repository, copy `DboClient.cs` into `Assets/BOforUnity/Scripts/` and copy `DboClientEditModeTests.cs` into `Assets/BOforUnity/Tests/EditMode/`. Run the EditMode suite in Unity Test Runner. Remove both copies and their generated `.meta` files after testing. The persistent originals remain in this directory. The standalone tests require no Python server.
