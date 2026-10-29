# Shared optimization backend

The common source repository stores the canonical Python implementation in its root `Backend/` directory. Source changes belong there. Engine releases contain generated copies of that implementation. The shared source includes optimizer code, dependency versions, source model instructions, and example data.

Run `python tools/sync_backends.py` from the repository root after changing this directory. Run `python tools/sync_backends.py --check` to verify both mirrors. The Unity mirror is `Unity/Assets/StreamingAssets/BOData` (the sync also creates its Unity `.meta` files). The Unreal mirror is `Unreal/Backend`, with dependency and data files beside it.

Unity and Unreal start the script of the selected optimizer directly: `bo.py`, `mobo.py`, `cabop_bo.py`, `cabop_mobo.py`, `dbo.py` or `meta_mobo.py`. `ibo_server.py` is the single entry point for the Python client (`Python/` in the repository) and new integrations: it reads the init message, picks the backend from `optimizerBackend` and the number of objectives, and hands the connection over. All of them speak the iBO protocol (`protocol/README.md` in the repository).

Install `requirements.txt` into a Python 3.13 virtual environment. Optional OpenBO installation uses the audited source revision specified in that file. Preserve the complete environment, run metadata, source population manifest, and seed for study replication.

Each backend listens on `127.0.0.1`. Set `BO_PORT` to an integer from 1 to 65535 to select its port. The default is 56001. Configure the same port in the engine wrapper. Concurrent studies require distinct ports and distinct log directories. The server accepts one client per process.

Set `BO_INIT_ROOT`, `BO_LOG_ROOT`, and `BO_META_ROOT` to control input data, output logs, and relative source model paths. The Unreal launcher sets the first two paths. The Unity wrapper uses its configured application data paths. Absolute paths remain supported.

Contextual optimization is available through the BoTorch backend. CABOP, DBO, and MetaTAF reject enabled contextual configuration. CABOP supports a scalarized multiple objective mode. DBO supports one drifting objective. MetaTAF requires at least two objectives and a compatible source model population by default.

The root license covers the original backend code. Vendored packages retain their license files and provenance records. Installed dependencies retain their own licenses.
