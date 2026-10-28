# Architecture: one backend, any front end

`Backend/` owns optimization, protocol validation, normalization, warm starts, logging, and provenance. The front end (Unity, Unreal, the Python client or any other program that speaks the protocol) owns the presentation of each design, measurement collection, scene or application lifecycle, and connection setup.

Front end and backend exchange newline delimited JSON over loopback TCP. The client sends its configuration. The backend returns a complete parameter map and an iteration identifier. The client evaluates that design and returns a complete objective map. Raw units and explicit minimization flags are preserved across this boundary. Numerical transformations run in the backend. The messages are specified in [protocol/README.md](../protocol/README.md).

Unity and Unreal start the backend script of the selected optimizer directly (`bo.py`, `mobo.py`, `cabop_bo.py`, `cabop_mobo.py`, `dbo.py`, `meta_mobo.py`). `Backend/ibo_server.py` is the single entry point for the Python client and new integrations: it reads the init message, picks the backend from `optimizerBackend` and the number of objectives, and hands the connection over.

The iteration identifier links engine telemetry to observation rows. Each study should retain its backend selection, source revision, environment, seed, and configuration. The model library is an experimental factor. Changing it during a study changes the numerical procedure.

Runtime copies let each engine package run independently after installation. `tools/sync_backends.py` generates both copies from `Backend/` and records canonical text hashes. Continuous integration verifies the copies. Numerical maintenance therefore remains centralized.

The Unity shared support assembly owns logging paths and integration contracts. The Unreal wrapper exposes a Blueprint component and native actor example. Both wrappers use configurable loopback ports. The Unreal launcher verifies the backend and objective count before numerical execution.

Releases use one repository tag and separate engine artifacts. Packaging excludes commercial questionnaire content, installers, caches, and unrelated examples. The source retains upstream MIT attribution and vendored licenses. Protocol changes require tests against both wrappers.
