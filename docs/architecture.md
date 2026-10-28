# Shared backend and engine wrappers

`Backend/` owns optimization, protocol validation, normalization, warm starts, logging, and provenance. Unity and Unreal own scene lifecycle, parameter application, measurement collection, presentation, and connection setup.

The engines exchange newline delimited JSON over loopback TCP. The client sends its configuration. The backend returns a complete parameter map and an iteration identifier. The client evaluates that design and returns a complete objective map. Raw units and explicit minimization flags are preserved across this boundary. Numerical transformations run in the backend.

The iteration identifier links engine telemetry to observation rows. Each study should retain its backend selection, source revision, environment, seed, and configuration. The model library is an experimental factor. Changing it during a study changes the numerical procedure.

Runtime copies let each engine package run independently after installation. `tools/sync_backends.py` generates both copies from `Backend/` and records canonical text hashes. Continuous integration verifies the copies. Numerical maintenance therefore remains centralized.

The Unity shared support assembly owns logging paths and integration contracts. The Unreal wrapper exposes a Blueprint component and native actor example. Both wrappers use configurable loopback ports. The Unreal launcher verifies the backend and objective count before numerical execution.

Releases use one repository tag and separate engine artifacts. Packaging excludes commercial questionnaire content, installers, caches, and unrelated examples. The source retains upstream MIT attribution and vendored licenses. Protocol changes require tests against both wrappers.
