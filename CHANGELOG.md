# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

How releasing works: collect notes under `## [Unreleased]` while developing. The [Release workflow](.github/workflows/release.yml) (Actions -> Release -> Run workflow) moves them under the new version heading, bumps `ProjectSettings/ProjectSettings.asset` `bundleVersion`, commits, and creates the GitHub release using that section as release notes.

Release notes for versions before 1.5.0 are available on the [GitHub releases page](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/releases).

## [Unreleased]

## [1.5.0] - 2026-07-16

### Contextual Optimization (LCE-M GP)
- Added contextual multi-task optimization built on BoTorch's `LCEMGP` (Feng et al., NeurIPS 2020) for both the single-objective (`bo.py`) and multi-objective (`mobo.py`) BoTorch backends.
- Context embeddings are definable: learned from data, supplied manually per context (any encoder, e.g. ViT-G/14 vectors), or computed from context images via open_clip (default `ViT-bigG-14`, the open_clip release of ViT-G/14) with content-hashed caching and optional L2 normalization.
- Warm-start parameter CSVs accept a `Context` column to transfer observations from other contexts (users, devices, environments); new observations are tagged with the current context.
- Run metrics (`coverage`, `IsBest`/`IsPareto`, hypervolume/best-objective traces) and the logged `Iteration` index are computed over current-context observations only; `ObservationsPerEvaluation.csv` gains a `Context` column.
- New `BoForUnityManager` inspector section with live validation and fail-fast startup checks; contextual mode is BoTorch-only (CABOP rejects it with a clear error).
- Worked around a BoTorch `LCEMGP` task-kernel dimensionality issue when `context_emb_feature` is provided.

### CABOP Fixes
- Fixed a critical parameter-ordering bug: with multiple CABOP groups whose parameters interleave in declaration order, parameter values were silently assigned to the wrong parameter names (in Unity payloads, observation logs, and warm-start data). Vectors now always follow parameter declaration order.
- Fixed a spurious `AssertionError` when the acquisition optimizer landed exactly on a parameter bound (floating-point overshoot); proposals are now clamped to bounds.
- `IsBest`/`IsPareto` marker flags are now derived from full-precision scalarized values instead of the rounded CSV values, and no longer re-scan the whole CSV every iteration.
- Zero-configured costs and degenerate GP predictions no longer produce division-by-zero in the acquisition function.
- The CABOP runtime now tolerates malformed/unrelated protocol messages the same way as the BoTorch backends.

### Runtime and Tooling
- The optimizer's Python process now runs with `PYTHONDONTWRITEBYTECODE=1`, keeping `__pycache__` folders out of `StreamingAssets`.
- Fixed a pandas dtype issue when writing `IsBest` flags during `bo.py` sampling runs.
- Context image paths configured on Windows now also resolve on macOS/Linux.
- `MainThreadDispatcher` no longer holds its queue lock while running actions, and one failing action can no longer abort the rest of the frame's queue.
- Removed the dead `Optimizer.UpdateParameter` API (it referenced CSV data that was never loaded).
- Added example warm-start CSVs with a `Context` column (`ExampleContextInitData*.csv`); documented the optional `open_clip_torch`/`pillow` dependencies in `requirements.txt`.
- Added a manually triggered Release workflow that tests, rolls this changelog, bumps `bundleVersion`, and creates the GitHub release.

### Tests
- Added unit tests for the context protocol and embedding pipeline, plus real-BoTorch integration tests running full contextual BO/MOBO loops (skipped automatically on CI environments without torch).
- Added CABOP backend tests, including a multi-group ordering regression test and bounds/zero-cost edge cases (skipped when scipy/scikit-learn/loguru are unavailable).
- Added Unity EditMode tests for final-design selection and objective key matching.
- Consolidated the duplicated torch/botorch test stubs into a shared `tests/_stubs.py`.
- Added a weekly/manual `full-stack-tests` CI job that runs the complete suite against the real pinned dependency stack (CPU torch).

### Documentation
- New README section 8.13 on contextual optimization and context embeddings (incl. ViT-G/14 guidance), new troubleshooting entries, and a note on `Iteration` numbering semantics for warm-started and contextual runs.
