# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

How releasing works: collect notes under `## [Unreleased]` while developing. The [Release workflow](.github/workflows/release.yml) (Actions -> Release -> Run workflow) moves them under the new version heading, bumps `ProjectSettings/ProjectSettings.asset` `bundleVersion`, commits, and creates the GitHub release using that section as release notes.

Release notes for versions before 1.5.0 are available on the [GitHub releases page](https://github.com/Pascal-Jansen/Bayesian-Optimization-for-Unity/releases).

## [Unreleased]

### Contextual Optimization (LCE-M GP)
- Added contextual multi-task optimization built on BoTorch's `LCEMGP` (Feng et al., NeurIPS 2020) for both the single-objective (`bo.py`) and multi-objective (`mobo.py`) BoTorch backends.
- Context embeddings are definable: learned from data, supplied manually per context (any encoder, e.g. ViT-G/14 vectors), or computed from context images via open_clip (default `ViT-bigG-14`, the open_clip release of ViT-G/14) with content-hashed caching and optional L2 normalization.
- Warm-start parameter CSVs accept a `Context` column to transfer observations from other contexts (users, devices, environments); new observations are tagged with the current context.
- Run metrics (`coverage`, `IsBest`/`IsPareto`, hypervolume/best-objective traces) and the logged `Iteration` index are computed over current-context observations only; `ObservationsPerEvaluation.csv` gains a `Context` column.
- New `BoForUnityManager` inspector section with live validation and fail-fast startup checks; contextual mode is BoTorch-only (CABOP rejects it with a clear error).
- Worked around a BoTorch `LCEMGP` task-kernel dimensionality issue when `context_emb_feature` is provided.

### Runtime and Tooling
- The optimizer's Python process now runs with `PYTHONDONTWRITEBYTECODE=1`, keeping `__pycache__` folders out of `StreamingAssets`.
- Fixed a pandas dtype issue when writing `IsBest` flags during `bo.py` sampling runs.
- Context image paths configured on Windows now also resolve on macOS/Linux.
- Added example warm-start CSVs with a `Context` column (`ExampleContextInitData*.csv`); documented the optional `open_clip_torch`/`pillow` dependencies in `requirements.txt`.
- Added a manually triggered Release workflow that tests, rolls this changelog, bumps `bundleVersion`, and creates the GitHub release.

### Tests
- Added unit tests for the context protocol and embedding pipeline, plus real-BoTorch integration tests running full contextual BO/MOBO loops (skipped automatically on CI environments without torch).

### Documentation
- New README section 8.13 on contextual optimization and context embeddings (incl. ViT-G/14 guidance) and new troubleshooting entries.
