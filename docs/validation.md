# Validation record

Validation was performed on Windows on 28 October 2026. The Unity Editor version was `6000.6.7f1`. Unreal Engine was `5.8`. URP and HDRP used version `17.6.0`. Python used the pinned requirements and OpenBO commit `43279884eb9ef6e97b998fc26b49942869c9eda1`.

This record covers [Bayesian-Optimization-for-Game-Engines](https://github.com/M-Colley/Bayesian-Optimization-for-Game-Engines) in the state iBO adopted, before the iBO renaming, the QuestionnaireToolkit add-on and the merge of version 1.9. The checks of iBO 2.0.0-rc.1 so far are listed at the end.

| Check | Result |
| --- | --- |
| Full Python suite in the clean common repository | 522 passed, one platform skip, 320 subtests passed |
| Release packaging regressions | Six passed |
| Unity core EditMode tests without QuestionnaireToolkit | 47 passed |
| Unity dispatcher PlayMode test in the authoring project | One passed |
| Repeated Play sessions with Domain Reload disabled | Passed inside the lifecycle regression |
| Unreal Editor Development, Win64 Development, Win64 Shipping | All builds passed |
| Unreal native protocol automation | One passed, zero warnings and errors |
| Built-in rendering | GPU readback passed |
| URP rendering | Full questionnaire independent package compiled and rendered correctly |
| HDRP rendering with both SRP packages installed | Full package compiled and rendered correctly |
| Fresh Unity asset import without QuestionnaireToolkit or Test Framework | Passed |
| Native Unreal analytic session | Eight evaluations passed against the real Python process |
| Backend source synchronization | Both generated copies match 39 canonical files |
| Concurrent protocol processes | Independent ports passed isolation checks |

Rendering checks generated Lit and Unlit materials through public validation APIs. They inspected distinct object colors and the background revealed by zero alpha. The pipeline fixtures also generated the analytic optimization scene and checked its objective values and component references. These checks establish shader and source compatibility. Study stimuli still require visual calibration for lighting, exposure, and tone mapping.

The Python run reported numerical convergence and dependency deprecation warnings. The single skip concerns POSIX permissions on Windows. The tests used the actual BoTorch, botorch-rust, CABOP, DBO, and optional OpenBO stack. Native installer actions and actual image encoder weights remain untested.

Run `python -m pytest -q` with `BO_RUN_INTEGRATION_TESTS=1` for the full Python suite. Run Unity tests through Test Runner. Run `Automation RunTests BayesianOptimization` for native protocol checks. The rendering fixture is documented in [render-pipeline-support.md](render-pipeline-support.md).

macOS, Linux, and standalone player deployment require validation before support is advertised. Asset Store acceptance also requires the actions in [publication-status.md](publication-status.md).

## iBO 2.0.0-rc.1

Checked on macOS (Apple Silicon) on 28 October 2026 with Python 3.13; Unity 6000.6 and Unreal Engine were not available.

| Check | Result |
| --- | --- |
| Backend, packaging and standalone-bridge tests (`python -m pytest -q tests docs/standalone-bridge`, without torch) | 464 passed, 69 skipped |
| Unreal portable tests | Passed |
| Python client tests, including real BoTorch (single and multi-objective) and DBO studies through `ibo_server.py` | Passed; the client tests also pass on Python 3.9 |
| Backend source synchronization | Both generated copies match 40 canonical files, Unity `.meta` files complete |
| Core and add-on compiled in a fresh Unity 6000.4.7f1 project | No compiler errors or warnings |
| Unity EditMode tests | 48 passed |
| Scenes and prefabs | No missing scripts in 9 scenes and 29 prefabs |

Open before the final release: the renamed package in Unity 6000.6 and Unreal Engine 5.8 (builds, Play-mode runs with Python, rendering checks), and engine validation on macOS and Linux.
