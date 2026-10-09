# Provenance and licensing

This document records exactly where this implementation comes from, because the
reference implementation of Dynamic Bayesian Optimization has a licensing
situation that is easy to get wrong.

## Summary

**This repository contains no MathWorks code, and no line-by-line translation of
MathWorks code.** It is an independent implementation of a published method,
written against the equations in the peer-reviewed papers. It is safe to
publish, fork, and build on.

## The situation with the reference implementation

The reference DBO implementation is distributed on MATLAB Central File Exchange
as [Dynamic Bayesian Optimization][fex] by GilHwan Kim. It is not a standalone
library. It is a modified copy of `BayesianOptimization.m`, a source file
shipped with the MathWorks Statistics and Machine Learning Toolbox, which the
user is instructed to copy over their MATLAB installation's own file.

That file carries `Copyright 2016-2018 The MathWorks, Inc.` The File Exchange
submission attaches a BSD-3-Clause `license.txt` naming GilHwan as copyright
holder, but a BSD grant can only cover what the grantor actually owns. The
overwhelming majority of that file is MathWorks-authored and is not GilHwan's to
relicense.

The MathWorks Software License Agreement permits licensees to *modify* supplied
source files for their own applications, but redistribution of modified source
files ("Derivative Forms") is not permitted. Its carve-out for freely
distributable "User Files" applies only to files that contain no code taken from
MathWorks-supplied source.

Two consequences follow:

1. The patched `BayesianOptimization.m` must not be redistributed, and it is not
   included in this repository. It is listed in `.gitignore` as a safety net.
2. A line-by-line Python transliteration of that file would be a derivative work
   of MathWorks' code and equally unpublishable.

## Why this implementation is clean

Copyright protects a specific expression of an idea, not the idea itself.
Mathematical methods and algorithms are not copyrightable subject matter. The
DBO method — including the separable covariance function that is the whole point
of it — is described in the published literature:

> k((u,t),(u′,t′)) = k_u(u,u′) · k_t(t,t′) = k_u(u,u′) · α^|t−t′|

That equation appears as Equation (1) of the RA-L paper. Implementing it is not
copying anyone's code.

This implementation was written from the published mathematical description. Two
specific low-level details that are *not* stated in the papers — the
reparameterisation of α and the use of the iteration index as the time
coordinate — were determined by inspecting GilHwan's own additions to the MATLAB
file. Those additions are genuinely his, and are genuinely BSD-3-Clause. This
repository therefore reproduces his BSD copyright notice in `LICENSE`, as that
licence requires. See "Details taken from GilHwan's contribution" below.

## Details taken from GilHwan's contribution

These behaviours are reproduced here because matching them is necessary for
numerical agreement with the published results. They are documented rather than
hidden, and are covered by the BSD notice in `LICENSE`.

| Detail | Behaviour |
|---|---|
| α parameterisation | α is not fitted directly. A positive parameter `p` is fitted in log space and α is recovered as `α = 1 − p`, then clamped to `(0, 1]`. |
| α initial value | `p` is initialised at `0.01`, so α starts at `0.99`. |
| Spatial kernel | Squared-exponential with ARD — note this differs from the `ardmatern52` used elsewhere in MATLAB's stock Bayesian optimization. |
| Time coordinate | The iteration index (1, 2, 3, …), not wall-clock time. |
| Time column | Appended as the last column of the GP training inputs. |

For parity runs, `DBOModelConfig.matlab_compatible()` also reproduces three
numerical conventions of the stock toolbox's GP fitting: a starting lengthscale
of half the domain width, starting signal and noise standard deviations of
`std(Y)/√2`, and a noise floor of 1% of `std(Y)` (never below `1e-6`). These
are numerical defaults, not code. They are re-expressed here from scratch, and
the Tier 3 parity check verifies them against `fitrgp`'s actual behaviour
rather than against any copied source.

Everything else here — the BoTorch/GPyTorch model construction, the acquisition
optimisation, the validation-iteration logic, the API surface,
the tests — is original.

## Citation

Cite the method papers, not this repository alone. Prof. Sergi's stated
preference is that the primary citation is the computational paper that
describes the method in detail:

- Kim, G. and Sergi, F. *Dynamic Bayesian optimization for non-stationary
  systems.* Computer Methods in Biomechanics and Biomedical Engineering, 2025.
  <https://doi.org/10.1080/10255842.2025.2595150>

- Kim, G. and Sergi, F. *Validation of Dynamic Bayesian Optimization for a
  Non-Stationary Human-in-the-Loop Optimization Problem.* IEEE Robotics and
  Automation Letters, 11(5):5733–5740, 2026.
  <https://doi.org/10.1109/LRA.2026.3665072>

A further paper is under review; a preprint is available at
<https://www.biorxiv.org/content/10.64898/2026.06.10.731447v1>.

Reference implementation:

- Kim, G. *Dynamic Bayesian Optimization.* MATLAB Central File Exchange, 2026.
  <https://uk.mathworks.com/matlabcentral/fileexchange/183999-dynamic-bayesian-optimization>

[fex]: https://uk.mathworks.com/matlabcentral/fileexchange/183999-dynamic-bayesian-optimization
