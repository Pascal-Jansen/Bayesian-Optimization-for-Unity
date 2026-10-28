"""CABOP package.

Vendored from the reference implementation of Langerak et al., "Cost-Aware Bayesian
Optimization for Prototyping Interactive Devices" (CHI 2026): https://github.com/aalto-ui/CABOP
(commit d62a38e, MIT licence, see LICENSE in this folder).

Local changes to bayesopt.py: parameter vectors follow declaration order, as the runtime's do
(group order misaligned names and values when groups interleave); EI is guarded
against degenerate posteriors and zero configured costs; optimizer results are clamped to the
bounds; the GP's hyperparameter restarts draw from the optimizer's seed (a fixed seed did not
reproduce a run); the initial design is drawn from a single Sobol sequence (one engine per
point made it i.i.d. uniform); the "both" update rule enters an intended design only when
snapping moved it (it was otherwise entered twice); the cost model measures distances in unit
space -- the reuse tolerance is a fraction of each parameter's range and the soft-cost kernel
bandwidth (sigma = 0.05) likewise, where both were in raw parameter units, so a parameter's
units decided whether designs counted as the same prototype (identical for [0, 1] parameters);
and the reported expected cost of a proposal is computed at the proposal in original units (it
was passed unit-space coordinates). utils/ is unchanged.
"""
