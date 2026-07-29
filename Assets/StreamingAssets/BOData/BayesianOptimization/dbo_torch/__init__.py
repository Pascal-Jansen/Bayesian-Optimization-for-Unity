"""Dynamic Bayesian Optimization on BoTorch.

Bayesian optimisation for problems whose objective drifts while you are
optimising it — motor adaptation, fatigue, learning, or any setting where the
system responds differently to the same input later than it did earlier.

The model discounts past observations by ``alpha ** |t - t'|``, where ``alpha``
is fitted from the data by marginal likelihood, so the optimiser infers how
fast the system is changing instead of assuming it is stationary.

See PROVENANCE.md for the method's origin and the papers to cite.
"""

from .kernels import TemporalDecayKernel
from .mo_optimizer import DynamicMOBO, MODBOConfig, MOObservation, as_stationary_mo
from .model import DBOModelConfig, build_model, fit_model, get_alpha, posterior_mean_std
from .optimizer import DBOConfig, DynamicBO, Observation, as_stationary

__version__ = "0.1.0"

__all__ = [
    "DynamicBO",
    "DynamicMOBO",
    "DBOConfig",
    "DBOModelConfig",
    "MODBOConfig",
    "MOObservation",
    "Observation",
    "TemporalDecayKernel",
    "as_stationary",
    "as_stationary_mo",
    "build_model",
    "fit_model",
    "get_alpha",
    "posterior_mean_std",
    "__version__",
]
