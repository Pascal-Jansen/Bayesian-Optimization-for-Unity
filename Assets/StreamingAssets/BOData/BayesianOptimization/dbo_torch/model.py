"""Gaussian process model for Dynamic Bayesian Optimization.

The model is an ordinary single-task GP whose covariance is the product of a
spatial factor over the control parameters and a temporal decay factor over an
appended time column. All hyperparameters, including the temporal decay rate
``alpha``, are fitted jointly by maximising the marginal log likelihood.

The time column is deliberately left out of input normalisation: ``alpha`` is
defined as a per-unit-time decay, so rescaling time would silently rescale the
meaning of the fitted value and break comparability with published results.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from .kernels import TemporalDecayKernel

__all__ = ["DBOModelConfig", "build_model", "fit_model"]


@dataclass
class DBOModelConfig:
    """Model-level settings.

    The defaults follow current BoTorch practice. For numerical comparison
    against the reference MATLAB implementation use
    :meth:`DBOModelConfig.matlab_compatible`, which disables the transforms
    MATLAB does not apply and selects the squared-exponential spatial kernel
    the reference DBO kernel uses.
    """

    #: Spatial covariance: ``"rbf"`` (squared exponential) or ``"matern52"``.
    spatial_kernel: str = "rbf"

    #: How ``alpha`` is presented to the fitting routine. See
    #: :class:`~dbo_torch.kernels.TemporalDecayKernel`.
    alpha_parameterization: str = "decay"

    #: Starting value for ``alpha``.
    initial_alpha: float = 0.99

    #: Scale control parameters to the unit cube before fitting. Never applied
    #: to the time column.
    normalize_inputs: bool = True

    #: Centre and scale observed costs before fitting.
    standardize_outcome: bool = True

    #: Lower bound on the observation noise *standard deviation*.
    noise_lower_bound: float = 1e-4

    #: Set ``alpha = 1`` and freeze it, reducing the model to stationary BO.
    #: This is how the BO baseline in the paper is reproduced.
    stationary: bool = False

    @classmethod
    def matlab_compatible(cls, **overrides) -> DBOModelConfig:
        """Settings that mirror the reference MATLAB implementation."""
        base = dict(
            spatial_kernel="rbf",
            alpha_parameterization="decay",
            initial_alpha=0.99,
            normalize_inputs=False,
            standardize_outcome=False,
        )
        base.update(overrides)
        return cls(**base)


def build_model(
    train_X: Tensor,
    train_Y: Tensor,
    config: DBOModelConfig | None = None,
) -> SingleTaskGP:
    """Construct the DBO GP.

    Parameters
    ----------
    train_X:
        ``(n, d + 1)`` tensor. The first ``d`` columns are control parameters;
        the final column is the time coordinate.
    train_Y:
        ``(n, 1)`` tensor of observed costs.
    config:
        Model settings. Defaults to :class:`DBOModelConfig`.
    """
    config = config or DBOModelConfig()

    if train_X.dim() != 2:
        raise ValueError(f"train_X must be 2-dimensional, got shape {tuple(train_X.shape)}")
    if train_X.size(-1) < 2:
        raise ValueError(
            "train_X needs at least two columns: one control parameter and one "
            f"time column. Got {train_X.size(-1)}."
        )
    if train_Y.dim() != 2 or train_Y.size(-1) != 1:
        raise ValueError(f"train_Y must have shape (n, 1), got {tuple(train_Y.shape)}")

    n_total = train_X.size(-1)
    d = n_total - 1
    spatial_dims = list(range(d))
    time_dim = d

    if config.spatial_kernel == "rbf":
        base = RBFKernel(ard_num_dims=d, active_dims=spatial_dims)
    elif config.spatial_kernel == "matern52":
        base = MaternKernel(nu=2.5, ard_num_dims=d, active_dims=spatial_dims)
    else:
        raise ValueError(
            f"spatial_kernel must be 'rbf' or 'matern52', got {config.spatial_kernel!r}"
        )

    covar_module = ScaleKernel(base)

    if not config.stationary:
        covar_module = covar_module * TemporalDecayKernel(
            parameterization=config.alpha_parameterization,
            initial_alpha=config.initial_alpha,
            active_dims=[time_dim],
        )

    # Noise is a variance, the configured bound is a standard deviation.
    likelihood = GaussianLikelihood(
        noise_constraint=GreaterThan(config.noise_lower_bound**2)
    )

    input_transform = (
        Normalize(d=n_total, indices=spatial_dims) if config.normalize_inputs else None
    )
    outcome_transform = Standardize(m=1) if config.standardize_outcome else None

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    if config.stationary:
        model._dbo_temporal_kernel = None
    else:
        model._dbo_temporal_kernel = covar_module.kernels[1]

    return model.to(train_X)


def get_alpha(model: SingleTaskGP) -> float:
    """Fitted temporal decay rate, or ``1.0`` for a stationary model.

    Works for models built by :func:`build_model` (which tag their temporal
    kernel) and, as a fallback, for any model whose covariance contains a
    :class:`TemporalDecayKernel` somewhere in its composition.
    """
    kernel = getattr(model, "_dbo_temporal_kernel", None)
    if kernel is None:
        kernel = next(
            (m for m in model.covar_module.modules() if isinstance(m, TemporalDecayKernel)),
            None,
        )
    if kernel is None:
        return 1.0
    return float(kernel.alpha.detach().reshape(-1)[0])


def fit_model(
    model: SingleTaskGP,
    max_attempts: int = 10,
    noise_growth: float = 2.0,
    alpha_jitter: float = 1.5,
) -> SingleTaskGP:
    """Fit hyperparameters by maximising the marginal log likelihood.

    GP fitting on human-in-the-loop data fails intermittently: the cost
    function in the source paper has a kink at the optimum, which makes the
    likelihood surface awkward and the covariance matrix occasionally
    ill-conditioned. On failure this retries with a raised noise floor and a
    perturbed starting ``alpha``, mirroring the reference implementation's
    recovery loop.

    Returns the model with fitted hyperparameters. If every attempt fails, the
    model is returned with its last usable hyperparameters and a warning is
    issued rather than raising, so a running study is never halted by a single
    bad iteration.
    """
    kernel = getattr(model, "_dbo_temporal_kernel", None)
    state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_gpytorch_mll(mll)
            return model
        except Exception as exc:  # noqa: BLE001 - recovery is the whole point
            last_error = exc

            model.load_state_dict(state_before)

            # Raise the noise floor, which is what usually rescues a failed
            # Cholesky, and nudge alpha away from wherever it stuck.
            constraint = model.likelihood.noise_covar.raw_noise_constraint
            new_floor = float(constraint.lower_bound) * (noise_growth ** (attempt + 1))
            model.likelihood.noise_covar.register_constraint(
                "raw_noise", GreaterThan(new_floor)
            )
            model.likelihood.noise = max(new_floor * 2.0, 1e-8)

            if kernel is not None:
                decay = 1.0 - float(kernel.alpha.detach().reshape(-1)[0])
                decay = min(max(decay, 1e-6) * (alpha_jitter ** (attempt + 1)), 0.5)
                kernel.alpha = 1.0 - decay

    warnings.warn(
        f"GP fitting did not converge after {max_attempts} attempts; "
        f"continuing with unfitted hyperparameters. Last error: {last_error}",
        RuntimeWarning,
        stacklevel=2,
    )
    return model


def posterior_mean_std(model: SingleTaskGP, X: Tensor) -> tuple[Tensor, Tensor]:
    """Posterior mean and standard deviation at ``X`` (time column included)."""
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X)
        return posterior.mean.squeeze(-1), posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
