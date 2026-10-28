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

import math
import warnings
from collections.abc import Sequence
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

# The reference implementation never lets the noise standard deviation floor
# fall below this, whatever the scale of the data.
_REFERENCE_MIN_NOISE_SD = 1e-6


@dataclass
class DBOModelConfig:
    """Model-level settings.

    Hyperparameters are fitted by plain maximum likelihood with no priors, as
    in the reference implementation; inputs are normalised and outcomes
    standardised. These are not current BoTorch defaults, which add
    dimension-scaled lengthscale priors — for a baseline, matching the
    reference matters more. For numerical comparison against the reference
    MATLAB implementation use :meth:`DBOModelConfig.matlab_compatible`, which
    also disables the transforms and starts fitting where the reference does.
    """

    #: Spatial covariance: ``"rbf"`` (squared exponential) or ``"matern52"``.
    spatial_kernel: str = "rbf"

    #: How ``alpha`` is presented to the fitting routine. See
    #: :class:`~dbo_torch.kernels.TemporalDecayKernel`.
    alpha_parameterization: str = "decay"

    #: Starting value for ``alpha``, in ``(0, 1)``.
    initial_alpha: float = 0.99

    #: Scale control parameters to the unit cube before fitting. Never applied
    #: to the time column. When :func:`build_model` is given the domain bounds
    #: (the optimisers always pass them) the domain maps to the unit cube;
    #: otherwise the range is learned from the training inputs.
    normalize_inputs: bool = True

    #: Centre and scale observed costs before fitting.
    standardize_outcome: bool = True

    #: Lower bound on the observation noise *standard deviation*.
    noise_lower_bound: float = 1e-4

    #: Set ``alpha = 1`` and freeze it, reducing the model to stationary BO.
    #: This is how the BO baseline in the paper is reproduced.
    stationary: bool = False

    #: If set, the lower bound on the noise standard deviation is this
    #: fraction of the outcome standard deviation (never below 1e-6), and
    #: replaces ``noise_lower_bound``. The reference implementation uses 0.01.
    noise_floor_fraction: float | None = None

    #: Start hyperparameter fitting from the reference implementation's
    #: data-dependent values instead of GPyTorch's defaults: lengthscales of
    #: half the domain width, and signal and noise standard deviations of
    #: ``std(Y) / sqrt(2)``. With no priors and few observations the
    #: likelihood has many local optima, so the starting point matters.
    reference_init: bool = False

    #: Fit once from the starting lengthscales scaled by each of these factors
    #: and keep the fit with the highest marginal likelihood. With few
    #: observations the likelihood has several optima, typically one that
    #: attributes the variation to the signal and one that attributes it to
    #: noise, and a single start can settle in the worse one: on RA-L-like
    #: data at 10-30 observations, one start from the default lengthscale fell
    #: short of the best optimum in up to half of the data sets, while these
    #: three starts reached it in all of them. Each factor costs one fit.
    #: ``(1.0,)`` fits once, as the reference does.
    lengthscale_starts: tuple[float, ...] = (1.0, 0.5, 0.25)

    @classmethod
    def matlab_compatible(cls, **overrides) -> DBOModelConfig:
        """Settings that mirror the reference MATLAB implementation."""
        base = dict(
            spatial_kernel="rbf",
            alpha_parameterization="decay",
            initial_alpha=0.99,
            normalize_inputs=False,
            standardize_outcome=False,
            noise_floor_fraction=0.01,
            reference_init=True,
            lengthscale_starts=(1.0,),
        )
        base.update(overrides)
        return cls(**base)


class _DBOSingleTaskGP(SingleTaskGP):
    """A SingleTaskGP whose unnormalised time column is exempt from BoTorch's
    unit-cube input check.

    Time is left unnormalised on purpose (see the module docstring), so the
    check would otherwise warn on every model build. This is the hook BoTorch's
    own ``MixedSingleTaskGP`` uses for its categorical dimensions.
    """

    def __init__(self, *args, time_dim: int, **kwargs) -> None:
        self._ignore_X_dims_scaling_check = [time_dim]
        super().__init__(*args, **kwargs)


def build_model(
    train_X: Tensor,
    train_Y: Tensor,
    config: DBOModelConfig | None = None,
    bounds: Tensor | None = None,
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
    bounds:
        Optional ``(2, d)`` tensor of lower and upper bounds on the control
        parameters. Input normalisation then maps this domain onto the unit
        cube, so a fitted lengthscale means the same thing at every iteration.
        Without it the normalising range is learned from ``train_X`` and
        shifts as data arrive.
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

    if bounds is not None:
        bounds = torch.as_tensor(bounds).to(train_X)
        if bounds.shape != (2, d):
            raise ValueError(f"bounds must have shape (2, {d}), got {tuple(bounds.shape)}")

    if config.spatial_kernel == "rbf":
        base = RBFKernel(ard_num_dims=d, active_dims=spatial_dims)
    elif config.spatial_kernel == "matern52":
        base = MaternKernel(nu=2.5, ard_num_dims=d, active_dims=spatial_dims)
    else:
        raise ValueError(
            f"spatial_kernel must be 'rbf' or 'matern52', got {config.spatial_kernel!r}"
        )

    scale = ScaleKernel(base)
    covar_module = scale
    if not config.stationary:
        covar_module = scale * TemporalDecayKernel(
            parameterization=config.alpha_parameterization,
            initial_alpha=config.initial_alpha,
            active_dims=[time_dim],
        )

    # Outcome scale in the units the GP is fitted in: Standardize leaves unit
    # standard deviation unless the data are constant.
    raw_sd = float(train_Y.std()) if train_Y.size(0) > 1 else float("nan")
    if config.standardize_outcome:
        outcome_sd = 1.0 if raw_sd >= 1e-8 else 0.0
    else:
        outcome_sd = raw_sd

    if config.noise_floor_fraction is not None:
        floor_sd = config.noise_floor_fraction * outcome_sd
        if not math.isfinite(floor_sd):
            floor_sd = 0.0
        floor_sd = max(floor_sd, _REFERENCE_MIN_NOISE_SD)
    else:
        floor_sd = config.noise_lower_bound

    # Noise is a variance, the floor is a standard deviation.
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(floor_sd**2))

    input_transform = None
    if config.normalize_inputs:
        input_transform = Normalize(d=n_total, indices=spatial_dims, bounds=bounds)
    outcome_transform = Standardize(m=1) if config.standardize_outcome else None

    model = _DBOSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        time_dim=time_dim,
    )

    if config.reference_init:
        # Half the domain width, as the reference does; in normalised
        # coordinates the domain is the unit cube, so that is 0.5.
        if config.normalize_inputs:
            lengthscale = torch.full((d,), 0.5)
        else:
            if bounds is not None:
                width = bounds[1] - bounds[0]
            else:
                spatial = train_X[:, :d]
                width = spatial.max(dim=0).values - spatial.min(dim=0).values
            lengthscale = torch.where(width > 0, width / 2, torch.ones_like(width))
        base.lengthscale = lengthscale.to(train_X).reshape(1, d)

        sigma_f = outcome_sd / math.sqrt(2.0)
        if not math.isfinite(sigma_f) or sigma_f == 0.0:
            sigma_f = 1.0
        scale.outputscale = torch.tensor(sigma_f**2).to(train_X)
        likelihood.noise = torch.tensor(max(sigma_f**2, 4.0 * floor_sd**2)).to(train_X)

    return model.to(train_X)


def _temporal_kernel(model) -> TemporalDecayKernel | None:
    """The model's :class:`TemporalDecayKernel`, or None for a stationary model."""
    covar = getattr(model, "covar_module", None)
    if covar is None:
        return None
    return next((m for m in covar.modules() if isinstance(m, TemporalDecayKernel)), None)


def get_alpha(model: SingleTaskGP) -> float:
    """Fitted temporal decay rate, or ``1.0`` for a stationary model.

    Works for any model whose covariance contains a
    :class:`TemporalDecayKernel` somewhere in its composition, not only those
    built by :func:`build_model`.
    """
    kernel = _temporal_kernel(model)
    if kernel is None:
        return 1.0
    return float(kernel.alpha.detach().reshape(-1)[0])


def _spatial_kernel(model):
    """The kernel carrying the control-parameter lengthscales, if any."""
    covar = getattr(model, "covar_module", None)
    if covar is None:
        return None
    return next((m for m in covar.modules() if isinstance(m, (RBFKernel, MaternKernel))), None)


def _state(model) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _total_log_likelihood(model) -> float:
    """Exact log marginal likelihood of the training data, summed, not averaged."""
    was_training = model.training
    model.train()
    try:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with torch.no_grad():
            value = mll(model(*model.train_inputs), model.train_targets)
    finally:
        if not was_training:
            model.eval()
    return float(value) * model.train_targets.numel()


def fit_model(
    model: SingleTaskGP,
    max_attempts: int = 10,
    noise_growth: float = 2.0,
    alpha_jitter: float = 1.5,
    lengthscale_starts: Sequence[float] = (1.0,),
) -> SingleTaskGP:
    """Fit hyperparameters by maximising the marginal log likelihood.

    With several ``lengthscale_starts``, fits once per factor, each time
    starting from the model's current hyperparameters with the lengthscales
    scaled by that factor, and keeps the fit with the highest marginal
    likelihood. The default fits once, from the current hyperparameters.

    GP fitting on human-in-the-loop data fails intermittently: the cost
    function in the source paper has a kink at the optimum, which makes the
    likelihood surface awkward and the covariance matrix occasionally
    ill-conditioned. On failure each fit retries from its starting
    hyperparameters with the noise floor doubled per attempt and a perturbed
    starting ``alpha``, mirroring the reference implementation's recovery loop.

    Returns the model with fitted hyperparameters. If every attempt fails, the
    model is returned at its starting hyperparameters, with the final
    attempt's raised noise floor, and a warning is issued rather than raising,
    so a running study is never halted by a single bad iteration.
    """
    starts = [float(f) for f in lengthscale_starts]
    if not starts or not all(f > 0 for f in starts):
        raise ValueError(f"lengthscale_starts must be positive factors, got {starts}")

    spatial = _spatial_kernel(model)
    if spatial is None or len(starts) == 1:
        if spatial is not None and starts[0] != 1.0:
            spatial.lengthscale = spatial.lengthscale.detach() * starts[0]
        return _fit_once(model, max_attempts, noise_growth, alpha_jitter)

    initial = _state(model)
    lengthscale = spatial.lengthscale.detach().clone()
    best_value, best_state = -math.inf, None
    for factor in starts:
        model.load_state_dict(initial)
        spatial.lengthscale = lengthscale * factor
        _fit_once(model, max_attempts, noise_growth, alpha_jitter)
        value = _total_log_likelihood(model)
        if value > best_value:
            best_value, best_state = value, _state(model)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _fit_once(
    model: SingleTaskGP, max_attempts: int, noise_growth: float, alpha_jitter: float
) -> SingleTaskGP:
    """One maximum-likelihood fit from the current hyperparameters, with recovery."""
    kernel = _temporal_kernel(model)
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

            # Restoring the state also restores the original noise floor, so
            # the floor below grows by noise_growth per attempt, not compounded.
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
        stacklevel=3,
    )
    return model


def posterior_mean_std(model: SingleTaskGP, X: Tensor) -> tuple[Tensor, Tensor]:
    """Posterior mean and standard deviation at ``X`` (time column included)."""
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X)
        return posterior.mean.squeeze(-1), posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
