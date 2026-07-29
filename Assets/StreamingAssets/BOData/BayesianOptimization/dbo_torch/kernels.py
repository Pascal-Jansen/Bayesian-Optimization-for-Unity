"""Covariance functions for Dynamic Bayesian Optimization.

DBO models a cost function that drifts over time by making the covariance
between two observations decay with how far apart in time they were taken.
The covariance is separable into a spatial and a temporal factor:

    k((u, t), (u', t')) = k_u(u, u') * alpha ** |t - t'|

with ``alpha`` in (0, 1]. At ``alpha = 1`` the temporal factor is identically
one and the model collapses to ordinary stationary BO; as alpha falls, older
observations are progressively discounted.

``alpha`` is fitted from data by maximising the marginal likelihood alongside
the other hyperparameters, so the optimiser infers the drift rate rather than
being told it.
"""

from __future__ import annotations

import torch
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from torch import Tensor

__all__ = ["TemporalDecayKernel"]

# Floor applied to alpha, mirroring the reference implementation. Keeps the
# temporal factor from collapsing to exactly zero, which would decouple every
# observation from every other and make the posterior degenerate.
_ALPHA_FLOOR = 1e-9


class TemporalDecayKernel(Kernel):
    r"""Temporal factor of the DBO covariance: :math:`k_t(t, t') = \alpha^{|t-t'|}`.

    This kernel acts on a single input dimension holding the time coordinate.
    Use it as the right-hand factor of a product kernel, with ``active_dims``
    pointing at the time column::

        spatial = ScaleKernel(RBFKernel(ard_num_dims=d, active_dims=range(d)))
        temporal = TemporalDecayKernel(active_dims=(d,))
        covar_module = spatial * temporal

    Parameters
    ----------
    parameterization:
        How ``alpha`` is represented to the optimiser.

        ``"decay"`` (default) fits a positive decay rate ``p`` in log space and
        recovers ``alpha = 1 - p``, clamped to ``(0, 1]``. This reproduces the
        reference MATLAB implementation, including its log-space transform and
        its clamping, and is the setting to use when comparing against it.

        ``"direct"`` fits ``alpha`` itself under an interval constraint. The
        likelihood surface is the same, but the optimiser sees a different
        geometry, so fitted values can differ slightly. It is better behaved
        when alpha is expected to be small (fast drift), because ``"decay"``
        pushes alpha through a clamp in that regime.
    initial_alpha:
        Starting value for alpha. Defaults to ``0.99``, matching the reference
        implementation's initial decay rate of ``0.01``.
    alpha_constraint:
        Overrides the default constraint on the fitted parameter.
    alpha_prior:
        Optional prior on alpha. Applied to the natural (post-transform) value.
    """

    has_lengthscale = False

    def __init__(
        self,
        parameterization: str = "decay",
        initial_alpha: float = 0.99,
        alpha_constraint=None,
        alpha_prior: Prior | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if parameterization not in ("decay", "direct"):
            raise ValueError(
                f"parameterization must be 'decay' or 'direct', got {parameterization!r}"
            )
        if not 0.0 < initial_alpha <= 1.0:
            raise ValueError(f"initial_alpha must lie in (0, 1], got {initial_alpha}")

        self.parameterization = parameterization

        if alpha_constraint is None:
            if parameterization == "decay":
                # exp/log rather than the GPyTorch default softplus, so the
                # search geometry matches the reference implementation.
                alpha_constraint = GreaterThan(
                    0.0, transform=torch.exp, inv_transform=torch.log
                )
            else:
                alpha_constraint = Interval(_ALPHA_FLOOR, 1.0)

        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )
        self.register_constraint("raw_alpha", alpha_constraint)

        if alpha_prior is not None:
            # gpytorch >= 1.4 requires prior closures that take the module as
            # their first argument; bound zero-arg methods are rejected.
            self.register_prior(
                "alpha_prior",
                alpha_prior,
                lambda m: m.alpha,
                lambda m, v: TemporalDecayKernel.alpha.fset(m, v),
            )

        self.alpha = initial_alpha

    # -- alpha as a natural-scale property ------------------------------

    @property
    def alpha(self) -> Tensor:
        """Time-scale hyperparameter, in ``(0, 1]``."""
        raw = self.raw_alpha_constraint.transform(self.raw_alpha)
        if self.parameterization == "decay":
            return (1.0 - raw).clamp(min=_ALPHA_FLOOR, max=1.0)
        return raw

    @alpha.setter
    def alpha(self, value) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=self.raw_alpha.dtype)
        value = value.to(self.raw_alpha)
        # Accept a (batch,)-shaped value for a (batch, 1) parameter, as stock
        # gpytorch kernels do for their hyperparameters.
        if value.numel() == self.raw_alpha.numel():
            value = value.reshape(self.raw_alpha.shape)

        # Invert the natural-scale value back through the parameterisation.
        raw_natural = (1.0 - value) if self.parameterization == "decay" else value
        if self.parameterization == "decay":
            # alpha == 1 means "no decay", i.e. a decay rate of exactly zero,
            # which has no finite preimage under log. Nudge it inside the domain.
            raw_natural = raw_natural.clamp(min=torch.finfo(raw_natural.dtype).tiny)
        else:
            # The Interval constraint's inverse transform is +/-inf exactly at
            # its boundaries, which would leave a non-finite raw parameter and
            # NaN gradients. Keep strictly inside the open interval.
            eps = torch.finfo(raw_natural.dtype).eps
            width = 1.0 - _ALPHA_FLOOR
            raw_natural = raw_natural.clamp(
                min=_ALPHA_FLOOR + eps * width, max=1.0 - eps * width
            )

        self.initialize(
            raw_alpha=self.raw_alpha_constraint.inverse_transform(
                raw_natural.expand_as(self.raw_alpha)
            )
        )

    # -- covariance -----------------------------------------------------

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        if last_dim_is_batch:
            raise RuntimeError(
                "TemporalDecayKernel does not support last_dim_is_batch; it "
                "operates on a single time dimension."
            )
        if x1.size(-1) != 1 or x2.size(-1) != 1:
            raise RuntimeError(
                "TemporalDecayKernel expects exactly one input dimension (time). "
                f"Got {x1.size(-1)} and {x2.size(-1)}. Set active_dims to the "
                "index of the time column."
            )

        # raw_alpha has shape (*batch_shape, 1), which already broadcasts against
        # a (*batch_shape, n) diagonal. The full (*batch_shape, n, m) matrix
        # needs one more trailing dimension.
        alpha = self.alpha

        if diag:
            # |t - t| == 0, so the factor is alpha ** 0 == 1 everywhere.
            lag = torch.zeros(
                *torch.broadcast_shapes(x1.shape[:-1], x2.shape[:-1]),
                dtype=x1.dtype,
                device=x1.device,
            )
        else:
            alpha = alpha.unsqueeze(-1)
            lag = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).squeeze(-1).abs()

        # alpha ** lag, evaluated as exp(lag * log alpha) for stability at
        # large lags and to keep the gradient well behaved near alpha -> 0.
        return torch.exp(lag * alpha.clamp_min(_ALPHA_FLOOR).log())

    def __repr__(self) -> str:
        values = [f"{v:g}" for v in self.alpha.detach().flatten().tolist()]
        alpha = values[0] if len(values) == 1 else "[" + ", ".join(values) + "]"
        return (
            f"{self.__class__.__name__}("
            f"alpha={alpha}, parameterization={self.parameterization!r})"
        )
