"""Dynamic Bayesian Optimization driver.

Implements the ask/tell loop used in the human-in-the-loop studies this package
reproduces, including *validation iterations*: periodic steps at which the
optimiser applies its current best estimate of the optimum instead of an
acquisition-driven candidate.

Validation iterations exist to make optimisers comparable. Two optimisers with
different exploration behaviour will visit different parts of the space, so
comparing the cost they happen to incur during ordinary iterations conflates
model quality with exploration policy. Testing the best estimate removes that
confound.

Basic use::

    opt = DynamicBO(
        bounds=[(-5.0, 9.0)],
        config=DBOConfig(seed_points=[[5.0], [7.0], [3.0]]),
    )
    for _ in range(80):
        x = opt.suggest()
        opt.observe(x, measure_cost(x))
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import torch
from botorch.acquisition import LogExpectedImprovement, PosteriorMean
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from ._base import DynamicOptimizerBase, _z_score
from .model import DBOModelConfig, get_alpha, posterior_mean_std

__all__ = ["DBOConfig", "DynamicBO", "Observation"]


@dataclass
class Observation:
    """One completed iteration."""

    iteration: int
    x: list[float]
    y: float
    time: float
    is_validation: bool = False
    #: Cost the model predicted before the point was evaluated, when known,
    #: at the time the measurement was taken.
    predicted_y: float | None = None
    predicted_sd: float | None = None

    def as_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "x": self.x,
            "y": self.y,
            "time": self.time,
            "is_validation": self.is_validation,
            "predicted_y": self.predicted_y,
            "predicted_sd": self.predicted_sd,
        }


@dataclass
class DBOConfig:
    """Optimiser settings."""

    #: Model configuration.
    model: DBOModelConfig = field(default_factory=DBOModelConfig)

    #: Fixed inputs applied at the first iterations, before any model is fitted.
    #: The source study used ``[[5.0], [7.0], [3.0]]``.
    seed_points: list[list[float]] | None = None

    #: Number of random seed points, used only when ``seed_points`` is None.
    num_seed_points: int = 3

    #: Run a validation iteration every N iterations. ``None`` disables
    #: automatic scheduling; call :meth:`DynamicBO.suggest_validation` directly.
    validation_every: int | None = None

    #: Guards against the search collapsing onto its current best guess. After
    #: the acquisition function picks a point, if the model's uncertainty there
    #: is below this multiple of the observation noise, the signal variance is
    #: temporarily inflated and the search repeated. Larger values explore
    #: more. The source study used 0.1, favouring exploitation. Set to 0 to
    #: disable the check and use plain Expected Improvement. This follows the
    #: stock "plus" behaviour; the patched reference file differs here (see
    #: the README's notes on the reference implementation).
    exploration_ratio: float = 0.1

    #: How many times the over-exploitation guard may re-search in one
    #: iteration before giving up and accepting the point.
    max_exploit_iterations: int = 5

    #: Tail probability for the validation-iteration bound. The default of 0.01
    #: matches the reference implementation and gives a multiplier of ~2.33 on
    #: the posterior standard deviation. Note this is the tail mass, not the
    #: confidence level: pass 0.01, not 0.99. 0.5 selects by posterior mean.
    validation_confidence: float = 0.01

    #: Restrict validation candidates to already-evaluated inputs. True matches
    #: the reference implementation, whose default best-point criterion is
    #: "min-visited-upper-confidence-interval", and is the setting to use when
    #: reproducing the published results. Setting it False searches the
    #: continuous domain instead, which tracks a drifting optimum better,
    #: because once the optimum has moved between two sampled inputs the best
    #: currently attainable input is one nobody has tried.
    validation_visited_only: bool = True

    #: Time value at which the acquisition function is evaluated, relative to
    #: the most recent observation. ``0`` reproduces the reference
    #: implementation, which scores candidates at the *current* time. ``1``
    #: scores them at the time they will actually be evaluated, which removes a
    #: one-step lag when drift is fast. Recorded predictions are always made
    #: at the time of evaluation, whatever this is set to.
    acquisition_time_offset: float = 0.0

    #: Restarts and raw samples for acquisition optimisation.
    num_restarts: int = 20
    raw_samples: int = 512

    #: Refit hyperparameters every N iterations. 1 refits every iteration.
    refit_every: int = 1

    #: Start each refit from the previous fit's hyperparameters rather than
    #: from the configured starting values. Faster and steadier from one
    #: iteration to the next, but departs from the reference implementation,
    #: which restarts every fit from the same values.
    warm_start: bool = False

    seed: int | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


class DynamicBO(DynamicOptimizerBase):
    """Dynamic Bayesian optimiser for a drifting, noisily observed cost function.

    The optimiser minimises cost. Time is measured in iterations: the first
    observation is at time 1, the second at time 2, and so on. The temporal
    kernel discounts observations by ``alpha ** |t - t'|``, with ``alpha``
    fitted from the data, so the model learns how fast the system is drifting
    rather than being told.

    A run can be saved with :meth:`save` and resumed with :meth:`load`.
    """

    _observation_type = Observation
    _config_type = DBOConfig

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        config: DBOConfig | None = None,
    ) -> None:
        super().__init__(bounds, config or DBOConfig())
        self._model = None

    # -- state ----------------------------------------------------------

    @property
    def alpha(self) -> float | None:
        """Fitted temporal decay rate, or None before the first fit."""
        return None if self._model is None else get_alpha(self._model)

    def _train_data(self) -> tuple[Tensor, Tensor]:
        Y = torch.tensor([[obs.y] for obs in self.observations], **self._tkwargs)
        return self._train_X(), Y

    def _ensure_model(self):
        """Fit or refresh the GP if needed. Returns None if data is insufficient."""
        if self.num_observations < 2:
            return None
        if self._model is None or self._model_stale:
            X, Y = self._train_data()
            self._model = self._refresh_model(X, Y, self._model)
            self._model_stale = False
        return self._model

    # -- ask ------------------------------------------------------------

    def suggest(self) -> list[float]:
        """Return the next input to evaluate.

        Produces a seed point while seeding, a best-estimate input on a
        scheduled validation iteration, and an acquisition-driven candidate
        otherwise.
        """
        with self._isolated_rng():
            return self._suggest()

    def _suggest(self) -> list[float]:
        if self.is_validation_iteration():
            return self._suggest_validation()

        x = self._next_seed_point()
        if x is not None:
            self._set_pending(x, is_validation=False)
            return x

        model = self._ensure_model()
        if model is None:
            x = self._random_point()
            self._set_pending(x, is_validation=False)
            return x

        x = self._optimize_acquisition(model)
        self._set_pending(x, is_validation=False, prediction=self._predict_at(model, x))
        return x

    def _optimize_acquisition(self, model) -> list[float]:
        """Maximise Expected Improvement at a fixed point in time.

        If the chosen point turns out to be one the model is already confident
        about, the search is repeated against a temporarily inflated signal
        variance. See :meth:`_exploiting_too_much`.
        """
        t_acq = self._acquisition_time()
        # Computed once: the incumbent is a property of the fitted model and
        # stays fixed while the guard below re-searches, as in the reference.
        incumbent = self._incumbent(model, t_acq)
        x = self._argmax_ei(model, t_acq, incumbent)

        if self.config.exploration_ratio <= 0 or self.config.max_exploit_iterations <= 0:
            return x

        # Guard against over-exploitation, following the "plus" variant of
        # expected improvement. Inflating the signal variance raises the
        # posterior uncertainty everywhere, which pushes the acquisition
        # function back towards unexplored regions. The inflation is discarded
        # afterwards, so it changes which point is picked and nothing else.
        scale_kernel = self._scale_kernel(model)
        if scale_kernel is None:
            return x

        original = scale_kernel.outputscale.detach().clone()
        try:
            for attempt in range(1, self.config.max_exploit_iterations + 1):
                if not self._exploiting_too_much(model, x, t_acq):
                    break
                factor = max(self.num_observations, 1) * (10.0 ** (attempt - 1))
                scale_kernel.outputscale = original * factor
                _clear_gp_caches(model)
                x = self._argmax_ei(model, t_acq, incumbent)
        finally:
            scale_kernel.outputscale = original
            _clear_gp_caches(model)

        return x

    @staticmethod
    def _scale_kernel(model):
        """The ScaleKernel carrying the signal variance, if the model has one."""
        covar = model.covar_module
        candidates = getattr(covar, "kernels", [covar])
        for kernel in candidates:
            if hasattr(kernel, "outputscale"):
                return kernel
        return None

    def _exploiting_too_much(self, model, x: Sequence[float], t: float) -> bool:
        """Is the model already confident about this point?

        Compares the latent-function standard deviation at the candidate
        against the observation noise. If the model's uncertainty about the
        function is small relative to the noise it expects from measuring it,
        evaluating there buys almost nothing and the search has collapsed onto
        its current best guess.

        Both quantities are taken from the posterior rather than reading
        ``likelihood.noise`` directly. When an outcome transform is active the
        likelihood's noise is in standardised units while the posterior is in
        the original ones, and comparing the two would be meaningless. The
        difference between the noisy and noiseless posterior variances gives
        the noise in the same units as the signal, whatever transforms are in
        play.
        """
        point = torch.tensor([list(x) + [t]], **self._tkwargs)

        model.eval()
        with torch.no_grad():
            latent = float(model.posterior(point).variance.reshape(-1)[0])
            noisy = float(
                model.posterior(point, observation_noise=True).variance.reshape(-1)[0]
            )

        signal_sd = max(latent, 0.0) ** 0.5
        noise_sd = max(noisy - latent, 0.0) ** 0.5

        return signal_sd < self.config.exploration_ratio * noise_sd

    def _argmax_ei(self, model, t_acq: float, incumbent: float) -> list[float]:
        # EI improves on the incumbent. The incumbent is the lowest cost the
        # model believes is currently *attainable*, not the lowest ever
        # measured: under drift an old low cost may be unreachable now, and
        # using it would flatten the acquisition function everywhere.
        acqf = LogExpectedImprovement(model=model, best_f=incumbent, maximize=False)
        x, _ = self._maximize(acqf, t_acq)
        return x

    def _incumbent(self, model, t: float) -> float:
        """Lowest posterior-mean cost attainable anywhere in the domain, now.

        Searched over the continuous domain rather than over evaluated points
        only: when the optimum has drifted between the sampled inputs, the best
        attainable cost is generally somewhere nobody has tried. The visited
        inputs, re-scored at ``t``, are kept as a floor on the search.
        """
        _, value = self._maximize(PosteriorMean(model, maximize=False), t)
        visited, _ = posterior_mean_std(model, self._visited_at(t))
        return min(-value, float(visited.min()))

    def _predict_at(self, model, x: Sequence[float], t: float | None = None):
        """Posterior mean and sd at ``x``, by default when it will be measured."""
        t = self._evaluation_time() if t is None else t
        pt = torch.tensor([list(x) + [t]], **self._tkwargs)
        mu, sd = posterior_mean_std(model, pt)
        return float(mu.reshape(-1)[0]), float(sd.reshape(-1)[0])

    def suggest_validation(self) -> list[float]:
        """Return the optimiser's current best estimate of the optimum.

        Selects the input minimising an upper confidence bound on cost,
        ``mu(u) + k * sd(u)``, evaluated at the acquisition time. Bounding
        from above and then minimising is deliberately risk-averse: it prefers
        an input the model is confident is good over one that is merely
        unexplored.

        Also records what the model expects to measure at the chosen point, at
        the time it will be measured, before it is measured. The gap between
        the two is the point of a validation iteration: it separates how well
        the optimiser understands the system from how lucky its exploration
        has been.
        """
        with self._isolated_rng():
            return self._suggest_validation()

    def _suggest_validation(self) -> list[float]:
        model = self._ensure_model()
        if model is None:
            x = self._fallback_point()
            self._set_pending(x, is_validation=True)
            return x

        x = self._best_estimate(model)
        self._set_pending(x, is_validation=True, prediction=self._predict_at(model, x))
        return x

    def _best_estimate(self, model) -> list[float]:
        t = self._acquisition_time()
        k = _z_score(1.0 - self.config.validation_confidence)

        probe = self._visited_at(t)
        mu, sd = posterior_mean_std(model, probe)
        bound = mu + k * sd
        best = int(torch.argmin(bound))
        visited_best = probe[best, : self.dim].tolist()

        if self.config.validation_visited_only:
            return visited_best

        # Search the continuous domain, keeping the best visited input if the
        # search finds nothing better.
        x, value = self._maximize(_NegatedUpperBound(model, k), t)
        return x if -value < float(bound[best]) else visited_best

    def _fallback_point(self) -> list[float]:
        if self.observations:
            best = min(self.observations, key=lambda o: o.y)
            return list(best.x)
        return ((self.bounds[0] + self.bounds[1]) / 2).tolist()

    # -- tell -----------------------------------------------------------

    def observe(
        self,
        x: Sequence[float],
        y: float,
        is_validation: bool | None = None,
        time: float | None = None,
    ) -> Observation:
        """Record a measured cost for an input.

        The validation flag and prediction recorded for the last suggestion
        are attached only if ``x`` is that suggestion (to within a float32
        round trip).
        """
        x = self._check_x(x)
        if not math.isfinite(y):
            raise ValueError(f"Cost must be finite, got {y}")
        return self._record(x, float(y), is_validation, time)

    # -- reporting ------------------------------------------------------

    def best_observed(self) -> Observation | None:
        return min(self.observations, key=lambda o: o.y) if self.observations else None

    def prediction_error(self) -> list[dict]:
        """Absolute gap between predicted and measured cost, per validation step.

        This is the model-accuracy measure reported in the source paper: it
        isolates how well the optimiser understands the system from how lucky
        its exploration was.
        """
        return [
            {
                "iteration": o.iteration,
                "predicted": o.predicted_y,
                "measured": o.y,
                "error": abs(o.predicted_y - o.y),
            }
            for o in self.observations
            if o.is_validation and o.predicted_y is not None
        ]

    def _summary(self) -> dict:
        return {"alpha": self.alpha}

    def __repr__(self) -> str:
        a = self.alpha
        return (
            f"DynamicBO(dim={self.dim}, n={self.num_observations}, "
            f"alpha={'unfitted' if a is None else f'{a:.4f}'})"
        )


def as_stationary(config: DBOConfig) -> DBOConfig:
    """Return a copy configured as plain stationary BO, for use as a baseline."""
    seeds = (
        None
        if config.seed_points is None
        # dataclasses.replace is shallow; copy so the baseline and the DBO run
        # cannot mutate each other's seed schedule.
        else [list(point) for point in config.seed_points]
    )
    return replace(config, model=replace(config.model, stationary=True), seed_points=seeds)


class _NegatedUpperBound(AnalyticAcquisitionFunction):
    """``-(mu + k * sd)``: maximising it minimises an upper confidence bound on cost.

    BoTorch's UpperConfidenceBound is optimistic in either direction; the
    validation criterion is deliberately pessimistic, hence this small class.
    """

    def __init__(self, model, k: float) -> None:
        super().__init__(model=model)
        self.k = k

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X)
        return -(mean + self.k * sigma).reshape(X.shape[:-2])


def _clear_gp_caches(model) -> None:
    """Invalidate gpytorch's cached prediction strategy.

    Assigning a new value to a kernel hyperparameter does not clear the caches
    an exact GP builds on first posterior evaluation; later posteriors would
    mix the old and new hyperparameters (observed as variances collapsing to
    ~0 during the over-exploitation guard). A train/eval round trip clears
    them and is posterior-neutral when the hyperparameters are unchanged.
    """
    model.train()
    model.eval()
