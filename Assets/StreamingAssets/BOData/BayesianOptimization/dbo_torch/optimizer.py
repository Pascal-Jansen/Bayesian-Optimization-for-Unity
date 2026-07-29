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

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.optim import optimize_acqf
from torch import Tensor

from .model import DBOModelConfig, build_model, fit_model, get_alpha, posterior_mean_std

__all__ = ["DBOConfig", "DynamicBO", "Observation"]


@dataclass
class Observation:
    """One completed iteration."""

    iteration: int
    x: list[float]
    y: float
    time: float
    is_validation: bool = False
    #: Cost the model predicted before the point was evaluated, when known.
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
    #: disable the check and use plain Expected Improvement.
    exploration_ratio: float = 0.1

    #: How many times the over-exploitation guard may re-search in one
    #: iteration before giving up and accepting the point.
    max_exploit_iterations: int = 5

    #: Tail probability for the validation-iteration bound. The default of 0.01
    #: matches the reference implementation and gives a multiplier of ~2.33 on
    #: the posterior standard deviation. Note this is the tail mass, not the
    #: confidence level: pass 0.01, not 0.99.
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
    #: one-step lag when drift is fast.
    acquisition_time_offset: float = 0.0

    #: Restarts and raw samples for acquisition optimisation.
    num_restarts: int = 20
    raw_samples: int = 512

    #: Refit hyperparameters every N iterations. 1 refits every iteration.
    refit_every: int = 1

    seed: int | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


class DynamicBO:
    """Dynamic Bayesian optimiser for a drifting, noisily observed cost function.

    The optimiser minimises cost. Time is measured in iterations: the first
    observation is at time 1, the second at time 2, and so on. The temporal
    kernel discounts observations by ``alpha ** |t - t'|``, with ``alpha``
    fitted from the data, so the model learns how fast the system is drifting
    rather than being told.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        config: DBOConfig | None = None,
    ) -> None:
        self.config = config or DBOConfig()
        self._tkwargs = {"dtype": self.config.dtype, "device": self.config.device}

        bounds_t = torch.tensor(
            [[lo for lo, _ in bounds], [hi for _, hi in bounds]], **self._tkwargs
        )
        if not torch.all(bounds_t[1] > bounds_t[0]):
            raise ValueError(f"Each bound must have hi > lo; got {list(bounds)}")

        self.bounds = bounds_t
        self.dim = bounds_t.size(-1)

        if self.config.seed_points is not None:
            for p in self.config.seed_points:
                if len(p) != self.dim:
                    raise ValueError(
                        f"Each seed point needs {self.dim} values, got {len(p)}: {p}"
                    )

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        self.observations: list[Observation] = []
        self._model = None
        self._model_stale = True
        self._pending: dict | None = None

    # -- state ----------------------------------------------------------

    @property
    def num_observations(self) -> int:
        return len(self.observations)

    @property
    def next_iteration(self) -> int:
        """1-based index of the iteration that :meth:`suggest` will produce."""
        return self.num_observations + 1

    @property
    def alpha(self) -> float | None:
        """Fitted temporal decay rate, or None before the first fit."""
        return None if self._model is None else get_alpha(self._model)

    @property
    def _seed_budget(self) -> int:
        seeds = self.config.seed_points
        return len(seeds) if seeds is not None else self.config.num_seed_points

    def _seeds_used(self) -> int:
        """Seed points consumed so far.

        Counted from non-validation observations rather than a cursor, so
        repeated :meth:`suggest` calls without an intervening observe stay
        idempotent, and a validation iteration can never displace a seed.
        """
        return sum(1 for o in self.observations if not o.is_validation)

    def is_validation_iteration(self, iteration: int | None = None) -> bool:
        every = self.config.validation_every
        if every is None:
            return False
        # No validation while the seed budget is unspent: with no model there
        # is no best estimate worth testing, and scheduling one would silently
        # skip a seed point.
        if self._seeds_used() < self._seed_budget:
            return False
        it = self.next_iteration if iteration is None else iteration
        return it % every == 0

    def _train_data(self) -> tuple[Tensor, Tensor]:
        X = torch.tensor(
            [obs.x + [obs.time] for obs in self.observations], **self._tkwargs
        )
        Y = torch.tensor([[obs.y] for obs in self.observations], **self._tkwargs)
        return X, Y

    def _current_time(self) -> float:
        return float(self.observations[-1].time) if self.observations else 0.0

    def _ensure_model(self):
        """Fit or refresh the GP if needed. Returns None if data is insufficient."""
        if self.num_observations < 2:
            return None

        refit_every = max(1, self.config.refit_every)
        needs_refit = self._model is None or self.num_observations % refit_every == 0

        if self._model is None or self._model_stale:
            X, Y = self._train_data()

            # Rebuilding for new data constructs fresh kernels at factory
            # defaults. On iterations that skip the refit, transplant the last
            # fitted hyperparameters instead of silently discarding them.
            # Transform buffers (Normalize/Standardize statistics) are left to
            # recompute against the new data, so carried-over scales are
            # approximate in the new units — inherent to any warm carry.
            old_state = None
            if self._model is not None and not needs_refit:
                old_state = {
                    k: v
                    for k, v in self._model.state_dict().items()
                    if k.split(".")[0] in ("covar_module", "likelihood", "mean_module")
                }

            self._model = build_model(X, Y, self.config.model)
            if old_state is not None:
                self._model.load_state_dict(old_state, strict=False)
            if needs_refit:
                self._model = fit_model(self._model)
            self._model_stale = False

        return self._model

    # -- ask ------------------------------------------------------------

    def suggest(self) -> list[float]:
        """Return the next input to evaluate.

        Produces a seed point while seeding, a best-estimate input on a
        scheduled validation iteration, and an acquisition-driven candidate
        otherwise.
        """
        if self.is_validation_iteration():
            # suggest_validation records its own pending state, including the
            # model's prediction for the point, so both the scheduled and the
            # directly-called flows behave identically.
            return self.suggest_validation()

        n_used = self._seeds_used()
        if n_used < self._seed_budget:
            seeds = self.config.seed_points
            if seeds is not None:
                x = list(seeds[n_used])
            else:
                lo, hi = self.bounds[0], self.bounds[1]
                x = (lo + (hi - lo) * torch.rand(self.dim, **self._tkwargs)).tolist()
            self._pending = {"x": list(x), "is_validation": False}
            return x

        model = self._ensure_model()
        if model is None:
            lo, hi = self.bounds[0], self.bounds[1]
            x = (lo + (hi - lo) * torch.rand(self.dim, **self._tkwargs)).tolist()
            self._pending = {"x": list(x), "is_validation": False}
            return x

        x = self._optimize_acquisition(model)
        mu, sd = self._predict_at(model, x)
        self._pending = {
            "x": list(x),
            "is_validation": False,
            "predicted_y": mu,
            "predicted_sd": sd,
        }
        return x

    def _acquisition_time(self) -> float:
        return self._current_time() + self.config.acquisition_time_offset

    def _optimize_acquisition(self, model) -> list[float]:
        """Maximise Expected Improvement at a fixed point in time.

        The candidate space is the control parameters only; the time coordinate
        is pinned, since we are choosing what to try *now*, not when to try it.

        If the chosen point turns out to be one the model is already confident
        about, the search is repeated against a temporarily inflated signal
        variance. See :meth:`_exploiting_too_much`.
        """
        t_acq = self._acquisition_time()
        x = self._argmax_ei(model, t_acq)

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
                x = self._argmax_ei(model, t_acq)
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

    def _argmax_ei(self, model, t_acq: float) -> list[float]:
        # EI improves on the incumbent. The incumbent is the lowest cost the
        # model believes is currently *attainable*, not the lowest ever
        # measured: under drift an old low cost may be unreachable now, and
        # using it would flatten the acquisition function everywhere.
        incumbent = self._incumbent(model, t_acq)

        # BoTorch maximises, so negate the outcome for a minimisation problem.
        acqf = LogExpectedImprovement(
            model=model,
            best_f=-incumbent,
            maximize=True,
            posterior_transform=ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], **self._tkwargs)
            ),
        )

        full_bounds = torch.cat(
            [self.bounds, torch.tensor([[t_acq], [t_acq]], **self._tkwargs)], dim=-1
        )

        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=full_bounds,
            q=1,
            num_restarts=self.config.num_restarts,
            raw_samples=self.config.raw_samples,
            fixed_features={self.dim: t_acq},
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidate.squeeze(0)[: self.dim].tolist()

    def _incumbent(self, model, t: float) -> float:
        """Lowest posterior-mean cost attainable anywhere in the domain, now.

        Searched over the continuous domain rather than over evaluated points
        only: when the optimum has drifted between the sampled inputs, the best
        attainable cost is generally somewhere nobody has tried.
        """
        lo, hi = self.bounds[0], self.bounds[1]
        width = hi - lo

        cand = lo + width * torch.rand(
            max(self.config.raw_samples, 256), self.dim, **self._tkwargs
        )
        X, _ = self._train_data()
        cand = torch.cat([cand, X[:, : self.dim]], dim=0)

        probe = torch.cat(
            [cand, torch.full((cand.size(0), 1), t, **self._tkwargs)], dim=-1
        )
        mu, _ = posterior_mean_std(model, probe)

        best = float(mu.min())
        starts = cand[torch.argsort(mu)[: max(1, self.config.num_restarts // 2)]]

        # Polish in unit-cube coordinates so one learning rate suits every
        # dimension regardless of how different the bound widths are.
        def mean_at(z: Tensor) -> Tensor:
            point = torch.cat(
                [lo + z * width, torch.tensor([t], **self._tkwargs)]
            ).unsqueeze(0)
            return model.posterior(point).mean.squeeze()

        for start in starts:
            z = ((start - lo) / width).detach().clone().requires_grad_(True)
            polish = torch.optim.Adam([z], lr=0.05)
            for _ in range(40):
                polish.zero_grad()
                mean_at(z).backward()
                polish.step()
                with torch.no_grad():
                    z.clamp_(0.0, 1.0)
            with torch.no_grad():
                best = min(best, float(mean_at(z)))

        return best

    def _predict_at(self, model, x: Sequence[float], t: float | None = None):
        t = self._acquisition_time() if t is None else t
        pt = torch.tensor([list(x) + [t]], **self._tkwargs)
        mu, sd = posterior_mean_std(model, pt)
        return float(mu.reshape(-1)[0]), float(sd.reshape(-1)[0])

    def suggest_validation(self) -> list[float]:
        """Return the optimiser's current best estimate of the optimum.

        Selects the input minimising an upper confidence bound on cost,
        ``mu(u) + k * sd(u)``, evaluated at the current time. Bounding from
        above and then minimising is deliberately risk-averse: it prefers an
        input the model is confident is good over one that is merely unexplored.

        Also records what the model expects to measure at the chosen point,
        before it is measured. The gap between the two is the point of a
        validation iteration: it separates how well the optimiser understands
        the system from how lucky its exploration has been.
        """
        model = self._ensure_model()
        if model is None:
            x = self._fallback_point()
            self._pending = {"x": list(x), "is_validation": True}
            return x

        x = self._best_estimate(model)
        mu, sd = self._predict_at(model, x)
        self._pending = {
            "x": list(x),
            "is_validation": True,
            "predicted_y": mu,
            "predicted_sd": sd,
        }
        return x

    def _best_estimate(self, model) -> list[float]:
        t = self._acquisition_time()
        k = _z_score(1.0 - self.config.validation_confidence)

        if self.config.validation_visited_only:
            X, _ = self._train_data()
            probe = X.clone()
            probe[:, -1] = t
            mu, sd = posterior_mean_std(model, probe)
            best = int(torch.argmin(mu + k * sd))
            return X[best, : self.dim].tolist()

        # Search the continuous domain on a multi-start grid, then polish.
        # The polish runs in unit-cube coordinates so a single learning rate
        # behaves sensibly however different the per-dimension scales are.
        lo, hi = self.bounds[0], self.bounds[1]
        width = hi - lo
        n_raw = max(self.config.raw_samples, 256)
        cand = lo + width * torch.rand(n_raw, self.dim, **self._tkwargs)

        X, _ = self._train_data()
        cand = torch.cat([cand, X[:, : self.dim]], dim=0)

        probe = torch.cat(
            [cand, torch.full((cand.size(0), 1), t, **self._tkwargs)], dim=-1
        )
        mu, sd = posterior_mean_std(model, probe)
        ucb = mu + k * sd

        starts = cand[torch.argsort(ucb)[: max(1, self.config.num_restarts // 2)]]
        best_x, best_v = starts[0], float(ucb.min())

        def bound_at(z: Tensor) -> Tensor:
            point = torch.cat(
                [lo + z * width, torch.tensor([t], **self._tkwargs)]
            ).unsqueeze(0)
            post = model.posterior(point)
            return (post.mean + k * post.variance.clamp_min(1e-12).sqrt()).squeeze()

        for start in starts:
            z = ((start - lo) / width).detach().clone().requires_grad_(True)
            polish = torch.optim.Adam([z], lr=0.05)
            for _ in range(60):
                polish.zero_grad()
                bound_at(z).backward()
                polish.step()
                with torch.no_grad():
                    z.clamp_(0.0, 1.0)
            with torch.no_grad():
                v = float(bound_at(z))
            if v < best_v:
                best_v, best_x = v, (lo + z.detach() * width).clone()

        return best_x.tolist()

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
        """Record a measured cost for an input."""
        x = list(map(float, x))
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} input values, got {len(x)}: {x}")
        if not math.isfinite(y):
            raise ValueError(f"Cost must be finite, got {y}")

        # Pending state (the prediction, the validation flag) belongs to the
        # point that was suggested. If the caller evaluated something else,
        # attaching it would label the record with a prediction for a different
        # input. The tolerance absorbs a float32 round trip through Unity.
        pending = self._pending or {}
        pending_x = pending.get("x")
        matches = (
            pending_x is not None
            and len(pending_x) == len(x)
            and all(
                math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-8)
                for a, b in zip(pending_x, x, strict=True)
            )
        )
        if is_validation is None:
            is_validation = bool(pending.get("is_validation", False)) if matches else False

        iteration = self.next_iteration
        obs = Observation(
            iteration=iteration,
            x=x,
            y=float(y),
            time=float(iteration) if time is None else float(time),
            is_validation=is_validation,
            predicted_y=pending.get("predicted_y") if matches else None,
            predicted_sd=pending.get("predicted_sd") if matches else None,
        )
        self.observations.append(obs)
        self._pending = None
        self._model_stale = True
        return obs

    # -- closed loop ----------------------------------------------------

    def run(
        self,
        objective: Callable[[list[float]], float],
        num_iterations: int,
        callback: Callable[[Observation], None] | None = None,
    ) -> list[Observation]:
        """Run a full closed loop against a callable objective.

        Convenience for simulation and testing. In a real study the ask/tell
        methods are driven by the experiment instead.
        """
        for _ in range(num_iterations):
            x = self.suggest()
            obs = self.observe(x, objective(x))
            if callback is not None:
                callback(obs)
        return self.observations

    # -- reporting ------------------------------------------------------

    def best_observed(self) -> Observation | None:
        return min(self.observations, key=lambda o: o.y) if self.observations else None

    def history(self) -> list[dict]:
        return [o.as_dict() for o in self.observations]

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

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bounds": self.bounds.T.tolist(),
            "alpha": self.alpha,
            "config": {
                "exploration_ratio": self.config.exploration_ratio,
                "validation_every": self.config.validation_every,
                "validation_confidence": self.config.validation_confidence,
                "acquisition_time_offset": self.config.acquisition_time_offset,
                "stationary": self.config.model.stationary,
            },
            "observations": self.history(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

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


def _z_score(p: float) -> float:
    """Inverse standard normal CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    return float(math.sqrt(2.0) * _erfinv(2.0 * p - 1.0))


def _erfinv(x: float) -> float:
    return float(torch.erfinv(torch.tensor(x, dtype=torch.float64)))
