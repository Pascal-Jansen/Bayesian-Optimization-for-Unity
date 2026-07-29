"""Multi-objective Dynamic Bayesian Optimization driver.

Extends :class:`~dbo_torch.optimizer.DynamicBO` to problems with several
simultaneously drifting objectives. The model is one GP per objective, each
with its own :class:`~dbo_torch.kernels.TemporalDecayKernel`, so each
objective fits its own drift rate ``alpha_i``: objectives can drift at
different speeds, and one can be stationary while another drifts.

Candidates are chosen by noisy expected hypervolume improvement evaluated at
the current time, so the Pareto front being improved upon is the front the
model believes is attainable *now*, not the front of stale measurements. See
:meth:`DynamicMOBO._optimize_acquisition` for the reasoning.

Basic use::

    opt = DynamicMOBO(bounds=[(-5.0, 5.0)], ref_point=[40.0, 40.0])
    for _ in range(30):
        x = opt.suggest()
        opt.observe(x, measure_costs(x))  # one cost per objective
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor

from .model import DBOModelConfig, build_model, fit_model, get_alpha, posterior_mean_std

__all__ = ["MODBOConfig", "DynamicMOBO", "MOObservation", "as_stationary_mo"]


@dataclass
class MOObservation:
    """One completed iteration, with one measured cost per objective."""

    iteration: int
    x: list[float]
    y: list[float]
    time: float
    is_validation: bool = False
    #: Per-objective costs the model predicted before the point was evaluated,
    #: when known.
    predicted_y: list[float] | None = None
    predicted_sd: list[float] | None = None

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
class MODBOConfig:
    """Optimiser settings.

    Mirrors :class:`~dbo_torch.optimizer.DBOConfig` where the concepts carry
    over. The single-objective over-exploitation guard has no analogue here:
    hypervolume improvement already rewards spreading along the front, so the
    search does not collapse onto one incumbent the way EI can.
    """

    #: Model configuration, applied to every objective. Each objective still
    #: gets its own kernel instances and therefore its own fitted ``alpha``.
    model: DBOModelConfig = field(default_factory=DBOModelConfig)

    #: Fixed inputs applied at the first iterations, before any model is fitted.
    seed_points: list[list[float]] | None = None

    #: Number of random seed points, used only when ``seed_points`` is None.
    num_seed_points: int = 3

    #: Run a validation iteration every N iterations. ``None`` disables
    #: automatic scheduling; call :meth:`DynamicMOBO.suggest_validation`
    #: directly.
    validation_every: int | None = None

    #: Time value at which the acquisition function is evaluated, relative to
    #: the most recent observation. Same semantics as the single-objective
    #: optimiser: ``0`` scores candidates at the current time, ``1`` at the
    #: time they will actually be evaluated.
    acquisition_time_offset: float = 0.0

    #: Restarts and raw samples for acquisition optimisation.
    num_restarts: int = 20
    raw_samples: int = 512

    #: Monte Carlo samples for the hypervolume-improvement estimate.
    mc_samples: int = 128

    #: Refit hyperparameters every N iterations. 1 refits every iteration.
    refit_every: int = 1

    seed: int | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


class DynamicMOBO:
    """Dynamic multi-objective Bayesian optimiser for drifting cost functions.

    The optimiser MINIMISES every objective, consistent with
    :class:`~dbo_torch.optimizer.DynamicBO`; observations are negated
    internally for BoTorch's maximisation-frame hypervolume machinery. Time is
    measured in iterations, appended as the last input column, exactly as in
    the single-objective optimiser.

    ``ref_point`` is given in the user's own minimisation units: a vector of
    the WORST acceptable value per objective. Hypervolume is measured against
    it, so observations worse than the reference point in any objective
    contribute nothing.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        ref_point: Sequence[float],
        config: MODBOConfig | None = None,
    ) -> None:
        self.config = config or MODBOConfig()
        self._tkwargs = {"dtype": self.config.dtype, "device": self.config.device}

        bounds_t = torch.tensor(
            [[lo for lo, _ in bounds], [hi for _, hi in bounds]], **self._tkwargs
        )
        if not torch.all(bounds_t[1] > bounds_t[0]):
            raise ValueError(f"Each bound must have hi > lo; got {list(bounds)}")

        self.bounds = bounds_t
        self.dim = bounds_t.size(-1)

        ref_point = [float(r) for r in ref_point]
        if len(ref_point) < 2:
            raise ValueError(
                "Multi-objective optimisation needs at least two objectives; "
                f"got a ref_point of length {len(ref_point)}. For one objective "
                "use DynamicBO."
            )
        if not all(math.isfinite(r) for r in ref_point):
            raise ValueError(f"ref_point values must be finite, got {ref_point}")

        self.num_objectives = len(ref_point)
        #: Reference point in the user's minimisation units.
        self.ref_point = ref_point
        # Negated once here; every hypervolume computation happens in the
        # maximisation frame BoTorch expects.
        self._neg_ref = -torch.tensor(ref_point, **self._tkwargs)

        if self.config.seed_points is not None:
            for p in self.config.seed_points:
                if len(p) != self.dim:
                    raise ValueError(
                        f"Each seed point needs {self.dim} values, got {len(p)}: {p}"
                    )

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        self.observations: list[MOObservation] = []
        self._models: list | None = None
        self._model_list: ModelListGP | None = None
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
    def alphas(self) -> list[float] | None:
        """Fitted temporal decay rate per objective, or None before the first fit."""
        if self._models is None:
            return None
        return [get_alpha(m) for m in self._models]

    def is_validation_iteration(self, iteration: int | None = None) -> bool:
        every = self.config.validation_every
        if every is None:
            return False
        it = self.next_iteration if iteration is None else iteration
        return it % every == 0

    def _train_data(self) -> tuple[Tensor, Tensor]:
        X = torch.tensor(
            [obs.x + [obs.time] for obs in self.observations], **self._tkwargs
        )
        Y = torch.tensor([obs.y for obs in self.observations], **self._tkwargs)
        return X, Y

    def _current_time(self) -> float:
        return float(self.observations[-1].time) if self.observations else 0.0

    def _acquisition_time(self) -> float:
        return self._current_time() + self.config.acquisition_time_offset

    def _ensure_models(self) -> ModelListGP | None:
        """Fit or refresh the per-objective GPs. Returns None if data is insufficient.

        One independent GP per objective, each with its own temporal kernel,
        wrapped in a :class:`ModelListGP`. Independence is what lets each
        objective fit its own drift rate.
        """
        if self.num_observations < 2:
            return None

        refit_every = max(1, self.config.refit_every)
        needs_refit = (
            self._model_list is None
            or self._model_stale
            and (self.num_observations % refit_every == 0 or self._model_list is None)
        )

        if self._model_list is None or self._model_stale:
            X, Y = self._train_data()
            models = []
            for i in range(self.num_objectives):
                model = build_model(X, Y[:, i : i + 1], self.config.model)
                if needs_refit:
                    model = fit_model(model)
                models.append(model)
            self._models = models
            self._model_list = ModelListGP(*models)
            self._model_stale = False

        return self._model_list

    # -- ask ------------------------------------------------------------

    def suggest(self) -> list[float]:
        """Return the next input to evaluate.

        Produces a seed point while seeding, a best-estimate input on a
        scheduled validation iteration, and a hypervolume-improvement
        candidate otherwise.
        """
        iteration = self.next_iteration

        if self.is_validation_iteration(iteration):
            x = self.suggest_validation()
            pending = {"is_validation": True}

            # Capture what the model expects to measure here, per objective,
            # before it is measured — same rationale as the single-objective
            # optimiser: the prediction/measurement gap isolates model quality
            # from exploration luck.
            model = self._ensure_models()
            if model is not None:
                mu, sd = self._predict_at(model, x)
                pending["predicted_y"] = mu
                pending["predicted_sd"] = sd

            self._pending = pending
            return x

        seeds = self.config.seed_points
        n_seed = len(seeds) if seeds is not None else self.config.num_seed_points

        if self.num_observations < n_seed:
            if seeds is not None:
                x = list(seeds[self.num_observations])
            else:
                lo, hi = self.bounds[0], self.bounds[1]
                x = (lo + (hi - lo) * torch.rand(self.dim, **self._tkwargs)).tolist()
            self._pending = {"is_validation": False}
            return x

        model = self._ensure_models()
        if model is None:
            lo, hi = self.bounds[0], self.bounds[1]
            x = (lo + (hi - lo) * torch.rand(self.dim, **self._tkwargs)).tolist()
            self._pending = {"is_validation": False}
            return x

        x = self._optimize_acquisition(model)
        mu, sd = self._predict_at(model, x)
        self._pending = {"is_validation": False, "predicted_y": mu, "predicted_sd": sd}
        return x

    def _optimize_acquisition(self, model: ModelListGP) -> list[float]:
        """Maximise noisy expected hypervolume improvement at a fixed point in time.

        The candidate space is the control parameters only; the time
        coordinate is pinned, since we are choosing what to try *now*, not
        when to try it.

        This is the scientific heart of the dynamic multi-objective step.
        qLogNEHVI improves on the hypervolume of the model posterior at
        ``X_baseline``, so the baseline is the set of observed inputs with
        their time column OVERWRITTEN TO THE CURRENT TIME. Under the temporal
        kernel that posterior is each past design's *currently predicted*
        objective vector: an observation taken long ago is discounted by
        ``alpha_i ** lag`` per objective and its prediction reverts toward
        the prior, exactly as much as the fitted drift rates say it should.
        The Pareto front the acquisition improves upon is therefore
        drift-adjusted automatically — the exact multi-objective analogue of
        the single-objective optimiser computing its incumbent from the
        posterior at t = now instead of from stale measured values, and it
        falls out of the baseline choice with no extra machinery.
        """
        t_acq = self._acquisition_time()
        X, _ = self._train_data()

        baseline = X.clone()
        baseline[:, -1] = t_acq

        # BoTorch maximises hypervolume; the objective negates every outcome
        # to map our minimisation problem into that frame, and the reference
        # point is negated to match.
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self._neg_ref.tolist(),
            X_baseline=baseline,
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size([self.config.mc_samples]),
                seed=self.config.seed,
            ),
            objective=WeightedMCMultiOutputObjective(
                weights=-torch.ones(self.num_objectives, **self._tkwargs)
            ),
            prune_baseline=True,
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

    def suggest_validation(self) -> list[float]:
        """Return the optimiser's current best single estimate on the front.

        Restricted to already-evaluated inputs. Every visited input is
        re-scored by the posterior mean of each objective at the current time,
        the non-dominated subset of those predictions is taken, and the point
        whose removal would cost the most hypervolume is returned. That is the
        design the model currently considers most indispensable to its
        drift-adjusted Pareto front — the natural multi-objective reading of
        "best current estimate".
        """
        model = self._ensure_models()
        if model is None:
            return self._fallback_point()

        X, _ = self._train_data()
        probe = X.clone()
        probe[:, -1] = self._acquisition_time()
        means = self._posterior_means(model, probe)

        best = self._best_contributor(-means)
        return X[best, : self.dim].tolist()

    def _best_contributor(self, neg_scores: Tensor) -> int:
        """Row index with the largest drop-one hypervolume contribution.

        ``neg_scores`` is ``(n, m)`` in the negated (maximisation) frame.
        Dominated rows are excluded first; among the rest, each row is scored
        by how much the front's hypervolume shrinks without it. If every row
        is beyond the reference point all contributions are zero and the
        first non-dominated row is returned.
        """
        mask = is_non_dominated(neg_scores)
        idx = torch.nonzero(mask).reshape(-1)
        if idx.numel() == 1:
            return int(idx[0])

        front = neg_scores[idx]
        total = self._hypervolume(front)
        contributions = [
            total - self._hypervolume(torch.cat([front[:j], front[j + 1 :]], dim=0))
            for j in range(front.size(0))
        ]
        best = max(range(len(contributions)), key=contributions.__getitem__)
        return int(idx[best])

    def _fallback_point(self) -> list[float]:
        if self.observations:
            _, Y = self._train_data()
            return list(self.observations[self._best_contributor(-Y)].x)
        return ((self.bounds[0] + self.bounds[1]) / 2).tolist()

    # -- posterior helpers ----------------------------------------------

    def _posterior_means(self, model: ModelListGP, X: Tensor) -> Tensor:
        """Per-objective posterior means at ``X``, shape ``(n, m)``, minimisation units."""
        cols = [posterior_mean_std(sub, X)[0].reshape(-1) for sub in model.models]
        return torch.stack(cols, dim=-1)

    def _predict_at(
        self, model: ModelListGP, x: Sequence[float], t: float | None = None
    ) -> tuple[list[float], list[float]]:
        t = self._acquisition_time() if t is None else t
        point = torch.tensor([list(x) + [t]], **self._tkwargs)
        mus, sds = [], []
        for sub in model.models:
            mu, sd = posterior_mean_std(sub, point)
            mus.append(float(mu.reshape(-1)[0]))
            sds.append(float(sd.reshape(-1)[0]))
        return mus, sds

    def _hypervolume(self, neg_Y: Tensor) -> float:
        """Hypervolume of ``neg_Y`` (maximisation frame) over the negated ref point."""
        if neg_Y.numel() == 0:
            return 0.0
        partitioning = DominatedPartitioning(ref_point=self._neg_ref, Y=neg_Y)
        return float(partitioning.compute_hypervolume())

    # -- tell -----------------------------------------------------------

    def observe(
        self,
        x: Sequence[float],
        y: Sequence[float],
        is_validation: bool | None = None,
        time: float | None = None,
    ) -> MOObservation:
        """Record one measured cost per objective for an input."""
        x = list(map(float, x))
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} input values, got {len(x)}: {x}")

        y = [float(v) for v in y]
        if len(y) != self.num_objectives:
            raise ValueError(
                f"Expected {self.num_objectives} objective values, got {len(y)}: {y}"
            )
        if not all(math.isfinite(v) for v in y):
            raise ValueError(f"Costs must be finite, got {y}")

        pending = self._pending or {}
        if is_validation is None:
            is_validation = bool(pending.get("is_validation", False))

        iteration = self.next_iteration
        obs = MOObservation(
            iteration=iteration,
            x=x,
            y=y,
            time=float(iteration) if time is None else float(time),
            is_validation=is_validation,
            predicted_y=pending.get("predicted_y"),
            predicted_sd=pending.get("predicted_sd"),
        )
        self.observations.append(obs)
        self._pending = None
        self._model_stale = True
        return obs

    # -- closed loop ----------------------------------------------------

    def run(
        self,
        objective: Callable[[list[float]], Sequence[float]],
        num_iterations: int,
        callback: Callable[[MOObservation], None] | None = None,
    ) -> list[MOObservation]:
        """Run a full closed loop against a callable vector objective.

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

    def pareto_front(self, at_current_time: bool = False) -> list[dict]:
        """Non-dominated visited designs, judged by raw or current-time values.

        With ``at_current_time=False`` the front is computed from the measured
        objective vectors as recorded. With ``True`` every visited input is
        re-scored by the posterior mean of each objective at the current time
        and the front is computed from those predictions instead.

        Under drift the two differ, and the second is the one that matters: a
        raw front is anchored by measurements taken when the system was in
        states it no longer occupies, so it can keep designs that are no
        longer good and miss ones that have become good. The current-time
        front is the set of designs the model believes are Pareto-optimal
        *now*.

        Returns dicts with keys ``iteration``, ``x`` and ``y``, where ``y``
        holds measured values or posterior means respectively. Falls back to
        the raw front when no model has been fitted yet.
        """
        if not self.observations:
            return []

        X, Y = self._train_data()
        scores = Y
        if at_current_time:
            model = self._ensure_models()
            if model is not None:
                probe = X.clone()
                probe[:, -1] = self._acquisition_time()
                scores = self._posterior_means(model, probe)

        mask = is_non_dominated(-scores)
        return [
            {
                "iteration": self.observations[i].iteration,
                "x": self.observations[i].x,
                "y": scores[i].tolist(),
            }
            for i in torch.nonzero(mask).reshape(-1).tolist()
        ]

    def hypervolume_trace(self) -> list[float]:
        """Hypervolume of the OBSERVED front after each iteration.

        Measured in the user's minimisation frame against the configured
        reference point. Because observations only accumulate — dominated
        points never leave the observed set — the trace is non-decreasing by
        construction. It says how much of objective space the run has covered,
        not what is attainable now; under drift the attainable front is the
        current-time front from :meth:`pareto_front`.
        """
        if not self.observations:
            return []
        _, Y = self._train_data()
        neg = -Y
        return [self._hypervolume(neg[: i + 1]) for i in range(neg.size(0))]

    def history(self) -> list[dict]:
        return [o.as_dict() for o in self.observations]

    def prediction_error(self) -> list[dict]:
        """Per-objective gap between predicted and measured cost, per validation step."""
        return [
            {
                "iteration": o.iteration,
                "predicted": o.predicted_y,
                "measured": o.y,
                "error": [abs(p - m) for p, m in zip(o.predicted_y, o.y, strict=True)],
            }
            for o in self.observations
            if o.is_validation and o.predicted_y is not None
        ]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bounds": self.bounds.T.tolist(),
            "ref_point": self.ref_point,
            "alphas": self.alphas,
            "config": {
                "validation_every": self.config.validation_every,
                "acquisition_time_offset": self.config.acquisition_time_offset,
                "stationary": self.config.model.stationary,
            },
            "hypervolume_trace": self.hypervolume_trace(),
            "observations": self.history(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def __repr__(self) -> str:
        alphas = self.alphas
        shown = (
            "unfitted"
            if alphas is None
            else "[" + ", ".join(f"{a:.4f}" for a in alphas) + "]"
        )
        return (
            f"DynamicMOBO(dim={self.dim}, m={self.num_objectives}, "
            f"n={self.num_observations}, alphas={shown})"
        )


def as_stationary_mo(config: MODBOConfig) -> MODBOConfig:
    """Return a copy configured as plain stationary multi-objective BO.

    Pins ``alpha = 1`` for every objective, giving the qNEHVI baseline the
    dynamic optimiser is compared against.
    """
    return replace(config, model=replace(config.model, stationary=True))
