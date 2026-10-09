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

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor

from ._base import DynamicOptimizerBase, _z_score
from .model import DBOModelConfig, get_alpha, posterior_mean_std

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
    #: when known, at the time the measurement was taken.
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

    #: Tail probability for the per-objective upper bound used to score
    #: validation candidates, as in the single-objective optimiser: 0.01
    #: gives a multiplier of ~2.33 on each posterior standard deviation, and
    #: 0.5 scores by posterior mean alone.
    validation_confidence: float = 0.01

    #: Time value at which the acquisition function is evaluated, relative to
    #: the most recent observation. Same semantics as the single-objective
    #: optimiser: ``0`` scores candidates at the current time, ``1`` at the
    #: time they will actually be evaluated. Recorded predictions are always
    #: made at the time of evaluation.
    acquisition_time_offset: float = 0.0

    #: Restarts and raw samples for acquisition optimisation.
    num_restarts: int = 20
    raw_samples: int = 512

    #: Monte Carlo samples for the hypervolume-improvement estimate.
    mc_samples: int = 128

    #: Refit hyperparameters every N iterations. 1 refits every iteration.
    refit_every: int = 1

    #: Start each refit from the previous fit's hyperparameters. See
    #: :attr:`DBOConfig.warm_start <dbo_torch.optimizer.DBOConfig.warm_start>`.
    warm_start: bool = False

    seed: int | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


class DynamicMOBO(DynamicOptimizerBase):
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

    A run can be saved with :meth:`save` and resumed with :meth:`load`.
    """

    _observation_type = MOObservation
    _config_type = MODBOConfig

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        ref_point: Sequence[float],
        config: MODBOConfig | None = None,
    ) -> None:
        super().__init__(bounds, config or MODBOConfig())

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

        self._models: list | None = None
        self._model_list: ModelListGP | None = None

    # -- state ----------------------------------------------------------

    @property
    def alphas(self) -> list[float] | None:
        """Fitted temporal decay rate per objective, or None before the first fit."""
        if self._models is None:
            return None
        return [get_alpha(m) for m in self._models]

    def _train_data(self) -> tuple[Tensor, Tensor]:
        Y = torch.tensor([obs.y for obs in self.observations], **self._tkwargs)
        return self._train_X(), Y

    def _ensure_models(self) -> ModelListGP | None:
        """Fit or refresh the per-objective GPs. Returns None if data is insufficient.

        One independent GP per objective, each with its own temporal kernel,
        wrapped in a :class:`ModelListGP`. Independence is what lets each
        objective fit its own drift rate.
        """
        if self.num_observations < 2:
            return None

        if self._model_list is None or self._model_stale:
            X, Y = self._train_data()
            previous = self._models or [None] * self.num_objectives
            self._models = [
                self._refresh_model(X, Y[:, i : i + 1], prev)
                for i, prev in enumerate(previous)
            ]
            self._model_list = ModelListGP(*self._models)
            self._model_stale = False

        return self._model_list

    # -- ask ------------------------------------------------------------

    def suggest(self) -> list[float]:
        """Return the next input to evaluate.

        Produces a seed point while seeding, a best-estimate input on a
        scheduled validation iteration, and a hypervolume-improvement
        candidate otherwise.
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

        model = self._ensure_models()
        if model is None:
            x = self._random_point()
            self._set_pending(x, is_validation=False)
            return x

        x = self._optimize_acquisition(model)
        self._set_pending(x, is_validation=False, prediction=self._predict_at(model, x))
        return x

    def _optimize_acquisition(self, model: ModelListGP) -> list[float]:
        """Maximise noisy expected hypervolume improvement at a fixed point in time.

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

        Once their times are overwritten, re-tested inputs (every validation
        iteration re-tests one) are exact duplicates. They are removed: they
        add nothing to the front and make the joint baseline covariance
        singular.
        """
        t_acq = self._acquisition_time()
        baseline = self._visited_at(t_acq, unique=True)

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
        x, _ = self._maximize(acqf, t_acq)
        return x

    def suggest_validation(self) -> list[float]:
        """Return the optimiser's current best single estimate on the front.

        Restricted to already-evaluated inputs. Every visited input is scored
        at the acquisition time by a per-objective upper confidence bound
        ``mu_i + k * sd_i`` — risk-averse, as in the single-objective
        optimiser; see ``validation_confidence`` — the non-dominated subset of
        those scores is taken, and the point whose removal would cost the most
        hypervolume is returned. That is the design the model is most
        confident is indispensable to its drift-adjusted Pareto front.

        Records the per-objective prediction at the chosen point, at the time
        it will be measured, so the validation flag and prediction reach the
        observation whether this is called directly or by :meth:`suggest`.
        """
        with self._isolated_rng():
            return self._suggest_validation()

    def _suggest_validation(self) -> list[float]:
        model = self._ensure_models()
        if model is None:
            x = self._fallback_point()
            self._set_pending(x, is_validation=True)
            return x

        probe = self._visited_at(self._acquisition_time())
        k = _z_score(1.0 - self.config.validation_confidence)
        means, sds = self._posterior_moments(model, probe)

        best = self._best_contributor(-(means + k * sds))
        x = probe[best, : self.dim].tolist()
        self._set_pending(x, is_validation=True, prediction=self._predict_at(model, x))
        return x

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

    def _posterior_moments(self, model: ModelListGP, X: Tensor) -> tuple[Tensor, Tensor]:
        """Per-objective posterior means and sds at ``X``, each ``(n, m)``."""
        moments = [posterior_mean_std(sub, X) for sub in model.models]
        means = torch.stack([mu.reshape(-1) for mu, _ in moments], dim=-1)
        sds = torch.stack([sd.reshape(-1) for _, sd in moments], dim=-1)
        return means, sds

    def _posterior_means(self, model: ModelListGP, X: Tensor) -> Tensor:
        """Per-objective posterior means at ``X``, shape ``(n, m)``, minimisation units."""
        return self._posterior_moments(model, X)[0]

    def _predict_at(
        self, model: ModelListGP, x: Sequence[float], t: float | None = None
    ) -> tuple[list[float], list[float]]:
        """Per-objective mean and sd at ``x``, by default when it will be measured."""
        t = self._evaluation_time() if t is None else t
        point = torch.tensor([list(x) + [t]], **self._tkwargs)
        means, sds = self._posterior_moments(model, point)
        return means[0].tolist(), sds[0].tolist()

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
        """Record one measured cost per objective for an input.

        The validation flag and prediction recorded for the last suggestion
        are attached only if ``x`` is that suggestion (to within a float32
        round trip).
        """
        x = self._check_x(x)

        y = [float(v) for v in y]
        if len(y) != self.num_objectives:
            raise ValueError(
                f"Expected {self.num_objectives} objective values, got {len(y)}: {y}"
            )
        if not all(math.isfinite(v) for v in y):
            raise ValueError(f"Costs must be finite, got {y}")

        return self._record(x, y, is_validation, time)

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

        _, Y = self._train_data()
        scores = Y
        if at_current_time:
            with self._isolated_rng():
                model = self._ensure_models()
            if model is not None:
                scores = self._posterior_means(model, self._visited_at(self._acquisition_time()))

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

    def _summary(self) -> dict:
        return {
            "ref_point": self.ref_point,
            "alphas": self.alphas,
            "hypervolume_trace": self.hypervolume_trace(),
        }

    @classmethod
    def _from_payload(cls, payload: dict, config):
        return cls(bounds=payload["bounds"], ref_point=payload["ref_point"], config=config)

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
    seeds = None if config.seed_points is None else [list(p) for p in config.seed_points]
    return replace(config, model=replace(config.model, stationary=True), seed_points=seeds)
