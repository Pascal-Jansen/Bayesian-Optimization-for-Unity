"""Machinery shared by the single- and multi-objective optimisers.

Everything here is independent of how many objectives there are: the domain,
the seeding schedule, validation scheduling, the record of what was suggested
(so a prediction is only ever attached to the input it was made for),
random-number isolation, the refit/carry-over model lifecycle, and saving and
resuming a run. It lives in one place deliberately: fixes made to one
optimiser previously failed to reach the other.
"""

from __future__ import annotations

import base64
import json
import math
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from importlib import metadata
from pathlib import Path

import torch
from botorch.optim import optimize_acqf
from torch import Tensor

from .model import DBOModelConfig, build_model, fit_model

# Tolerance for matching an observed input to the one that was suggested.
# Absorbs a float32 round trip through Unity.
_MATCH_RTOL = 1e-5
_MATCH_ATOL = 1e-8

_SAVE_FORMAT = 2


class DynamicOptimizerBase:
    """Ask/tell machinery shared by :class:`~dbo_torch.DynamicBO` and
    :class:`~dbo_torch.DynamicMOBO`. Not meant to be used directly."""

    #: Observation record type and config type, set by each subclass.
    _observation_type: type
    _config_type: type

    def __init__(self, bounds: Sequence[tuple[float, float]], config) -> None:
        self.config = config
        self._tkwargs = {"dtype": config.dtype, "device": config.device}

        bounds_t = torch.tensor(
            [[lo for lo, _ in bounds], [hi for _, hi in bounds]], **self._tkwargs
        )
        if not torch.all(bounds_t[1] > bounds_t[0]):
            raise ValueError(f"Each bound must have hi > lo; got {list(bounds)}")

        self.bounds = bounds_t
        self.dim = bounds_t.size(-1)

        if config.seed_points is not None:
            for p in config.seed_points:
                if len(p) != self.dim:
                    raise ValueError(
                        f"Each seed point needs {self.dim} values, got {len(p)}: {p}"
                    )

        # A private generator: the optimiser never reseeds or consumes torch's
        # global one. See _isolated_rng.
        self._generator = torch.Generator()
        if config.seed is not None:
            self._generator.manual_seed(config.seed)
        else:
            self._generator.seed()

        self.observations: list = []
        self._pending: dict | None = None
        self._model_stale = True

    # -- state ----------------------------------------------------------

    @property
    def num_observations(self) -> int:
        return len(self.observations)

    @property
    def next_iteration(self) -> int:
        """1-based index of the iteration that :meth:`suggest` will produce."""
        return self.num_observations + 1

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

    def _current_time(self) -> float:
        return float(self.observations[-1].time) if self.observations else 0.0

    def _acquisition_time(self) -> float:
        """Time at which candidates are scored; see ``acquisition_time_offset``."""
        return self._current_time() + self.config.acquisition_time_offset

    def _evaluation_time(self) -> float:
        """Time at which the next suggestion will actually be measured.

        Predictions recorded for a suggestion are made here, whatever
        ``acquisition_time_offset`` says about where candidates are scored:
        a prediction is only comparable with the measurement if both refer to
        the same moment. On the default iteration clock this is
        ``next_iteration``.
        """
        return self._current_time() + 1.0

    def _train_X(self) -> Tensor:
        """Observed inputs with their time column, ``(n, d + 1)``."""
        return torch.tensor([o.x + [o.time] for o in self.observations], **self._tkwargs)

    def _visited_at(self, t: float, unique: bool = False) -> Tensor:
        """Observed inputs with the time column overwritten to ``t``."""
        X = self._train_X()[:, : self.dim]
        if unique:
            X = torch.unique(X, dim=0)
        return torch.cat([X, torch.full((X.size(0), 1), t, **self._tkwargs)], dim=-1)

    def _random_point(self) -> list[float]:
        lo, hi = self.bounds[0], self.bounds[1]
        return (lo + (hi - lo) * torch.rand(self.dim, **self._tkwargs)).tolist()

    def _next_seed_point(self) -> list[float] | None:
        """The next seed point, or None once the seed budget is spent."""
        n_used = self._seeds_used()
        if n_used >= self._seed_budget:
            return None
        seeds = self.config.seed_points
        return list(seeds[n_used]) if seeds is not None else self._random_point()

    # -- randomness -----------------------------------------------------

    @contextmanager
    def _isolated_rng(self) -> Iterator[None]:
        """Run a block on torch's global RNG, seeded from this optimiser alone.

        BoTorch draws raw samples and restart points from the global generator.
        Forking it here makes every suggestion a function of this optimiser's
        seed and history only, and leaves the global state exactly as it was —
        so two optimisers in one process (a DBO arm and a BO arm, say) cannot
        perturb each other, whatever order they run in.
        """
        seed = int(torch.randint(0, 2**62, (1,), generator=self._generator))
        device = torch.device(self.config.device)
        cuda_index = None
        if device.type == "cuda":
            cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.random.fork_rng(devices=[] if cuda_index is None else [cuda_index]):
            torch.default_generator.manual_seed(seed)
            if cuda_index is not None:
                torch.cuda.default_generators[cuda_index].manual_seed(seed)
            yield

    # -- suggestions and their predictions ------------------------------

    def _set_pending(
        self,
        x: Sequence[float],
        is_validation: bool,
        prediction: tuple | None = None,
    ) -> None:
        """Remember what was suggested, and what the model expects to measure."""
        pending = {"x": list(x), "is_validation": is_validation}
        if prediction is not None:
            pending["predicted_y"], pending["predicted_sd"] = prediction
        self._pending = pending

    def _claim_pending(self, x: Sequence[float]) -> dict:
        """Pending state if ``x`` is the input last suggested, else ``{}``.

        Pending state (the prediction, the validation flag) belongs to the
        point that was suggested. If the caller evaluated something else,
        attaching it would label the record with a prediction for a different
        input. Claiming clears it either way.
        """
        pending, self._pending = self._pending or {}, None
        pending_x = pending.get("x")
        if pending_x is None or len(pending_x) != len(x):
            return {}
        if all(
            math.isclose(a, b, rel_tol=_MATCH_RTOL, abs_tol=_MATCH_ATOL)
            for a, b in zip(pending_x, x, strict=True)
        ):
            return pending
        return {}

    def _check_x(self, x: Sequence[float]) -> list[float]:
        x = list(map(float, x))
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} input values, got {len(x)}: {x}")
        return x

    def _record(self, x: list[float], y, is_validation: bool | None, time: float | None):
        pending = self._claim_pending(x)
        if is_validation is None:
            is_validation = bool(pending.get("is_validation", False))

        iteration = self.next_iteration
        obs = self._observation_type(
            iteration=iteration,
            x=x,
            y=y,
            time=float(iteration) if time is None else float(time),
            is_validation=is_validation,
            predicted_y=pending.get("predicted_y"),
            predicted_sd=pending.get("predicted_sd"),
        )
        self.observations.append(obs)
        self._model_stale = True
        return obs

    # -- model lifecycle ------------------------------------------------

    def _refresh_model(self, X: Tensor, Y: Tensor, previous):
        """A GP for ``(X, Y)``: refitted, or carrying ``previous``'s hyperparameters.

        Rebuilding for new data constructs kernels at their starting values.
        On iterations that skip the refit (``refit_every > 1``) the last fitted
        hyperparameters are carried over instead of silently discarded; with
        ``warm_start`` they are also the starting point of each refit.
        Normalisation uses the fixed domain, so carried lengthscales keep
        their meaning; Standardize statistics are recomputed for the new data,
        so carried signal and noise scales are approximate in the new units.
        """
        refit = previous is None or self.num_observations % max(1, self.config.refit_every) == 0
        model = build_model(X, Y, self.config.model, bounds=self.bounds)
        if previous is not None and (not refit or self.config.warm_start):
            model.load_state_dict(_hyperparameter_state(previous), strict=False)
        if refit:
            model = fit_model(model, lengthscale_starts=self.config.model.lengthscale_starts)
        return model

    # -- acquisition ----------------------------------------------------

    def _maximize(self, acq_function, t: float) -> tuple[list[float], float]:
        """Maximise an acquisition function over the domain at fixed time ``t``.

        The candidate space is the control parameters only; the time
        coordinate is pinned, since we are choosing what to try, not when.
        """
        full_bounds = torch.cat(
            [self.bounds, torch.tensor([[t], [t]], **self._tkwargs)], dim=-1
        )
        candidate, value = optimize_acqf(
            acq_function=acq_function,
            bounds=full_bounds,
            q=1,
            num_restarts=self.config.num_restarts,
            raw_samples=self.config.raw_samples,
            fixed_features={self.dim: t},
            # All restarts in one batch. For q = 1 and study-sized data this
            # is about twice as fast as smaller batches, with the same optimum.
            options={"batch_limit": self.config.num_restarts, "maxiter": 200},
        )
        return candidate.squeeze(0)[: self.dim].tolist(), float(value)

    # -- closed loop ----------------------------------------------------

    def run(self, objective: Callable, num_iterations: int, callback: Callable | None = None):
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

    # -- persistence ----------------------------------------------------

    def history(self) -> list[dict]:
        return [o.as_dict() for o in self.observations]

    def _summary(self) -> dict:
        """Subclass-specific fields written by :meth:`save`."""
        return {}

    def save(self, path: str | Path) -> Path:
        """Write the run to JSON, sufficient to resume it with :meth:`load`.

        Records the full configuration, the library versions, the pending
        suggestion and the random-number state alongside the observations.
        The file is written to a temporary name and then moved into place, so
        a crash mid-write never leaves a truncated record.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng_state = self._generator.get_state().numpy().tobytes()
        payload = {
            "format": _SAVE_FORMAT,
            "kind": type(self).__name__,
            "versions": _versions(),
            "bounds": self.bounds.T.tolist(),
            **self._summary(),
            "config": _config_to_dict(self.config),
            "observations": self.history(),
            "pending": self._pending,
            "rng_state": base64.b64encode(rng_state).decode("ascii"),
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def _from_payload(cls, payload: dict, config):
        return cls(bounds=payload["bounds"], config=config)

    @classmethod
    def load(cls, path: str | Path):
        """Resume a run written by :meth:`save`.

        The model is refitted on the next suggestion. With the default
        ``refit_every=1`` and ``warm_start=False`` a resumed run continues
        exactly as the uninterrupted one would have; otherwise the first
        suggestion after resuming refits from scratch where the uninterrupted
        run would have carried hyperparameters over.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = payload.get("kind")
        if kind is not None and kind != cls.__name__:
            raise ValueError(f"{path} holds a {kind} run, not a {cls.__name__} run")

        config = _config_from_dict(cls._config_type, payload.get("config", {}))
        opt = cls._from_payload(payload, config)
        opt.observations = [opt._observation_type(**o) for o in payload["observations"]]
        opt._pending = payload.get("pending")
        state = payload.get("rng_state")
        if state is not None:
            raw = bytearray(base64.b64decode(state))
            opt._generator.set_state(torch.frombuffer(raw, dtype=torch.uint8).clone())
        opt._model_stale = True
        return opt


def _hyperparameter_state(model) -> dict[str, Tensor]:
    """Fitted hyperparameters, without constraint bounds or transform statistics.

    Constraint bounds are excluded so a noise floor raised by a failed fit is
    not carried into later models, and so a data-dependent floor is set from
    the current data.
    """
    return {
        k: v
        for k, v in model.state_dict().items()
        if k.split(".")[0] in ("covar_module", "likelihood", "mean_module")
        and "_constraint." not in k
    }


def _versions() -> dict:
    from . import __version__

    out = {"dbo_torch": __version__}
    for package in ("torch", "botorch", "gpytorch"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def _config_to_dict(config) -> dict:
    out = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if is_dataclass(value):
            value = _config_to_dict(value)
        elif isinstance(value, torch.dtype):
            value = str(value).removeprefix("torch.")
        out[f.name] = value
    return out


def _config_from_dict(config_type: type, data: dict):
    known = {f.name: f for f in fields(config_type)}
    kwargs = {}
    for name, value in data.items():
        if name not in known:
            continue  # tolerate files written by other versions
        if name == "model" and isinstance(value, dict):
            value = _config_from_dict(DBOModelConfig, value)
        elif name == "dtype" and isinstance(value, str):
            value = getattr(torch, value)
        elif isinstance(known[name].default, tuple) and isinstance(value, list):
            value = tuple(value)  # JSON has no tuples
        kwargs[name] = value
    # Files written before format 2 kept only the stationary flag, at the top
    # level of the optimiser config.
    if "model" in known and "model" not in data and "stationary" in data:
        kwargs["model"] = DBOModelConfig(stationary=bool(data["stationary"]))
    return config_type(**kwargs)


def _z_score(p: float) -> float:
    """Inverse standard normal CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    return float(torch.special.ndtri(torch.tensor(p, dtype=torch.float64)))
