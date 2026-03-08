"""
Cost-Aware Bayesian Optimization (CABOP)

This module implements Bayesian optimization with cost-awareness for experimental
design scenarios where different parameter configurations have different
fabrication or acquisition costs.

The key components are:
- BOSpace: Defines the parameter space with bounds, groups, and cost structures
- CostModel: Tracks history and computes expected costs based on past samples
- BayesOpt: The main optimizer using Gaussian Process regression and
  cost-weighted Expected Improvement acquisition
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

from loguru import logger
from .utils.result import ProposeLocationResult
from .utils.utils import rbf

# Type aliases for clarity
DesignDict = dict[str, dict[str, float]]


# -----------------------------------------------------------------------------
# Parameter Space Definition
# -----------------------------------------------------------------------------

@dataclass
class BOSpace:
    """
    Defines the parameter space for Bayesian optimization.

    The parameters dict should contain:
        - "groups": List of group names (e.g., ["hardware", "software"])
        - "parameters": Dict mapping parameter names to their config:
            - "bound": np.ndarray of [lower, upper] bounds
            - "tolerance": float tolerance for matching previous samples
            - "group": str group name this parameter belongs to
        - "cost": Dict mapping groups to cost values for unchanged/swapped/acquired
        - "actual_cost": Dict mapping groups to actual realized costs

    Example:
        parameters = {
            "groups": ["hardware", "software"],
            "parameters": {
                "x1": {"bound": np.array([0, 1]), "tolerance": 0.05, "group": "hardware"},
                "x2": {"bound": np.array([0, 1]), "tolerance": 0.05, "group": "software"},
            },
            "cost": {"hardware": {"unchanged": 1, "swapped": 10, "acquired": 100}, ...},
            "actual_cost": {...}
        }
    """

    parameters: dict

    @property
    def n_parameters(self) -> int:
        """Return the total number of parameters across all groups."""
        return len(self.parameters["parameters"])

    @property
    def bounds(self) -> np.ndarray:
        """
        Return parameter bounds as a (n_parameters, 2) array.

        Parameters are ordered by group, then by parameter name within each group.
        """
        bounds = np.zeros((self.n_parameters, 2))
        idx = 0
        for group in self.parameters["groups"]:
            for param_name, param_config in self.parameters["parameters"].items():
                if param_config["group"] == group:
                    bounds[idx, :] = param_config["bound"]
                    idx += 1
        return bounds


# -----------------------------------------------------------------------------
# Cost Model
# -----------------------------------------------------------------------------

class CostModel:
    """
    Tracks experimental history and computes expected fabrication costs.

    The cost model considers three scenarios for each parameter group:
    - unchanged: Same as the most recent sample (cheapest)
    - swapped: Matches a previous sample in history (medium cost)
    - acquired: New configuration requiring fabrication (most expensive)
    """

    def __init__(self, space: BOSpace):
        self.space = space
        self.history: list[dict] = []

    def update(self, x_dict: dict) -> None:
        """
        Add a realized design to the cost history.

        Args:
            x_dict: Design as nested dict {group: {param: value, ...}, ...}
        """
        self.history.append(x_dict)

    def select_sample(
        self,
        sampled_x: np.ndarray,
        numpy_to_design: Callable,
        design_to_numpy: Callable,
        prefab: Optional[dict] = None,
    ) -> Tuple[Tuple[float, ...], np.ndarray]:
        """
        Resolve realized sample and costs based on fabrication constraints.

        This method checks if the proposed sample can reuse previous configurations
        (unchanged or swapped) or requires new fabrication (acquired).

        Args:
            sampled_x: Proposed sample as numpy array
            numpy_to_design: Function to convert numpy array to design dict
            design_to_numpy: Function to convert design dict to numpy array
            prefab: Optional dict of prefabricated values to snap to

        Returns:
            Tuple of (costs per group, realized sample as numpy array)
        """
        costs = []
        intended_x = numpy_to_design(sampled_x)
        realized_x = numpy_to_design(sampled_x)

        # Snap to prefabricated values if provided
        if prefab is not None:
            intended_x = self._match_prefabricated(intended_x, prefab)
            realized_x = self._match_prefabricated(realized_x, prefab)

        # Determine cost for each parameter group
        for group in self.space.parameters["groups"]:
            cost = self.space.parameters["actual_cost"][group]["acquired"]

            if len(self.history) == 0:
                # No history - must acquire new
                pass
            elif self._within_tolerance(self.history[-1][group], intended_x):
                # Matches most recent sample - unchanged
                cost = self.space.parameters["actual_cost"][group]["unchanged"]
                realized_x[group] = self.history[-1][group]
            else:
                # Check if we can swap from any previous sample
                for prev in self.history:
                    if self._within_tolerance(prev[group], intended_x):
                        cost = self.space.parameters["actual_cost"][group]["swapped"]
                        realized_x[group] = prev[group]
                        break

            costs.append(cost)

        return tuple(costs), design_to_numpy(realized_x)

    def smooth_cost(
        self,
        x: np.ndarray,
        numpy_to_design: Callable,
        design_to_numpy: Callable,
    ) -> Tuple[float, Tuple[float, ...]]:
        """
        Compute expected fabrication cost using soft matching.

        Uses RBF kernel weighting to compute a smooth, differentiable cost
        function based on distance to previous samples.

        Args:
            x: Candidate point as numpy array
            numpy_to_design: Function to convert numpy array to design dict
            design_to_numpy: Function to convert design dict to numpy array

        Returns:
            Tuple of (expected total cost, matched design as tuple)
        """
        groups = list(self.space.parameters["groups"])
        params_dict = self.space.parameters["parameters"]

        # Map groups to their parameter names
        group_params = {
            g: [name for name, cfg in params_dict.items() if cfg["group"] == g]
            for g in groups
        }

        x_dict = numpy_to_design(x)
        total_cost = 0.0
        matched_dict = {g: {} for g in groups}

        for group in groups:
            names = group_params[group]
            xg = np.array([x_dict[group][n] for n in names], dtype=float)
            tg = np.array([params_dict[n]["tolerance"] for n in names], dtype=float)

            last_g, samples_g = self._get_group_history(group, names)

            cost_dict = self.space.parameters["cost"][group]
            c_unchanged = float(cost_dict["unchanged"])
            c_swapped = float(cost_dict["swapped"])
            c_new = float(cost_dict["acquired"])

            c_exp, x_exp = self._compute_soft_cost(
                xg, tg, last_g, samples_g, c_unchanged, c_swapped, c_new
            )
            total_cost += c_exp

            for name, val in zip(names, x_exp):
                matched_dict[group][name] = float(val)

        matched_vec = design_to_numpy(matched_dict)
        return float(total_cost), tuple(float(v) for v in matched_vec)

    def _match_prefabricated(self, x_dict: dict, prefab: dict) -> dict:
        """Snap parameter values to nearest prefabricated option."""
        for param, prefab_values in prefab.items():
            matching_keys = [k for k, v in x_dict.items() if param in v]
            if matching_keys:
                key = matching_keys[0]
                diffs = np.abs(np.asarray(prefab_values) - x_dict[key][param])
                min_idx = np.argmin(diffs)
                x_dict[key][param] = prefab_values[min_idx]
        return x_dict

    def _get_group_history(
        self, group: str, names: list[str]
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Extract history for a specific parameter group.

        Returns:
            Tuple of (most recent sample for group or None, array of all samples)
        """
        last_g = None
        samples_list = []

        # Find most recent sample for this group
        for entry in reversed(self.history):
            if group in entry and all(n in entry[group] for n in names):
                last_g = np.array([entry[group][n] for n in names], dtype=float)
                break

        # Collect all samples for this group
        for entry in self.history:
            if group in entry and all(n in entry[group] for n in names):
                samples_list.append(
                    np.array([entry[group][n] for n in names], dtype=float)
                )

        samples_g = (
            np.vstack(samples_list)
            if samples_list
            else np.empty((0, len(names)), float)
        )
        return last_g, samples_g

    def _compute_soft_cost(
        self,
        xg: np.ndarray,
        tg: np.ndarray,
        last_g: Optional[np.ndarray],
        samples_g: np.ndarray,
        c_unchanged: float,
        c_swapped: float,
        c_new: float,
        sigma: float = 0.05,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute soft expected cost for a parameter group using RBF weighting.

        Args:
            xg: Candidate values for this group
            tg: Tolerance values for this group (unused, kept for future)
            last_g: Most recent sample for this group (or None)
            samples_g: Array of all previous samples for this group
            c_unchanged: Cost if unchanged from last sample
            c_swapped: Cost if swapped from history
            c_new: Cost if newly acquired
            sigma: RBF kernel bandwidth

        Returns:
            Tuple of (expected cost, expected matched point)
        """
        # Weight for unchanged (most recent sample)
        if last_g is None:
            w_last = 0.0
        else:
            d_last = float(np.linalg.norm(xg - last_g))
            w_last = float(rbf(np.array([d_last]), sigma).item())

        # Weight for swapped (any historical sample)
        if samples_g.size > 0:
            dists = np.linalg.norm(samples_g - xg[None, :], axis=1)
            w = rbf(dists, sigma)
            w_samp = float(np.sum(w))
            xm = (
                (w[:, None] * samples_g).sum(axis=0) / w_samp
                if w_samp > 0.0
                else xg
            )
        else:
            w_samp = 0.0
            xm = xg

        # Small weight for new acquisition to avoid division issues
        w_new = 1e-1
        Z = w_last + w_samp + w_new

        # Compute expected cost as weighted average
        expected_cost = (
            (w_last / Z) * c_unchanged
            + (w_samp / Z) * c_swapped
            + (w_new / Z) * c_new
        )

        # Compute expected matched point
        denom = w_last + w_samp
        if denom > 0.0 and last_g is not None:
            x_exp = (w_last * last_g + w_samp * xm) / denom
        else:
            x_exp = xm

        return float(expected_cost), x_exp.astype(float, copy=False)

    def _within_tolerance(self, a: dict, all_params: dict) -> bool:
        """Check if parameter values are within tolerance of each other."""
        flattened = {k: all_params[g][k] for g in all_params for k in all_params[g]}
        for key in a.keys():
            tolerance = self.space.parameters["parameters"][key]["tolerance"]
            if abs(a[key] - flattened[key]) > tolerance:
                return False
        return True


# -----------------------------------------------------------------------------
# Bayesian Optimizer
# -----------------------------------------------------------------------------

class BayesOpt:
    """
    Cost-aware Bayesian optimizer using Gaussian Process regression.

    This optimizer extends standard Bayesian optimization by incorporating
    fabrication costs into the acquisition function, making it suitable for
    experimental design scenarios where changing parameters has associated costs.

    Args:
        space: BOSpace defining the parameter space and cost structure
        ifCost: If True, use cost-aware EI; if False, use standard EI
        random_state: Optional random seed for reproducibility
    """

    def __init__(
        self,
        space: BOSpace,
        ifCost: bool = True,
        random_state: Optional[int] = None,
    ):
        self.space = space
        self.ifCost = ifCost
        self.rng = np.random.default_rng(random_state)

        logger.info(f"[BayesOpt] Cost-aware mode = {self.ifCost}")

        # Initialize GP with Matern kernel (good for optimization)
        kernel = C(1.0) * Matern(length_scale=0.15, nu=1.5) + WhiteKernel(noise_level=1e-1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=15
        )

        self.cost_model = CostModel(space)
        self.sample_map: dict = {}

        # Sample storage
        self.X_sample: Optional[np.ndarray] = None
        self.Y_sample: Optional[np.ndarray] = None
        self.X_sample_intended: Optional[np.ndarray] = None
        self.Y_sample_intended: Optional[np.ndarray] = None

        # Fitted data (may differ from samples based on update_rule)
        self.X_fit: Optional[np.ndarray] = None
        self.Y_fit: Optional[np.ndarray] = None

        self.current_best = {"x": None, "y": float("inf")}

    # -------------------------------------------------------------------------
    # Coordinate Conversions
    # -------------------------------------------------------------------------

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        """Convert from original space to [0, 1] unit hypercube."""
        lo, hi = self.space.bounds[:, 0], self.space.bounds[:, 1]
        return (x - lo) / (hi - lo)

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        """Convert from [0, 1] unit hypercube to original space."""
        lo, hi = self.space.bounds[:, 0], self.space.bounds[:, 1]
        return lo + u * (hi - lo)

    def _numpy_to_design(self, x: np.ndarray) -> dict[str, dict[str, float]]:
        """Convert numpy array to nested design dict {group: {param: value}}."""
        d: dict[str, dict[str, float]] = {
            group: {} for group in self.space.parameters["groups"]
        }
        i = 0
        for group in self.space.parameters["groups"]:
            for name, param in self.space.parameters["parameters"].items():
                if param["group"] == group:
                    d[group][name] = float(x[i])
                    i += 1
        return d

    def _design_to_numpy(self, d: dict[str, dict[str, float]]) -> np.ndarray:
        """Convert nested design dict to numpy array."""
        x = np.empty(self.space.n_parameters, dtype=float)
        i = 0
        for group in self.space.parameters["groups"]:
            for name, param in self.space.parameters["parameters"].items():
                if param["group"] == group:
                    x[i] = d[group][name]
                    i += 1
        return x

    # -------------------------------------------------------------------------
    # Acquisition Functions
    # -------------------------------------------------------------------------

    def _expected_improvement(
        self,
        X: np.ndarray,
        X_sample: np.ndarray,
        Y_sample: np.ndarray,
        gp: GaussianProcessRegressor,
        xi: float = 0.01,
    ) -> np.ndarray:
        """
        Compute Expected Improvement acquisition function.

        Args:
            X: Points to evaluate, shape (n_points, n_dims) in unit space
            X_sample: Observed samples in unit space
            Y_sample: Observed values
            gp: Fitted Gaussian Process
            xi: Exploration-exploitation trade-off parameter

        Returns:
            EI values for each point in X
        """
        if X_sample is None:
            return np.zeros(X.shape[0])

        mu, sigma = gp.predict(X, return_std=True)
        mu_opt = np.min(Y_sample)

        imp = mu_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        return ei

    def _expected_improvement_per_cost(
        self,
        X: np.ndarray,
        X_sample: np.ndarray,
        Y_sample: np.ndarray,
        gp: GaussianProcessRegressor,
        xi: float = 0.01,
    ) -> np.ndarray:
        """
        Compute cost-weighted Expected Improvement.

        Returns EI divided by expected cost, favoring points that are both
        promising and cheap to evaluate.
        """
        if X_sample is None:
            return np.zeros(X.shape[0])

        ei = self._expected_improvement(X, X_sample, Y_sample, gp, xi)
        costs = self._compute_costs(X)

        return ei / costs

    def _compute_costs(self, X: np.ndarray) -> np.ndarray:
        """Compute expected costs for candidate points in unit space."""
        costs = np.array([
            self.cost_model.smooth_cost(
                self._from_unit(x), self._numpy_to_design, self._design_to_numpy
            )[0]
            for x in X
        ])
        return costs

    def _ei_at_point(
        self,
        x: np.ndarray,
        X_sample: np.ndarray,
        Y_sample: np.ndarray,
        gp: GaussianProcessRegressor,
        xi: float = 0.01,
    ) -> float:
        """Compute EI at a single point (for reporting)."""
        if X_sample is None:
            return 0.0

        X = np.atleast_2d(x)
        mu, sigma = gp.predict(X, return_std=True)
        mu_opt = np.min(Y_sample)

        imp = mu_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        return float(ei[0])

    # -------------------------------------------------------------------------
    # Sampling and Optimization
    # -------------------------------------------------------------------------

    def _sobol_sample(self, n: int) -> np.ndarray:
        """Generate n quasi-random samples in unit hypercube using Sobol sequence."""
        sampler = Sobol(
            d=self.space.bounds.shape[0],
            scramble=True,
            seed=int(self.rng.integers(1_000_000_000)),
        )
        return sampler.random(n)

    def _optimize_acquisition(
        self,
        acquisition: Callable,
        X_sample: Optional[np.ndarray],
        Y_sample: Optional[np.ndarray],
        gp: GaussianProcessRegressor,
        dim: int,
        random_sample: bool = False,
        n_restarts: int = 10,
    ) -> Tuple[np.ndarray, ProposeLocationResult]:
        """
        Find the point that maximizes the acquisition function.

        Args:
            acquisition: Acquisition function to maximize
            X_sample: Observed samples (or None if no observations)
            Y_sample: Observed values (or None if no observations)
            gp: Gaussian Process model
            dim: Dimensionality of the search space
            random_sample: If True, return a random sample (for initialization)
            n_restarts: Number of random restarts for optimization

        Returns:
            Tuple of (best point in original space, ProposeLocationResult)
        """
        min_val = np.inf
        min_x = None
        unit_bounds = [(0.0, 1.0)] * dim

        def neg_acquisition(u):
            """Negative acquisition for minimization."""
            return -acquisition(u.reshape(1, dim), X_sample, Y_sample, gp).item()

        if random_sample:
            logger.info("[BayesOpt] Using random sample (initialization phase)")
            u_best = self._sobol_sample(1)[0]
            min_val = -acquisition(u_best.reshape(1, dim), X_sample, Y_sample, gp)
            min_x = self._from_unit(u_best)
        else:
            # Multi-start L-BFGS-B optimization
            u_best = None
            for _ in range(n_restarts):
                u0 = self.rng.uniform(0.0, 1.0, size=dim)
                result = minimize(
                    neg_acquisition, x0=u0, bounds=unit_bounds, method="L-BFGS-B"
                )
                if result.fun < min_val:
                    min_val = result.fun
                    min_x = self._from_unit(result.x)
                    u_best = result.x

        # Validate bounds
        assert np.all(min_x >= self.space.bounds[:, 0]), "Solution below lower bounds"
        assert np.all(min_x <= self.space.bounds[:, 1]), "Solution above upper bounds"

        # Compute metrics for result
        predicted_cost = float(
            self.cost_model.smooth_cost(
                u_best if u_best is not None else self._to_unit(min_x),
                self._numpy_to_design,
                self._design_to_numpy,
            )[0]
        )
        predicted_ei = (
            self._ei_at_point(
                u_best if u_best is not None else self._to_unit(min_x),
                X_sample,
                Y_sample,
                gp,
            )
            if X_sample is not None
            else 0.0
        )

        result = ProposeLocationResult(
            proposed_x=self._numpy_to_design(min_x),
            acquistion_min=-min_val,
            acquisition=acquisition,
            X_sample=X_sample,
            Y_sample=Y_sample,
            gp=gp,
            last_state=None,
            expected_cost=predicted_cost,
            expected_ei=predicted_ei,
            delta_info=0.0,
            sample_map=self.sample_map,
        )

        return min_x, result

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def ask(self, n_init: int = 5) -> Tuple[np.ndarray, ProposeLocationResult]:
        """
        Propose the next point to evaluate.

        During the initialization phase (fewer than n_init samples), returns
        quasi-random samples. After initialization, optimizes the acquisition
        function to find the most promising point.

        Args:
            n_init: Number of initial random samples before optimization

        Returns:
            Tuple of (proposed point as numpy array, ProposeLocationResult)
        """
        acq = (
            self._expected_improvement_per_cost
            if self.ifCost
            else self._expected_improvement
        )
        dim = self.space.bounds.shape[0]

        # Fit GP if we have enough samples
        if self.X_sample is not None and len(self.X_sample) >= n_init:
            self.gp.fit(self.X_fit, self.Y_fit)

        x_proposed, result = self._optimize_acquisition(
            acquisition=acq,
            X_sample=self.X_sample,
            Y_sample=self.Y_sample,
            gp=self.gp,
            dim=dim,
            random_sample=(self.X_sample is None or len(self.X_sample) < n_init),
        )

        return x_proposed, result

    def tell(
        self,
        x_realized: np.ndarray | DesignDict,
        y: float,
        x_intended: Optional[np.ndarray | DesignDict] = None,
        update_rule: str = "actual",
    ) -> None:
        """
        Report an observation to the optimizer.

        Args:
            x_realized: The actual point that was evaluated (may differ from
                       proposed due to fabrication constraints). Can be numpy
                       array or design dict.
            y: The observed objective value
            x_intended: The originally proposed point (if different from realized)
            update_rule: How to update the GP model:
                - "actual": Fit on realized points only
                - "intended": Fit on intended points only
                - "both": Fit on both realized and intended points
        """
        if x_realized is None or y is None:
            logger.warning("[BayesOpt] tell() called with None values, skipping")
            return

        # Convert to design dict if needed
        x_realized_dict: DesignDict
        if isinstance(x_realized, np.ndarray):
            x_realized_dict = self._numpy_to_design(x_realized)
        else:
            x_realized_dict = x_realized

        x_intended_dict: Optional[DesignDict] = None
        if x_intended is not None:
            if isinstance(x_intended, np.ndarray):
                x_intended_dict = self._numpy_to_design(x_intended)
            else:
                x_intended_dict = x_intended

        # Store in sample map
        self.sample_map[tuple(self._design_to_numpy(x_realized_dict))] = {"y": y}

        # Convert to unit space
        x_realized_norm = self._to_unit(self._design_to_numpy(x_realized_dict))
        x_intended_norm: Optional[np.ndarray] = None
        if x_intended_dict is not None:
            x_intended_norm = self._to_unit(self._design_to_numpy(x_intended_dict))

        # Update cost model history
        self.cost_model.update(x_realized_dict)

        # Update sample arrays
        if self.X_sample is None:
            self.X_sample = np.array([x_realized_norm])
            self.Y_sample = np.array([y])
            self.Y_sample_intended = np.array([y])
        else:
            self.X_sample = np.vstack([self.X_sample, x_realized_norm])
            self.Y_sample = np.append(self.Y_sample, y)
            self.Y_sample_intended = np.append(self.Y_sample_intended, y)

        if x_intended_norm is not None:
            if self.X_sample_intended is None:
                self.X_sample_intended = np.array([x_intended_norm])
            else:
                self.X_sample_intended = np.vstack(
                    [self.X_sample_intended, x_intended_norm]
                )

        # Set fitting data based on update rule
        if update_rule == "intended":
            if self.X_sample_intended is None:
                raise ValueError("Cannot use 'intended' rule without intended samples")
            self.X_fit = self.X_sample_intended
            self.Y_fit = self.Y_sample_intended
        elif update_rule == "actual":
            self.X_fit = self.X_sample
            self.Y_fit = self.Y_sample
        elif update_rule == "both":
            if self.X_sample_intended is None:
                raise ValueError("Cannot use 'both' rule without intended samples")
            self.X_fit = np.vstack([self.X_sample_intended, self.X_sample])
            self.Y_fit = np.append(self.Y_sample_intended, self.Y_sample)
        else:
            raise ValueError(f"Invalid update rule: {update_rule}")

        # Update best observed
        if y < self.current_best.get("y", float("inf")):
            self.current_best.update({"y": y, "x": x_realized_dict})

    def select_sample(
        self, sampled_x: np.ndarray, prefab: Optional[dict] = None
    ) -> Tuple[Tuple[float, ...], np.ndarray]:
        """
        Apply fabrication constraints to a proposed sample.

        Args:
            sampled_x: Proposed sample as numpy array
            prefab: Optional dict of prefabricated values to snap to

        Returns:
            Tuple of (costs per group, realized sample as numpy array)
        """
        return self.cost_model.select_sample(
            sampled_x, self._numpy_to_design, self._design_to_numpy, prefab
        )
