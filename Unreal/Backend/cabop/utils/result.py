"""Result dataclass for Bayesian optimization iterations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class ProposeLocationResult:
    """
    Result of a single Bayesian optimization iteration.

    Contains the proposed point, acquisition function values, and metadata
    about the optimization state at the time of proposal.

    Attributes:
        proposed_x: The proposed design point as a nested dict
        acquistion_min: Minimum acquisition function value found
        acquisition: The acquisition function used
        X_sample: Observed sample locations (in unit space)
        Y_sample: Observed objective values
        gp: The fitted Gaussian Process model
        last_state: Previous optimization state (if any)
        sample_map: Mapping of samples to their outcomes
        expected_cost: Expected fabrication cost for this proposal
        expected_ei: Expected improvement value
        delta_info: Information gain metric
        current_best_val: Current best observed value
        current_best_x: Design point with best observed value
        realized_x: The actually realized design (after constraints)
        realized_cost: Actual cost incurred
        regret: Distance from known optimum (for benchmarking)
        actual_y: Noise-free objective value (for benchmarking)
        expected_y: Observed (possibly noisy) objective value
        iter_count: Iteration number
        random_phase: Whether this was during random initialization
        running_cost: Cumulative cost so far
    """

    # Required fields
    proposed_x: dict
    acquistion_min: float
    acquisition: Callable
    X_sample: Optional[np.ndarray]
    Y_sample: Optional[np.ndarray]
    gp: Any
    last_state: Optional[dict]
    sample_map: dict
    expected_cost: float
    expected_ei: float
    delta_info: float

    # Optional fields with defaults
    current_best_val: Optional[float] = None
    current_best_x: Optional[dict] = None
    realized_x: Optional[dict] = None
    realized_cost: float = 0.0
    regret: float = 0.0
    actual_y: float = 0.0
    expected_y: float = 0.0
    iter_count: int = 0
    random_phase: bool = False
    running_cost: float = 0.0
