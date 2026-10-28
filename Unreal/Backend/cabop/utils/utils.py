"""Utility functions for CABOP."""

import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return super().default(obj)


def rbf(d: np.ndarray, sigma: float) -> np.ndarray:
    """
    Radial basis function (Gaussian) kernel.

    Args:
        d: Array of distances
        sigma: Kernel bandwidth

    Returns:
        RBF kernel values: exp(-d^2 / (2 * sigma^2))
    """
    return np.exp(-(d**2) / (2.0 * sigma**2))
