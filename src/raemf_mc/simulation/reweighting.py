"""EBM-reweighted Monte Carlo helpers."""

from __future__ import annotations

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    weights = weights / np.maximum(weights.sum(), 1e-12)
    return float(1.0 / np.sum(weights**2))
