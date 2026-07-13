"""Statistical comparison placeholders."""

from __future__ import annotations

import numpy as np


def loss_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return paired loss difference a - b."""
    return np.asarray(a) - np.asarray(b)
