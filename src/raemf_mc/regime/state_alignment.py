"""State alignment placeholder utilities."""

from __future__ import annotations

import numpy as np


def sort_states_by_mean_return(state_means: np.ndarray) -> np.ndarray:
    """Return a stable state order from low to high mean return."""
    return np.argsort(state_means)
