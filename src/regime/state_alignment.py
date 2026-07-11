"""HMM state label alignment using Hungarian profile matching."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def align_state_profiles(reference: np.ndarray, candidate: np.ndarray) -> dict[int, int]:
    """Map candidate state indexes to reference indexes by standardized distance."""
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    if reference.shape != candidate.shape:
        raise ValueError("state profile arrays must have identical shape")
    scale = np.nanstd(np.vstack([reference, candidate]), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-10)] = 1.0
    cost = np.linalg.norm((candidate[:, None, :] - reference[None, :, :]) / scale, axis=2)
    candidate_idx, reference_idx = linear_sum_assignment(cost)
    return {int(c): int(r) for c, r in zip(candidate_idx, reference_idx)}


def align_probabilities(probabilities: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Reorder posterior columns after label matching."""
    source = np.asarray(probabilities, float)
    out = np.zeros_like(source)
    for candidate, reference in mapping.items():
        out[:, reference] = source[:, candidate]
    return out

