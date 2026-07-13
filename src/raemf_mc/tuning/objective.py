"""Resource-aware multiclass tuning objective."""

from __future__ import annotations

import numpy as np


def composite_loss(metrics: dict[str, float]) -> float:
    """Lower-is-better normalized loss emphasizing probability quality and tails."""
    log_scale = float(np.log(4.0))
    return float(
        0.25 * np.clip(metrics["brier"] / 2.0, 0.0, 2.0)
        + 0.20 * np.clip(metrics["log_loss"] / log_scale, 0.0, 3.0)
        + 0.20 * (1.0 - metrics["macro_f1"])
        + 0.10 * (1.0 - metrics["balanced_accuracy"])
        + 0.075 * (1.0 - metrics["recall_bear"])
        + 0.075 * (1.0 - metrics["recall_stress"])
        + 0.10 * np.clip(metrics["ece"], 0.0, 1.0)
    )


def composite_score(macro_f1: float, balanced_accuracy: float, brier: float) -> float:
    """Backward-compatible higher-is-better score."""
    return 0.45 * macro_f1 + 0.25 * balanced_accuracy + 0.30 * (1.0 - brier / 2.0)
