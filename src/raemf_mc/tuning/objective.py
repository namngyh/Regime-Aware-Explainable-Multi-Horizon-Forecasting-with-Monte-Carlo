"""Tuning objective."""

from __future__ import annotations


def composite_score(macro_f1: float, balanced_accuracy: float, brier: float) -> float:
    """Laptop-safe objective from the specification."""
    return 0.45 * macro_f1 + 0.25 * balanced_accuracy + 0.30 * (1.0 - brier / 2.0)
