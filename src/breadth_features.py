"""Optional market-breadth interface; no synthetic breadth is generated."""

from __future__ import annotations

import pandas as pd

BREADTH_COLUMNS = [
    "advance_decline_ratio", "percentage_above_ma20", "percentage_above_ma50",
    "percentage_above_ma200", "new_high_new_low", "cross_sectional_dispersion",
    "ceiling_floor_ratio", "sector_breadth",
]


def add_optional_breadth_features(frame: pd.DataFrame, breadth: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """As-of merge genuine breadth data when supplied; otherwise return unchanged."""
    if breadth is None:
        return frame.copy(), []
    available = [c for c in BREADTH_COLUMNS if c in breadth.columns]
    if not available:
        return frame.copy(), []
    left = frame.sort_values("date")
    right = breadth[["date", *available]].sort_values("date")
    return pd.merge_asof(left, right, on="date", direction="backward"), available

