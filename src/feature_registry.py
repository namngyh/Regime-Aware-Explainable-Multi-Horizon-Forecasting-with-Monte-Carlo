"""Feature metadata used in reports and leakage checks."""

from __future__ import annotations

import pandas as pd


def build_feature_registry(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in feature_columns:
        group = "volume" if "volume" in column else "volatility" if any(x in column for x in ("vol_", "variance", "drawdown", "atr", "parkinson")) else "momentum_trend"
        rows.append({"feature": column, "group": group, "available": column in frame, "missing_rate": float(frame[column].isna().mean()), "definition": column.replace("_", " ")})
    return pd.DataFrame(rows)

