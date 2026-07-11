"""Feature redundancy, drift and range diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_redundancy(frame: pd.DataFrame, columns: list[str], threshold: float = 0.995) -> pd.DataFrame:
    corr = frame[columns].corr().abs()
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if np.isfinite(value) and value >= threshold:
                rows.append({"feature_a": left, "feature_b": right, "abs_correlation": value})
    return pd.DataFrame(rows)


def feature_drift_report(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        a, b = train[column].dropna(), test[column].dropna()
        scale = max(float(a.std()), 1e-12)
        rows.append({"feature": column, "train_mean": a.mean(), "test_mean": b.mean(), "standardized_mean_shift": abs(b.mean() - a.mean()) / scale, "train_min": a.min(), "train_max": a.max(), "test_out_of_range_rate": ((b < a.min()) | (b > a.max())).mean()})
    return pd.DataFrame(rows)

