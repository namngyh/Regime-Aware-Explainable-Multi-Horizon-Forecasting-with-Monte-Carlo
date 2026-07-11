"""Leakage-aware multi-horizon path targets for RAEMF-MC."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
CLASS_ORDER = ("Bull", "Sideway", "Bear", "Stress")


def create_forward_log_returns(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Add forward log return and its exact target end trading date."""
    out = df.copy()
    log_price = np.log(out["close"].astype(float))
    for horizon in horizons:
        out[f"forward_log_return_{horizon}"] = log_price.shift(-horizon) - log_price
        out[f"target_end_date_{horizon}"] = out["date"].shift(-horizon)
    return out


def create_future_path_metrics(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Create MAE/MFE from future prices; these columns are labels, never features."""
    out = df.copy()
    price = out["close"].to_numpy(dtype=float)
    n = len(out)
    for horizon in horizons:
        mae = np.full(n, np.nan)
        mfe = np.full(n, np.nan)
        for i in range(n - horizon):
            path_returns = price[i + 1 : i + horizon + 1] / price[i] - 1.0
            mae[i] = np.min(path_returns)
            mfe[i] = np.max(path_returns)
        out[f"mae_path_{horizon}"] = mae
        out[f"mfe_path_{horizon}"] = mfe
    return out


def create_volatility_scaled_labels(
    df: pd.DataFrame,
    horizons: Iterable[int],
    threshold: float = 0.5,
    volatility_window: int = 20,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Create direction labels using only ex-ante rolling volatility at t."""
    out = df.copy()
    if "log_ret_1" not in out:
        out["log_ret_1"] = np.log(out["close"]).diff()
    sigma = out["log_ret_1"].rolling(volatility_window).std()
    out["target_volatility_scale_daily"] = sigma
    for horizon in horizons:
        scale = sigma * np.sqrt(horizon)
        z = out[f"forward_log_return_{horizon}"] / (scale + epsilon)
        label = np.select([z > threshold, z < -threshold], ["Bull", "Bear"], default="Sideway")
        label = pd.Series(label, index=out.index, dtype="object").where(z.notna())
        out[f"target_scale_{horizon}"] = scale
        out[f"standardized_forward_return_{horizon}"] = z
        out[f"direction_label_{horizon}"] = label
    return out


def create_stress_labels(
    df: pd.DataFrame,
    horizons: Iterable[int],
    stress_lambda: float = 1.5,
) -> pd.DataFrame:
    """Override direction with Stress when the future path has extreme drawdown."""
    out = df.copy()
    for horizon in horizons:
        threshold = -stress_lambda * out[f"target_scale_{horizon}"]
        stress = out[f"mae_path_{horizon}"] < threshold
        label = out[f"direction_label_{horizon}"].copy()
        out[f"stress_flag_{horizon}"] = stress.where(label.notna())
        out[f"target_{horizon}"] = label.mask(stress, "Stress")
    return out


def create_multihorizon_targets(
    df: pd.DataFrame,
    horizons: Iterable[int] = (20, 40, 60),
    direction_threshold: float = 0.5,
    stress_lambda: float = 1.5,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Build all causal scales and future-dependent target columns."""
    horizons = tuple(horizons)
    out = create_forward_log_returns(df, horizons)
    out = create_future_path_metrics(out, horizons)
    out = create_volatility_scaled_labels(out, horizons, direction_threshold, volatility_window)
    return create_stress_labels(out, horizons, stress_lambda)


def validate_target_distribution(
    df: pd.DataFrame, horizons: Iterable[int], min_count: int = 20, max_share: float = 0.80
) -> list[str]:
    """Return warnings for sparse or severely imbalanced target classes."""
    warnings: list[str] = []
    for horizon in horizons:
        counts = df[f"target_{horizon}"].value_counts(dropna=True)
        if counts.empty:
            warnings.append(f"h={horizon}: no labeled observations")
            continue
        if counts.max() / counts.sum() > max_share:
            warnings.append(f"h={horizon}: dominant class share exceeds {max_share:.0%}")
        missing = [c for c in CLASS_ORDER if counts.get(c, 0) < min_count]
        if missing:
            warnings.append(f"h={horizon}: sparse classes {missing}")
    for warning in warnings:
        LOGGER.warning(warning)
    return warnings


def summarize_target_statistics(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Summarize target counts/shares by horizon and calendar year."""
    rows: list[dict[str, object]] = []
    years = pd.to_datetime(df["date"]).dt.year
    for horizon in horizons:
        frame = pd.DataFrame({"year": years, "target": df[f"target_{horizon}"]}).dropna()
        for (year, target), count in frame.groupby(["year", "target"]).size().items():
            total = int((frame["year"] == year).sum())
            rows.append({"horizon": horizon, "year": year, "target": target, "count": int(count), "share": count / total})
    return pd.DataFrame(rows)

