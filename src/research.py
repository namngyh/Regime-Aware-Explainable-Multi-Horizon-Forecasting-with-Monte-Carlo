"""Medium-depth robustness checks for overlapping 20-session forecasts."""

from itertools import combinations
from math import erfc, sqrt

import numpy as np
import pandas as pd

from src.metrics import classification_metrics, regression_metrics
from src.tuning import forecast_score


def _scored_metrics(frame):
    classification = classification_metrics(
        frame["actual_direction"], frame["pred_direction"], frame["score_up"]
    )
    regression = regression_metrics(
        frame["actual_return"], frame["pred_return"], frame["current_close"]
    )
    return {
        "forecast_score": forecast_score(classification, regression),
        "mae": regression["mae"],
        "rmse": regression["rmse"],
        "price_mae": regression["price_mae"],
        "price_rmse": regression["price_rmse"],
        "balanced_accuracy": classification["balanced_accuracy"],
    }


def block_bootstrap_intervals(
    predictions,
    n_bootstrap=1000,
    block_size=20,
    random_state=42,
):
    """Moving-block bootstrap CIs that preserve short-range dependence."""
    rng = np.random.default_rng(random_state)
    rows = []
    for model, group in predictions.groupby("model", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        blocks_needed = int(np.ceil(n / block_size))
        max_start = max(n - block_size + 1, 1)
        samples = []
        for _ in range(n_bootstrap):
            starts = rng.integers(0, max_start, size=blocks_needed)
            indices = np.concatenate(
                [np.arange(start, min(start + block_size, n)) for start in starts]
            )[:n]
            samples.append(_scored_metrics(group.iloc[indices]))
        sample_frame = pd.DataFrame(samples)
        estimates = _scored_metrics(group)
        for metric, estimate in estimates.items():
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower_95": sample_frame[metric].quantile(0.025),
                    "ci_upper_95": sample_frame[metric].quantile(0.975),
                    "n_bootstrap": n_bootstrap,
                    "block_size": block_size,
                }
            )
    return pd.DataFrame(rows)


def _newey_west_long_run_variance(values, max_lag):
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    n = len(centered)
    variance = np.dot(centered, centered) / n
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1 - lag / (max_lag + 1)
        covariance = np.dot(centered[lag:], centered[:-lag]) / n
        variance += 2 * weight * covariance
    return max(variance, 0.0)


def pairwise_dm_tests(predictions, horizon=20):
    """Diebold-Mariano-style tests with HAC lag horizon-1 and Holm correction."""
    frame = predictions.sort_values("date")
    actual = frame.groupby("date")["actual_return"].first()
    forecast = frame.pivot(index="date", columns="model", values="pred_return")
    rows = []
    for model_a, model_b in combinations(forecast.columns, 2):
        joined = pd.concat(
            [actual, forecast[model_a], forecast[model_b]], axis=1, join="inner"
        ).dropna()
        joined.columns = ["actual", "forecast_a", "forecast_b"]
        loss_a = (joined["actual"] - joined["forecast_a"]) ** 2
        loss_b = (joined["actual"] - joined["forecast_b"]) ** 2
        loss_diff = (loss_a - loss_b).to_numpy()
        long_run_variance = _newey_west_long_run_variance(
            loss_diff, max_lag=horizon - 1
        )
        standard_error = sqrt(long_run_variance / len(loss_diff))
        dm_stat = loss_diff.mean() / standard_error if standard_error > 0 else np.nan
        p_value = erfc(abs(dm_stat) / sqrt(2)) if np.isfinite(dm_stat) else np.nan
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "observations": len(loss_diff),
                "mean_squared_loss_difference_a_minus_b": loss_diff.mean(),
                "dm_stat": dm_stat,
                "p_value": p_value,
                "lower_loss_model": model_a if loss_diff.mean() < 0 else model_b,
                "hac_max_lag": horizon - 1,
            }
        )
    result = pd.DataFrame(rows)
    valid = result["p_value"].notna()
    ordered = result.loc[valid].sort_values("p_value")
    adjusted = {}
    running_max = 0.0
    total = len(ordered)
    for rank, (index, row) in enumerate(ordered.iterrows(), start=1):
        candidate = min(1.0, (total - rank + 1) * row["p_value"])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    result["holm_p_value"] = pd.Series(adjusted)
    result["significant_5pct"] = result["holm_p_value"] < 0.05
    return result
