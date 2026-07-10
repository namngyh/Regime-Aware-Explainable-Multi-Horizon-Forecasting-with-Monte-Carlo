"""Leakage-aware tuning for the two retained 20-session models."""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.metrics import classification_metrics, regression_metrics
from src.models import (
    fit_predict_hmm,
    fit_predict_macd,
    macd_bullish_signal,
)


SEARCH_SPACE = {
    "MACD": [
        {"fast": 12, "slow": 26, "signal": 9},
        {"fast": 5, "slow": 20, "signal": 5},
        {"fast": 8, "slow": 21, "signal": 5},
        {"fast": 8, "slow": 24, "signal": 9},
        {"fast": 10, "slow": 30, "signal": 9},
        {"fast": 12, "slow": 30, "signal": 9},
        {"fast": 15, "slow": 35, "signal": 9},
        {"fast": 20, "slow": 50, "signal": 9},
    ],
    "HMM Regime": [
        {"n_components": 4, "covariance_type": "diag", "n_iter": 400},
        {"n_components": 3, "covariance_type": "diag", "n_iter": 350},
        {"n_components": 5, "covariance_type": "diag", "n_iter": 450},
        {"n_components": 4, "covariance_type": "tied", "n_iter": 400},
        {"n_components": 2, "covariance_type": "diag", "n_iter": 350},
        {"n_components": 6, "covariance_type": "diag", "n_iter": 500},
    ],
}


def forecast_score(classification, regression):
    """65% return, 25% price and 10% direction quality."""

    def safe(key):
        value = regression.get(key, np.nan)
        return value if np.isfinite(value) else -1.0

    return (
        0.35 * safe("mae_skill_vs_zero")
        + 0.30 * safe("rmse_skill_vs_zero")
        + 0.15 * safe("price_mae_skill_vs_no_change")
        + 0.10 * safe("price_rmse_skill_vs_no_change")
        + 0.10 * classification["balanced_accuracy"]
    )


def _evaluate_candidate(name, params, frame, feature_cols, horizon, splits):
    target_return = f"future_return_{horizon}d"
    target_up = f"future_up_{horizon}d"
    working = frame.copy()
    if name == "MACD":
        working["macd_selected"] = macd_bullish_signal(
            working["close"], params["fast"], params["slow"], params["signal"]
        ).to_numpy()

    rows = []
    for fold, (train_index, valid_index) in enumerate(splits, start=1):
        train = working.iloc[train_index]
        valid = working.iloc[valid_index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "MACD":
                bundle = fit_predict_macd(train, valid, horizon)
            else:
                bundle = fit_predict_hmm(
                    train,
                    valid,
                    feature_cols,
                    horizon,
                    hmm_params=params,
                )
        classification = classification_metrics(
            valid[target_up].astype(int), bundle.pred_direction, bundle.score_up
        )
        regression = regression_metrics(
            valid[target_return], bundle.pred_return, valid["close"]
        )
        rows.append(
            {
                "fold": fold,
                "cv_score": forecast_score(classification, regression),
                "mae": regression["mae"],
                "rmse": regression["rmse"],
                "price_mae": regression["price_mae"],
                "price_rmse": regression["price_rmse"],
                "balanced_accuracy": classification["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def tune_horizon(frame, feature_cols, horizon=20, n_splits=3):
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    splits = list(splitter.split(frame))
    trial_rows = []
    best_rows = []
    selected = {}

    for name, candidates in SEARCH_SPACE.items():
        model_trials = []
        for candidate_id, params in enumerate(candidates):
            started = time.perf_counter()
            error = ""
            try:
                folds = _evaluate_candidate(
                    name, params, frame, feature_cols, horizon, splits
                )
                aggregate = folds.mean(numeric_only=True).to_dict()
                score_std = folds["cv_score"].std(ddof=0)
            except Exception as exc:
                aggregate = {
                    key: np.nan
                    for key in [
                        "mae",
                        "rmse",
                        "price_mae",
                        "price_rmse",
                        "balanced_accuracy",
                    ]
                }
                aggregate["cv_score"] = -np.inf
                score_std = np.nan
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "horizon": horizon,
                "model": name,
                "candidate_id": candidate_id,
                "params_json": json.dumps(params, sort_keys=True),
                "cv_score": aggregate["cv_score"],
                "cv_score_std": score_std,
                "cv_mae": aggregate["mae"],
                "cv_rmse": aggregate["rmse"],
                "cv_price_mae": aggregate["price_mae"],
                "cv_price_rmse": aggregate["price_rmse"],
                "cv_balanced_accuracy": aggregate["balanced_accuracy"],
                "fit_seconds": time.perf_counter() - started,
                "error": error,
            }
            trial_rows.append(row)
            model_trials.append((row, params))

        best_trial, best_params = max(
            model_trials, key=lambda item: item[0]["cv_score"]
        )
        selected[name] = best_params
        best_rows.append(
            {**best_trial, "selected_params": best_trial["params_json"]}
        )

    selected_keys = {
        (row["model"], row["candidate_id"]) for row in best_rows
    }
    for row in trial_rows:
        row["selected"] = (row["model"], row["candidate_id"]) in selected_keys

    return (
        selected["MACD"],
        selected["HMM Regime"],
        pd.DataFrame(trial_rows),
        pd.DataFrame(best_rows),
    )
