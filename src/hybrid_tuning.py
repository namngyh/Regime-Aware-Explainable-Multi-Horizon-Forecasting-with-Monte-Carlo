"""Two-stage tuning for EGARCH features and LightGBM median quantile."""

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.hybrid import fit_quantile_median, prepare_hybrid_features
from src.metrics import classification_metrics, regression_metrics
from src.tuning import forecast_score


EGARCH_SPACE = [
    {"p": 1, "o": 0, "q": 1, "dist": "normal"},
    {"p": 1, "o": 0, "q": 1, "dist": "t"},
    {"p": 1, "o": 1, "q": 1, "dist": "t"},
]

LIGHTGBM_QUANTILE_SPACE = [
    {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3, "min_child_samples": 50, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 4.0},
    {"n_estimators": 450, "learning_rate": 0.02, "num_leaves": 7, "max_depth": 3, "min_child_samples": 40, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 6.0},
    {"n_estimators": 600, "learning_rate": 0.015, "num_leaves": 7, "max_depth": 3, "min_child_samples": 60, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 8.0},
    {"n_estimators": 350, "learning_rate": 0.025, "num_leaves": 15, "max_depth": 4, "min_child_samples": 40, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5.0},
    {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 15, "max_depth": 4, "min_child_samples": 30, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 6.0},
    {"n_estimators": 700, "learning_rate": 0.012, "num_leaves": 15, "max_depth": 4, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0},
    {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 20, "max_depth": 5, "min_child_samples": 40, "subsample": 0.85, "colsample_bytree": 0.8, "reg_lambda": 8.0},
    {"n_estimators": 600, "learning_rate": 0.015, "num_leaves": 20, "max_depth": 5, "min_child_samples": 60, "subsample": 0.8, "colsample_bytree": 0.85, "reg_lambda": 12.0},
    {"n_estimators": 300, "learning_rate": 0.025, "num_leaves": 24, "max_depth": 6, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.75, "reg_lambda": 10.0},
    {"n_estimators": 500, "learning_rate": 0.015, "num_leaves": 24, "max_depth": 6, "min_child_samples": 70, "subsample": 0.75, "colsample_bytree": 0.8, "reg_lambda": 14.0},
    {"n_estimators": 800, "learning_rate": 0.01, "num_leaves": 12, "max_depth": 4, "min_child_samples": 80, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 16.0},
    {"n_estimators": 900, "learning_rate": 0.008, "num_leaves": 7, "max_depth": 3, "min_child_samples": 100, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 20.0},
]


def tune_hybrid_quantile(
    frame,
    feature_cols,
    horizon,
    hmm_params,
    n_splits=6,
):
    target_return = f"future_return_{horizon}d"
    target_up = f"future_up_{horizon}d"
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    split_indices = list(splitter.split(frame))
    trial_rows = []

    for egarch_id, egarch_params in enumerate(EGARCH_SPACE):
        prepared_folds = []
        preparation_started = time.perf_counter()
        for fold, (train_index, valid_index) in enumerate(split_indices, start=1):
            train = frame.iloc[train_index]
            valid = frame.iloc[valid_index]
            train_features, valid_features, _ = prepare_hybrid_features(
                train,
                valid,
                feature_cols,
                hmm_params,
                egarch_params,
            )
            prepared_folds.append(
                (fold, train, valid, train_features, valid_features)
            )
        preparation_seconds = time.perf_counter() - preparation_started

        for lightgbm_id, lightgbm_params in enumerate(LIGHTGBM_QUANTILE_SPACE):
            started = time.perf_counter()
            fold_scores = []
            fold_rows = []
            error = ""
            try:
                for fold, train, valid, train_features, valid_features in prepared_folds:
                    pred_return, _ = fit_quantile_median(
                        train_features,
                        train[target_return],
                        valid_features,
                        lightgbm_params,
                    )
                    pred_direction = (pred_return > 0).astype(int)
                    scale = max(float(train[target_return].std()), 1e-12)
                    score_up = 1 / (1 + np.exp(-pred_return / scale))
                    classification = classification_metrics(
                        valid[target_up].astype(int), pred_direction, score_up
                    )
                    regression = regression_metrics(
                        valid[target_return], pred_return, valid["close"]
                    )
                    score = forecast_score(classification, regression)
                    fold_scores.append(score)
                    fold_rows.append(
                        {
                            "mae": regression["mae"],
                            "rmse": regression["rmse"],
                            "price_mae": regression["price_mae"],
                            "balanced_accuracy": classification[
                                "balanced_accuracy"
                            ],
                        }
                    )
                aggregate = pd.DataFrame(fold_rows).mean().to_dict()
                score_mean = float(np.mean(fold_scores))
                score_std = float(np.std(fold_scores))
            except Exception as exc:
                aggregate = {
                    "mae": np.nan,
                    "rmse": np.nan,
                    "price_mae": np.nan,
                    "balanced_accuracy": np.nan,
                }
                score_mean = -np.inf
                score_std = np.nan
                error = f"{type(exc).__name__}: {exc}"
            trial_rows.append(
                {
                    "egarch_candidate_id": egarch_id,
                    "lightgbm_candidate_id": lightgbm_id,
                    "egarch_params": json.dumps(egarch_params, sort_keys=True),
                    "lightgbm_params": json.dumps(
                        lightgbm_params, sort_keys=True
                    ),
                    "cv_score": score_mean,
                    "cv_score_std": score_std,
                    "cv_robust_score": (
                        score_mean - 0.25 * score_std
                        if np.isfinite(score_mean)
                        else -np.inf
                    ),
                    "cv_mae": aggregate["mae"],
                    "cv_rmse": aggregate["rmse"],
                    "cv_price_mae": aggregate["price_mae"],
                    "cv_balanced_accuracy": aggregate[
                        "balanced_accuracy"
                    ],
                    "feature_preparation_seconds": preparation_seconds,
                    "fit_seconds": time.perf_counter() - started,
                    "error": error,
                }
            )

    trials = pd.DataFrame(trial_rows)
    best_index = trials["cv_robust_score"].idxmax()
    trials["selected"] = trials.index == best_index
    best = trials.loc[best_index]
    return (
        json.loads(best["egarch_params"]),
        json.loads(best["lightgbm_params"]),
        trials,
        best.to_dict(),
    )
