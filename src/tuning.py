"""Medium-high, leakage-aware tuning for HMM, SVR and Random Forest."""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.metrics import classification_metrics, regression_metrics
from src.models import (
    fit_predict_hmm,
    fit_predict_random_forest,
    fit_predict_svr,
)


SEARCH_SPACE = {
    "HMM Regime": [
        {
            "n_components": components,
            "covariance_type": covariance,
            "n_iter": 500,
            "tol": tolerance,
            "random_state": seed,
        }
        for components, covariance, tolerance, seed in [
            (2, "diag", 1e-3, 7),
            (2, "diag", 1e-3, 42),
            (2, "diag", 1e-3, 123),
            (3, "diag", 1e-3, 7),
            (3, "diag", 1e-3, 42),
            (4, "diag", 1e-3, 7),
            (4, "diag", 1e-3, 42),
            (5, "diag", 1e-3, 42),
            (6, "diag", 1e-3, 42),
            (2, "tied", 1e-4, 42),
            (3, "tied", 1e-4, 7),
            (3, "tied", 1e-4, 42),
            (4, "tied", 1e-4, 7),
            (4, "tied", 1e-4, 42),
            (5, "tied", 1e-4, 42),
            (6, "tied", 1e-4, 42),
            (3, "full", 1e-3, 42),
            (4, "full", 1e-3, 42),
        ]
    ],
    "SVR": [
        {"kernel": "rbf", "C": c, "epsilon": epsilon, "gamma": gamma}
        for c, epsilon, gamma in [
            (0.05, 0.005, "scale"),
            (0.10, 0.005, "scale"),
            (0.20, 0.005, "scale"),
            (0.50, 0.005, "scale"),
            (1.00, 0.005, "scale"),
            (2.00, 0.005, "scale"),
            (5.00, 0.005, "scale"),
            (10.0, 0.005, "scale"),
            (0.10, 0.010, "scale"),
            (0.20, 0.010, "scale"),
            (0.50, 0.010, "scale"),
            (1.00, 0.010, "scale"),
            (2.00, 0.010, "scale"),
            (5.00, 0.010, "scale"),
            (10.0, 0.010, "scale"),
            (0.20, 0.020, "scale"),
            (0.50, 0.020, "scale"),
            (1.00, 0.020, "scale"),
            (2.00, 0.020, "scale"),
            (0.20, 0.005, 0.01),
            (0.50, 0.005, 0.01),
            (1.00, 0.010, 0.01),
            (2.00, 0.010, 0.01),
            (5.00, 0.010, 0.01),
            (0.20, 0.005, 0.03),
            (0.50, 0.010, 0.03),
            (1.00, 0.010, 0.03),
            (2.00, 0.020, 0.03),
        ]
    ]
    + [
            {"kernel": "linear", "C": 0.005, "epsilon": 0.005},
            {"kernel": "linear", "C": 0.010, "epsilon": 0.010},
            {"kernel": "linear", "C": 0.050, "epsilon": 0.010},
            {"kernel": "linear", "C": 0.100, "epsilon": 0.020},
        ],
    "Random Forest": [
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 50, "max_features": "sqrt"},
        {"n_estimators": 400, "max_depth": 5, "min_samples_leaf": 40, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 5, "min_samples_leaf": 30, "max_features": "sqrt"},
        {"n_estimators": 600, "max_depth": 6, "min_samples_leaf": 24, "max_features": "sqrt"},
        {"n_estimators": 700, "max_depth": 7, "min_samples_leaf": 20, "max_features": "sqrt"},
        {"n_estimators": 800, "max_depth": 8, "min_samples_leaf": 16, "max_features": "sqrt"},
        {"n_estimators": 600, "max_depth": 5, "min_samples_leaf": 30, "max_features": 0.4},
        {"n_estimators": 700, "max_depth": 7, "min_samples_leaf": 20, "max_features": 0.4},
        {"n_estimators": 800, "max_depth": 10, "min_samples_leaf": 12, "max_features": 0.4},
        {"n_estimators": 600, "max_depth": 5, "min_samples_leaf": 30, "max_features": 0.7},
        {"n_estimators": 700, "max_depth": 8, "min_samples_leaf": 16, "max_features": 0.7},
        {"n_estimators": 800, "max_depth": 12, "min_samples_leaf": 10, "max_features": 0.7},
        {"n_estimators": 700, "max_depth": None, "min_samples_leaf": 20, "max_features": "sqrt", "max_samples": 0.8},
        {"n_estimators": 900, "max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt", "max_samples": 0.8},
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
    rows = []
    for fold, (train_index, valid_index) in enumerate(splits, start=1):
        train = frame.iloc[train_index]
        valid = frame.iloc[valid_index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "HMM Regime":
                bundle = fit_predict_hmm(
                    train,
                    valid,
                    feature_cols,
                    horizon,
                    hmm_params=params,
                )
            elif name == "SVR":
                bundle = fit_predict_svr(
                    train,
                    valid,
                    feature_cols,
                    horizon,
                    params=params,
                )
            else:
                bundle = fit_predict_random_forest(
                    train,
                    valid,
                    feature_cols,
                    horizon,
                    params=params,
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
                "r2": regression["r2"],
                "spearman_ic": regression["spearman_ic"],
                "price_mae": regression["price_mae"],
                "price_rmse": regression["price_rmse"],
                "balanced_accuracy": classification["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def tune_horizon(frame, feature_cols, horizon=20, n_splits=6):
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
                        "r2",
                        "spearman_ic",
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
                "cv_robust_score": (
                    aggregate["cv_score"] - 0.25 * score_std
                    if np.isfinite(aggregate["cv_score"])
                    else -np.inf
                ),
                "cv_mae": aggregate["mae"],
                "cv_rmse": aggregate["rmse"],
                "cv_r2": aggregate["r2"],
                "cv_spearman_ic": aggregate["spearman_ic"],
                "cv_price_mae": aggregate["price_mae"],
                "cv_price_rmse": aggregate["price_rmse"],
                "cv_balanced_accuracy": aggregate["balanced_accuracy"],
                "fit_seconds": time.perf_counter() - started,
                "error": error,
            }
            trial_rows.append(row)
            model_trials.append((row, params))

        best_trial, best_params = max(
            model_trials, key=lambda item: item[0]["cv_robust_score"]
        )
        selected[name] = best_params
        best_rows.append(
            {**best_trial, "selected_params": best_trial["params_json"]}
        )
        print(
            f"  {name}: candidate {best_trial['candidate_id']} "
            f"(robust CV={best_trial['cv_robust_score']:.4f})"
        )

    selected_keys = {
        (row["model"], row["candidate_id"]) for row in best_rows
    }
    for row in trial_rows:
        row["selected"] = (row["model"], row["candidate_id"]) in selected_keys

    return (
        selected["HMM Regime"],
        selected["SVR"],
        selected["Random Forest"],
        pd.DataFrame(trial_rows),
        pd.DataFrame(best_rows),
    )
