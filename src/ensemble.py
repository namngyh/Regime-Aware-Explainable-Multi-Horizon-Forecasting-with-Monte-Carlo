"""Out-of-fold weight selection for a causal HMM-Random Forest ensemble."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.metrics import classification_metrics, regression_metrics
from src.models import (
    PredictionBundle,
    fit_predict_hmm,
    fit_predict_random_forest,
)
from src.tuning import forecast_score


def tune_oof_hmm_rf_ensemble(
    frame,
    feature_cols,
    horizon,
    hmm_params,
    random_forest_params,
    n_splits=6,
):
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    fold_data = []
    oof_rows = []
    target_return = f"future_return_{horizon}d"
    target_up = f"future_up_{horizon}d"
    for fold, (train_index, valid_index) in enumerate(splitter.split(frame), start=1):
        train = frame.iloc[train_index]
        valid = frame.iloc[valid_index]
        hmm = fit_predict_hmm(
            train,
            valid,
            feature_cols,
            horizon,
            hmm_params=hmm_params,
        )
        forest = fit_predict_random_forest(
            train,
            valid,
            feature_cols,
            horizon,
            params=random_forest_params,
        )
        fold_data.append((fold, valid, hmm, forest))
        for index, date in enumerate(valid["date"]):
            oof_rows.append(
                {
                    "fold": fold,
                    "date": date,
                    "actual_return": valid[target_return].iloc[index],
                    "hmm_pred_return": hmm.pred_return[index],
                    "random_forest_pred_return": forest.pred_return[index],
                }
            )

    trial_rows = []
    for hmm_weight in np.linspace(0, 1, 101):
        fold_scores = []
        fold_rmse = []
        for fold, valid, hmm, forest in fold_data:
            pred_return = (
                hmm_weight * hmm.pred_return
                + (1 - hmm_weight) * forest.pred_return
            )
            pred_direction = (pred_return > 0).astype(int)
            score_up = (
                hmm_weight * hmm.score_up
                + (1 - hmm_weight) * forest.score_up
            )
            classification = classification_metrics(
                valid[target_up].astype(int), pred_direction, score_up
            )
            regression = regression_metrics(
                valid[target_return], pred_return, valid["close"]
            )
            fold_scores.append(forecast_score(classification, regression))
            fold_rmse.append(regression["rmse"])
        mean_score = float(np.mean(fold_scores))
        score_std = float(np.std(fold_scores))
        trial_rows.append(
            {
                "hmm_weight": hmm_weight,
                "random_forest_weight": 1 - hmm_weight,
                "cv_score": mean_score,
                "cv_score_std": score_std,
                "cv_robust_score": mean_score - 0.25 * score_std,
                "cv_rmse": float(np.mean(fold_rmse)),
            }
        )
    trials = pd.DataFrame(trial_rows)
    best = trials.loc[trials["cv_robust_score"].idxmax()].to_dict()
    trials["selected"] = np.isclose(trials["hmm_weight"], best["hmm_weight"])
    return best, trials, pd.DataFrame(oof_rows)


def combine_hmm_random_forest(hmm_bundle, forest_bundle, weights):
    hmm_weight = float(weights["hmm_weight"])
    forest_weight = float(weights["random_forest_weight"])
    pred_return = (
        hmm_weight * hmm_bundle.pred_return
        + forest_weight * forest_bundle.pred_return
    )
    pred_direction = (pred_return > 0).astype(int)
    score_up = (
        hmm_weight * hmm_bundle.score_up
        + forest_weight * forest_bundle.score_up
    )
    return PredictionBundle(
        "OOF HMM-RF Ensemble",
        pred_direction,
        pred_return,
        score_up,
        {
            "hmm_weight": hmm_weight,
            "random_forest_weight": forest_weight,
        },
    )
