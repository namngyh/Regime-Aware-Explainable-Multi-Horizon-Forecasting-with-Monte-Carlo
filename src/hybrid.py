"""Causal HMM-EGARCH features with LightGBM quantile forecasts."""

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from lightgbm import LGBMRegressor

from src.models import PredictionBundle, fit_causal_hmm_context


def _egarch_features(train_df, test_df, egarch_params):
    selected = {
        "p": 1,
        "o": 1,
        "q": 1,
        "dist": "t",
    }
    selected.update(egarch_params or {})
    train_return = train_df["ret_1"].fillna(0).to_numpy() * 100
    test_return = test_df["ret_1"].fillna(0).to_numpy() * 100
    combined_return = np.concatenate([train_return, test_return])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = arch_model(
            train_return,
            mean="Constant",
            vol="EGARCH",
            p=selected["p"],
            o=selected["o"],
            q=selected["q"],
            dist=selected["dist"],
            rescale=False,
        ).fit(disp="off", show_warning=False)
        fixed = arch_model(
            combined_return,
            mean="Constant",
            vol="EGARCH",
            p=selected["p"],
            o=selected["o"],
            q=selected["q"],
            dist=selected["dist"],
            rescale=False,
        ).fix(fitted.params)
    daily_vol = np.asarray(fixed.conditional_volatility) / 100
    standardized = combined_return / 100 / np.maximum(daily_vol, 1e-8)
    features = pd.DataFrame(
        {
            "egarch_vol_1d": daily_vol,
            "egarch_vol_20d": daily_vol * np.sqrt(20),
            "egarch_log_vol": np.log(np.maximum(daily_vol, 1e-8)),
            "egarch_standardized_return": standardized,
            "egarch_abs_standardized_return": np.abs(standardized),
        }
    )
    split = len(train_df)
    return (
        features.iloc[:split].reset_index(drop=True),
        features.iloc[split:].reset_index(drop=True),
        fitted,
    )


def prepare_hybrid_features(
    train_df,
    test_df,
    feature_cols,
    hmm_params,
    egarch_params,
):
    context = fit_causal_hmm_context(
        train_df,
        test_df,
        feature_cols,
        hmm_params=hmm_params,
    )
    train_features = train_df[feature_cols].reset_index(drop=True).copy()
    test_features = test_df[feature_cols].reset_index(drop=True).copy()
    for state in range(context["model"].n_components):
        train_features[f"hmm_prob_state_{state}"] = context[
            "train_probabilities"
        ][:, state]
        test_features[f"hmm_prob_state_{state}"] = context[
            "test_probabilities"
        ][:, state]
    train_features["hmm_state"] = context["train_states"]
    test_features["hmm_state"] = context["test_states"]
    train_egarch, test_egarch, fitted_egarch = _egarch_features(
        train_df, test_df, egarch_params
    )
    train_features = pd.concat([train_features, train_egarch], axis=1)
    test_features = pd.concat([test_features, test_egarch], axis=1)
    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    test_features = test_features.replace([np.inf, -np.inf], np.nan)
    medians = train_features.median(numeric_only=True)
    train_features = train_features.fillna(medians).fillna(0)
    test_features = test_features.fillna(medians).fillna(0)
    return train_features, test_features, {
        "hmm_context": context,
        "egarch_fit": fitted_egarch,
        "feature_names": list(train_features.columns),
    }


def _quantile_estimator(params, alpha, random_state=42):
    selected = {
        "n_estimators": 400,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 4.0,
    }
    selected.update(params or {})
    return LGBMRegressor(
        **selected,
        objective="quantile",
        alpha=alpha,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        subsample_freq=1,
    )


def fit_quantile_median(train_features, y_train, test_features, params):
    estimator = _quantile_estimator(params, alpha=0.5)
    estimator.fit(train_features, y_train)
    return estimator.predict(test_features), estimator


def fit_predict_hybrid_quantile(
    train_df,
    test_df,
    feature_cols,
    horizon,
    hmm_params,
    egarch_params,
    lightgbm_params,
):
    train_features, test_features, context = prepare_hybrid_features(
        train_df,
        test_df,
        feature_cols,
        hmm_params,
        egarch_params,
    )
    target = f"future_return_{horizon}d"
    y_train = train_df[target].to_numpy()
    estimators = {}
    raw_predictions = []
    for alpha in (0.1, 0.5, 0.9):
        estimator = _quantile_estimator(lightgbm_params, alpha=alpha)
        estimator.fit(train_features, y_train)
        estimators[alpha] = estimator
        raw_predictions.append(estimator.predict(test_features))
    ordered = np.sort(np.vstack(raw_predictions), axis=0)
    q10, pred_return, q90 = ordered
    pred_direction = (pred_return > 0).astype(int)
    interval_width = np.maximum(q90 - q10, 1e-6)
    score_up = 1 / (1 + np.exp(-pred_return / interval_width))
    importance = pd.DataFrame(
        {
            "feature": train_features.columns,
            "importance": estimators[0.5].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return PredictionBundle(
        "HMM-EGARCH-LightGBM Quantile",
        pred_direction,
        pred_return,
        score_up,
        {
            **context,
            "estimators": estimators,
            "q10": q10,
            "q50": pred_return,
            "q90": q90,
            "feature_importance": importance,
            "egarch_params": egarch_params,
            "lightgbm_params": lightgbm_params,
        },
    )
