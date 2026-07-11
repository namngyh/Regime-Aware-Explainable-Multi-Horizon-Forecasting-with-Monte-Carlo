from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass
class PredictionBundle:
    model: str
    pred_direction: np.ndarray
    pred_return: np.ndarray
    score_up: np.ndarray
    extra: dict


def _causal_hmm_states(model, observations, initial_posterior=None):
    """Filter states forward so time t never uses observations after t."""
    log_likelihood = model._compute_log_likelihood(observations)
    states = np.empty(len(observations), dtype=int)
    posteriors = np.empty((len(observations), model.n_components), dtype=float)
    previous = initial_posterior
    for index, emission in enumerate(log_likelihood):
        if previous is None:
            prior = model.startprob_
        else:
            prior = previous @ model.transmat_
        log_weight = np.log(np.clip(prior, 1e-300, None)) + emission
        log_weight -= np.max(log_weight)
        posterior = np.exp(log_weight)
        posterior /= posterior.sum()
        states[index] = int(np.argmax(posterior))
        posteriors[index] = posterior
        previous = posterior
    return states, posteriors


def fit_causal_hmm_context(
    train_df,
    test_df,
    feature_cols,
    random_state=42,
    hmm_params=None,
):
    hmm_features = [
        column
        for column in [
            "log_ret_1",
            "ret_5",
            "ret_20",
            "vol_20",
            "vol_60",
            "range_pct",
            "macd_hist",
            "rsi_14",
        ]
        if column in feature_cols
    ]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[hmm_features])
    x_test = scaler.transform(test_df[hmm_features])
    params = {
        "n_components": 4,
        "covariance_type": "tied",
        "n_iter": 400,
        "random_state": random_state,
    }
    params.update(hmm_params or {})
    model = GaussianHMM(**params)
    model.fit(x_train)
    train_states, train_posteriors = _causal_hmm_states(model, x_train)
    test_states, test_posteriors = _causal_hmm_states(
        model, x_test, initial_posterior=train_posteriors[-1]
    )
    return {
        "model": model,
        "scaler": scaler,
        "features": hmm_features,
        "train_states": train_states,
        "test_states": test_states,
        "train_probabilities": train_posteriors,
        "test_probabilities": test_posteriors,
    }


def fit_predict_hmm(
    train_df,
    test_df,
    feature_cols,
    horizon=20,
    random_state=42,
    hmm_params=None,
):
    context = fit_causal_hmm_context(
        train_df,
        test_df,
        feature_cols,
        random_state=random_state,
        hmm_params=hmm_params,
    )
    model = context["model"]
    scaler = context["scaler"]
    hmm_features = context["features"]
    train_states = context["train_states"]
    test_states = context["test_states"]
    train_posteriors = context["train_probabilities"]
    test_posteriors = context["test_probabilities"]
    target = f"future_return_{horizon}d"
    state_means = (
        pd.DataFrame(
            {"state": train_states, "target": train_df[target].to_numpy()}
        )
        .groupby("state")["target"]
        .mean()
    )
    state_counts = pd.Series(train_states).value_counts().sort_index().to_dict()
    fallback = train_df[target].mean()
    pred_return = np.array(
        [state_means.get(state, fallback) for state in test_states]
    )
    pred_direction = (pred_return > 0).astype(int)
    score_up = pd.Series(pred_return).rank(pct=True).to_numpy()
    return PredictionBundle(
        "HMM Regime",
        pred_direction,
        pred_return,
        score_up,
        {
            "model": model,
            "scaler": scaler,
            "features": hmm_features,
            "predicted_states": test_states,
            "state_probabilities": test_posteriors,
            "state_mean_return": state_means.to_dict(),
            "state_counts": state_counts,
        },
    )


def fit_predict_svr(
    train_df,
    test_df,
    feature_cols,
    horizon=20,
    params=None,
):
    target_return = f"future_return_{horizon}d"
    selected = {
        "kernel": "rbf",
        "C": 0.5,
        "epsilon": 0.005,
        "gamma": "scale",
    }
    selected.update(params or {})
    estimator = make_pipeline(StandardScaler(), SVR(**selected))
    estimator.fit(train_df[feature_cols], train_df[target_return])
    pred_return = estimator.predict(test_df[feature_cols])
    pred_direction = (pred_return > 0).astype(int)
    scale = max(float(train_df[target_return].std()), 1e-12)
    score_up = 1 / (1 + np.exp(-pred_return / scale))
    return PredictionBundle(
        "SVR",
        pred_direction,
        pred_return,
        score_up,
        {"estimator": estimator},
    )


def fit_predict_random_forest(
    train_df,
    test_df,
    feature_cols,
    horizon=20,
    random_state=42,
    params=None,
):
    target_return = f"future_return_{horizon}d"
    target_up = f"future_up_{horizon}d"
    selected = {
        "n_estimators": 400,
        "max_depth": 5,
        "min_samples_leaf": 30,
        "max_features": "sqrt",
    }
    selected.update(params or {})
    common = {
        **selected,
        "random_state": random_state,
        "n_jobs": -1,
    }
    classifier = RandomForestClassifier(
        **common,
        class_weight="balanced_subsample",
    )
    regressor = RandomForestRegressor(**common)
    classifier.fit(train_df[feature_cols], train_df[target_up].astype(int))
    regressor.fit(train_df[feature_cols], train_df[target_return])
    pred_direction = classifier.predict(test_df[feature_cols]).astype(int)
    score_up = classifier.predict_proba(test_df[feature_cols])[:, 1]
    pred_return = regressor.predict(test_df[feature_cols])
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": regressor.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return PredictionBundle(
        "Random Forest",
        pred_direction,
        pred_return,
        score_up,
        {
            "classifier": classifier,
            "regressor": regressor,
            "feature_importance": importance,
        },
    )
