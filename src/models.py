from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@dataclass
class PredictionBundle:
    model: str
    pred_direction: np.ndarray
    pred_return: np.ndarray
    score_up: np.ndarray
    extra: dict


def macd_bullish_signal(close, fast=8, slow=24, signal=9):
    close = pd.Series(close)
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(
        span=slow, adjust=False
    ).mean()
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return (macd > signal_line).astype(int)


def fit_predict_macd(train_df, test_df, horizon=20, signal_col="macd_selected"):
    target = f"future_return_{horizon}d"
    mapping = train_df.groupby(signal_col)[target].mean().to_dict()
    fallback = train_df[target].mean()
    pred_direction = test_df[signal_col].astype(int).to_numpy()
    pred_return = np.array(
        [mapping.get(int(value), fallback) for value in pred_direction]
    )
    return PredictionBundle(
        "MACD 8-24-9",
        pred_direction,
        pred_return,
        pred_direction.astype(float),
        {"signal_return_map": mapping},
    )


def fit_predict_hmm(
    train_df,
    test_df,
    feature_cols,
    horizon=20,
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
    train_states = model.predict(x_train)
    test_states = model.predict(x_test)
    target = f"future_return_{horizon}d"
    state_means = (
        pd.DataFrame(
            {"state": train_states, "target": train_df[target].to_numpy()}
        )
        .groupby("state")["target"]
        .mean()
    )
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
            "state_mean_return": state_means.to_dict(),
        },
    )
