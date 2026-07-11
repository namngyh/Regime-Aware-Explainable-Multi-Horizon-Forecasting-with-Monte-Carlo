"""Train-only scaled Gaussian HMM with forward filtered probabilities."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

HMM_FEATURES = ["ret_5", "ret_20", "ret_60", "vol_20", "vol_60", "rolling_max_drawdown_60", "trend_strength", "range_volatility"]


@dataclass
class FilteredHMMResult:
    model: GaussianHMM
    scaler: StandardScaler
    features: list[str]
    train_probabilities: np.ndarray
    eval_probabilities: np.ndarray
    diagnostics: pd.DataFrame
    converged: bool


def filtered_probabilities(model: GaussianHMM, observations: np.ndarray, initial: np.ndarray | None = None) -> np.ndarray:
    """Compute P(S_t | F_t), never smoothed P(S_t | F_T)."""
    emissions = model._compute_log_likelihood(observations)
    out = np.empty((len(observations), model.n_components))
    previous = initial
    for i, emission in enumerate(emissions):
        prior = model.startprob_ if previous is None else previous @ model.transmat_
        log_weight = np.log(np.clip(prior, 1e-300, None)) + emission
        weight = np.exp(log_weight - log_weight.max()); previous = weight / weight.sum()
        out[i] = previous
    return out


def _state_diagnostics(frame: pd.DataFrame, probabilities: np.ndarray, transition: np.ndarray) -> pd.DataFrame:
    state = probabilities.argmax(axis=1)
    rows = []
    for k in range(probabilities.shape[1]):
        mask = state == k; returns = frame.loc[mask, "log_ret_1"]
        rows.append({"state": k, "observations": int(mask.sum()), "occupancy": float(mask.mean()), "mean_return": returns.mean(), "volatility": returns.std() * np.sqrt(252), "downside_volatility": returns[returns < 0].std() * np.sqrt(252), "drawdown": frame.loc[mask, "rolling_max_drawdown_60"].mean(), "trend_strength": frame.loc[mask, "trend_strength"].mean(), "transition_persistence": transition[k, k], "expected_duration": 1 / max(1 - transition[k, k], 1e-8)})
    return pd.DataFrame(rows)


def fit_filtered_hmm(train: pd.DataFrame, evaluation: pd.DataFrame, n_states: int = 4, seed: int = 11, covariance_type: str = "diag", n_iter: int = 300) -> FilteredHMMResult:
    features = [c for c in HMM_FEATURES if c in train]
    scaler = StandardScaler(); x_train = scaler.fit_transform(train[features]); x_eval = scaler.transform(evaluation[features])
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, random_state=seed, min_covar=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); model.fit(x_train)
    train_p = filtered_probabilities(model, x_train); eval_p = filtered_probabilities(model, x_eval, train_p[-1])
    return FilteredHMMResult(model, scaler, features, train_p, eval_p, _state_diagnostics(train.reset_index(drop=True), train_p, model.transmat_), bool(model.monitor_.converged))


def posterior_features(probabilities: np.ndarray, transition: np.ndarray) -> pd.DataFrame:
    """Create entropy/confidence/duration/switch features without future states."""
    p = np.clip(probabilities, 1e-12, 1); k = p.shape[1]
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(k); states = p.argmax(axis=1)
    duration = np.ones(len(states));
    for i in range(1, len(states)): duration[i] = duration[i-1] + 1 if states[i] == states[i-1] else 1
    out = {f"hmm_prob_state_{i}": p[:, i] for i in range(k)}
    out.update({"hmm_entropy": entropy, "hmm_confidence": 1-entropy, "hmm_state_duration": duration, "hmm_transition_probability": [transition[s, s] for s in states], "hmm_switch_probability": [1-transition[s, s] for s in states]})
    return pd.DataFrame(out)

