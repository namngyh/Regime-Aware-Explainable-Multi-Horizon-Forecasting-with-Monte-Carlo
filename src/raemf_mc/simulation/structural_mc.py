"""State-conditioned structural Monte Carlo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raemf_mc.simulation.risk_metrics import max_drawdown


def simulate_paths(
    last_price: float,
    current_state_prob: np.ndarray,
    transition: np.ndarray,
    state_mean: np.ndarray,
    sigma: float,
    horizon: int,
    paths: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed + horizon)
    n_states = len(current_state_prob)
    sim = np.empty((paths, horizon + 1), dtype=float)
    sim[:, 0] = last_price
    states = rng.choice(n_states, size=paths, p=current_state_prob / current_state_prob.sum())
    for t in range(1, horizon + 1):
        means = state_mean[states]
        shocks = rng.standard_t(df=6, size=paths) * sigma
        ret = np.clip(means + shocks, -0.25, 0.25)
        sim[:, t] = np.maximum(sim[:, t - 1] * np.exp(ret), 1e-6)
        for i in range(paths):
            states[i] = rng.choice(n_states, p=transition[states[i]] / transition[states[i]].sum())
    terminal_return = np.log(sim[:, -1] / last_price)
    dd = max_drawdown(sim)
    summary = pd.DataFrame(
        {
            "horizon": [horizon],
            "expected_return": [float(terminal_return.mean())],
            "median_return": [float(np.median(terminal_return))],
            "q01": [float(np.quantile(terminal_return, 0.01))],
            "q05": [float(np.quantile(terminal_return, 0.05))],
            "q25": [float(np.quantile(terminal_return, 0.25))],
            "q50": [float(np.quantile(terminal_return, 0.50))],
            "q75": [float(np.quantile(terminal_return, 0.75))],
            "q95": [float(np.quantile(terminal_return, 0.95))],
            "q99": [float(np.quantile(terminal_return, 0.99))],
            "prob_negative": [float((terminal_return < 0).mean())],
            "prob_drawdown_gt_10pct": [float((dd < -0.10).mean())],
            "var_95": [float(-np.quantile(terminal_return, 0.05))],
            "cvar_95": [float(-terminal_return[terminal_return <= np.quantile(terminal_return, 0.05)].mean())],
            "max_drawdown_mean": [float(dd.mean())],
        }
    )
    return sim, summary
