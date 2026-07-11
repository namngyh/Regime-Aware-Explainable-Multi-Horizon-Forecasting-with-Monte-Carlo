"""Student-t volatility model comparison and causal multi-step simulation."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from arch import arch_model

@dataclass
class VolatilityResult:
    name: str; fitted: object; diagnostics: dict[str, float]; forecasts: dict[int, float]


def _spec(name: str) -> dict[str, object]:
    return {"EGARCH-t": dict(vol="EGARCH", p=1, o=1, q=1), "GJR-GARCH-t": dict(vol="GARCH", p=1, o=1, q=1), "GARCH-t": dict(vol="GARCH", p=1, o=0, q=1)}[name]


def fit_volatility_candidate(returns: pd.Series, name: str = "EGARCH-t", seed: int = 42, simulations: int = 1000) -> VolatilityResult:
    values = returns.dropna().to_numpy() * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = arch_model(values, mean="Constant", dist="t", rescale=False, **_spec(name)).fit(disp="off", show_warning=False)
        forecast = fitted.forecast(horizon=60, method="simulation", simulations=simulations, rng=np.random.default_rng(seed).standard_normal, reindex=False)
    variance = np.asarray(forecast.variance)[-1] / 10000
    cond = np.asarray(fitted.conditional_volatility) / 100
    realized = pd.Series(values / 100).pow(2).rolling(20).mean().shift(-20).to_numpy()
    valid = np.isfinite(realized)
    pred = cond**2
    qlike = np.mean(np.log(np.clip(pred[valid], 1e-12, None)) + realized[valid] / np.clip(pred[valid], 1e-12, None))
    forecasts = {h: float(np.sqrt(variance[:h].sum())) for h in (1, 20, 40, 60)}
    diag = {"aic": fitted.aic, "bic": fitted.bic, "qlike_in_sample_proxy": qlike, "log_likelihood": fitted.loglikelihood, "converged": float(getattr(fitted, "convergence_flag", 1) == 0)}
    return VolatilityResult(name, fitted, diag, forecasts)


def select_volatility_model(returns: pd.Series, candidates=("EGARCH-t", "GJR-GARCH-t", "GARCH-t"), seed: int = 42, simulations: int = 1000) -> tuple[VolatilityResult, pd.DataFrame]:
    results = [fit_volatility_candidate(returns, name, seed, simulations) for name in candidates]
    report = pd.DataFrame([{"model": r.name, **r.diagnostics, **{f"vol_{h}": v for h, v in r.forecasts.items()}} for r in results])
    best = min(results, key=lambda r: (not bool(r.diagnostics["converged"]), r.diagnostics["qlike_in_sample_proxy"]))
    return best, report

