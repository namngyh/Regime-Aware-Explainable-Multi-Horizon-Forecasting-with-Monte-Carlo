"""Regime-transition and volatility-conditioned Student-t scenarios."""
from __future__ import annotations
import numpy as np
import pandas as pd


def structural_monte_carlo(initial_price: float, initial_state_probability: np.ndarray, transition: np.ndarray, regime_mean: np.ndarray, regime_scale: np.ndarray, horizon: int=60, paths: int=1000, degrees_freedom: float=6., seed: int=42) -> dict[str,np.ndarray]:
    rng=np.random.default_rng(seed); k=len(initial_state_probability)
    states=np.empty((paths,horizon),int); returns=np.empty((paths,horizon),float)
    states[:,0]=rng.choice(k,size=paths,p=initial_state_probability)
    innovations=rng.standard_t(degrees_freedom,size=(paths,horizon))*np.sqrt((degrees_freedom-2)/degrees_freedom)
    for t in range(horizon):
        if t: states[:,t]=np.array([rng.choice(k,p=transition[s]) for s in states[:,t-1]])
        s=states[:,t]; returns[:,t]=regime_mean[s]+regime_scale[s]*innovations[:,t]
    prices=initial_price*np.exp(np.cumsum(returns,axis=1)); return {"states":states,"returns":returns,"prices":prices}


def classify_paths(prices: np.ndarray, initial_price: float, scale: float, threshold: float=.5, stress_lambda: float=1.5) -> np.ndarray:
    terminal=np.log(prices[:,-1]/initial_price); adverse=(prices.min(1)/initial_price-1)
    labels=np.where(terminal>threshold*scale,"Bull",np.where(terminal<-threshold*scale,"Bear","Sideway")); labels[adverse < -stress_lambda*scale]="Stress"; return labels


def reweight_paths(labels: np.ndarray, target_probabilities: dict[str,float]) -> np.ndarray:
    weights=np.zeros(len(labels));
    for label,target in target_probabilities.items():
        mask=labels==label; weights[mask]=target/max(mask.sum(),1)
    return weights/weights.sum() if weights.sum() else np.full(len(labels),1/len(labels))


def summarize_scenarios(simulation: dict[str,np.ndarray], initial_price: float, horizons=(20,40,60), weights: np.ndarray|None=None) -> pd.DataFrame:
    prices=simulation["prices"]; rows=[]
    for h in horizons:
        terminal=prices[:,h-1]; ret=terminal/initial_price-1; dd=(prices[:,:h].min(1)/initial_price-1)
        q=lambda x,p: float(np.quantile(x,p)) if weights is None else float(x[np.argsort(x)][np.searchsorted(np.cumsum(weights[np.argsort(x)]),p,side="left").clip(max=len(x)-1)])
        row={"horizon":h,"expected_return":float(np.average(ret,weights=weights)),"median_return":q(ret,.5),"probability_return_positive":float(np.average(ret>0,weights=weights)),"probability_return_below_minus_5pct":float(np.average(ret<-.05,weights=weights)),"probability_max_drawdown_below_minus_8pct":float(np.average(dd<-.08,weights=weights)),"expected_max_drawdown":float(np.average(dd,weights=weights))}
        for p in (.01,.05,.10,.25,.50,.75,.90,.95,.99): row[f"return_q{int(p*100):02d}"]=q(ret,p)
        for p in (.05,.10,.25,.50,.75,.90,.95): row[f"price_q{int(p*100):02d}"]=q(terminal,p)
        rows.append(row)
    return pd.DataFrame(rows)

