"""MACD probabilistic rule baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raemf_mc import CLASS_ORDER


def macd_probabilities(close: pd.Series, volatility: pd.Series, smooth: float = 0.12) -> pd.DataFrame:
    """Return smoothed class probabilities from a transparent MACD rule."""
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd - signal
    hist_z = hist / (hist.rolling(60, min_periods=20).std() + 1e-12)
    vol_z = volatility / (volatility.rolling(252, min_periods=40).median() + 1e-12)
    pred = np.full(len(close), "Sideway", dtype=object)
    pred[(hist_z > 0.35) & (macd > signal)] = "Bull"
    pred[(hist_z < -0.35)] = "Bear"
    pred[(hist_z < -0.80) & (vol_z > 1.15)] = "Stress"
    proba = np.full((len(close), len(CLASS_ORDER)), smooth / (len(CLASS_ORDER) - 1))
    for i, cls in enumerate(pred):
        proba[i, CLASS_ORDER.index(cls)] = 1.0 - smooth
    return pd.DataFrame(proba, index=close.index, columns=[f"prob_{c}" for c in CLASS_ORDER])
