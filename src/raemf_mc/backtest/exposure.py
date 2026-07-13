"""Exposure mapping and backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPOSURE_MAP = {"Risk-on": 1.0, "Neutral": 0.5, "Risk-off": 0.0, "Uncertain": 0.25}


def labels_to_exposure(labels: pd.Series) -> pd.Series:
    return labels.map(EXPOSURE_MAP).astype(float)


def backtest_exposure(close: pd.Series, exposure: pd.Series, cost_bps: float = 10) -> pd.DataFrame:
    ret = np.log(close / close.shift(1)).fillna(0.0)
    exp = exposure.reindex(close.index).fillna(0.0)
    turnover = exp.diff().abs().fillna(exp.abs())
    strategy = exp.shift(1).fillna(0.0) * ret - turnover * (cost_bps / 10000.0)
    return pd.DataFrame({"return": ret, "exposure": exp, "turnover": turnover, "strategy_return": strategy})
