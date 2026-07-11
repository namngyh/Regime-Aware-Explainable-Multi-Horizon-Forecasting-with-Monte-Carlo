"""Rule-based multiclass MACD benchmark including an explicit Stress extension."""
from __future__ import annotations
import numpy as np
import pandas as pd


def macd_rule(frame: pd.DataFrame, volatility_threshold: float|None=None) -> pd.Series:
    hist=frame["macd_hist"]; threshold=max(float(hist.abs().median())*.25,1e-12); vol=frame["vol_20"]
    vcut=float(volatility_threshold if volatility_threshold is not None else vol.quantile(.75))
    labels=np.where(hist>threshold,"Bull",np.where(hist<-threshold,"Bear","Sideway")); labels[(hist<-threshold)&(vol>vcut)]="Stress"
    return pd.Series(labels,index=frame.index,name="prediction")

