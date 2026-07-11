"""Central market-filter rules and a stock-position adjustment interface."""
from __future__ import annotations
import numpy as np


def derive_market_filter(probabilities: dict[int,dict[str,float]], confidence: str, drawdown_probability: float, volatility_percentile: float=0.5) -> dict[str,object]:
    p20,p40,p60=probabilities[20],probabilities[40],probabilities[60]
    if confidence=="Uncertain": state="Uncertain"
    elif p40["Bull"]>=.55 and p60["Bull"]>=.55 and p20["Stress"]<=.15 and drawdown_probability<.30: state="Risk-on"
    elif p20["Bear"]+p20["Stress"]>=.50 or p40["Stress"]>=.25 or drawdown_probability>=.45: state="Risk-off"
    else: state="Neutral"
    score=np.mean([p20["Bull"]-p20["Bear"]-1.5*p20["Stress"],p40["Bull"]-p40["Bear"]-1.5*p40["Stress"],p60["Bull"]-p60["Bear"]-1.5*p60["Stress"]])
    base={"Risk-on":1.,"Neutral":.55,"Risk-off":.15,"Uncertain":.30}[state]
    exposure=float(np.clip(base*(1-.5*volatility_percentile)*(1-.7*drawdown_probability),0,1))
    return {"market_state":state,"market_score":float(score),"exposure_multiplier":exposure,"allowed_strategies":"long trend" if state=="Risk-on" else "defensive/selective","restricted_strategies":"leveraged long" if state!="Risk-on" else "none","risk_flags":",".join(x for x,v in (("high_drawdown",drawdown_probability>.3),("uncertain",state=="Uncertain")) if v),"human_readable_summary":f"Trạng thái {state}; exposure tham khảo {exposure:.0%}.","technical_explanation":f"score={score:.3f}, drawdown_probability={drawdown_probability:.3f}"}


def adjust_stock_position(base_position: float, exposure_multiplier: float, market_state: str, confidence: str) -> float:
    if base_position<0: raise ValueError("base_position must be non-negative")
    penalty=.75 if confidence in {"Low","Uncertain"} else 1.; return float(base_position*np.clip(exposure_multiplier,0,1)*penalty)

