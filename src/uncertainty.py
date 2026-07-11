"""Confidence labels that distinguish probability from certainty."""
from __future__ import annotations
import numpy as np


def uncertainty_snapshot(probabilities: np.ndarray, hmm_confidence: float, bootstrap_std: float=np.nan, feature_drift: float=0., disagreement: float=0.) -> dict[str,float|str]:
    p=np.asarray(probabilities,float); ordered=np.sort(p)
    maximum=float(ordered[-1]); margin=float(ordered[-1]-ordered[-2]); entropy=float(-(np.clip(p,1e-12,1)*np.log(np.clip(p,1e-12,1))).sum()/np.log(len(p)))
    uncertain=maximum<.45 or margin<.10 or feature_drift>2 or disagreement>.35 or hmm_confidence<.30 or (np.isfinite(bootstrap_std) and bootstrap_std>.12)
    if uncertain: label="Uncertain"
    elif maximum>=.65 and margin>=.25 and hmm_confidence>=.70 and (not np.isfinite(bootstrap_std) or bootstrap_std<=.07): label="High"
    elif maximum>=.52 and margin>=.15: label="Medium"
    else: label="Low"
    return {"maximum_probability":maximum,"probability_margin":margin,"ebm_entropy":entropy,"hmm_confidence":hmm_confidence,"bootstrap_std":bootstrap_std,"feature_drift":feature_drift,"model_disagreement":disagreement,"confidence_label":label}

