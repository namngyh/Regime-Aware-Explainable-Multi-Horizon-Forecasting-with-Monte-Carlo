"""Moving block bootstrap preserving local serial dependence."""
from __future__ import annotations
import numpy as np


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if not 1 <= block_length <= n: raise ValueError("block_length must be between 1 and n")
    starts = rng.integers(0, n-block_length+1, size=int(np.ceil(n/block_length)))
    return np.concatenate([np.arange(s,s+block_length) for s in starts])[:n]


def bootstrap_metric_interval(values: np.ndarray, confidence: float = .95) -> dict[str,float]:
    values=np.asarray(values,float); alpha=(1-confidence)/2
    return {"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)),"lower":float(np.quantile(values,alpha)),"upper":float(np.quantile(values,1-alpha))}

