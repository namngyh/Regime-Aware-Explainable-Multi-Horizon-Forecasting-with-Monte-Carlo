"""Moving block bootstrap for metric differences."""

from __future__ import annotations

import numpy as np
import pandas as pd


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Sample consecutive blocks until n positions are obtained."""
    if n <= 0:
        return np.array([], dtype=int)
    starts = np.arange(max(n - block_length + 1, 1))
    out: list[int] = []
    while len(out) < n:
        s = int(rng.choice(starts))
        out.extend(range(s, min(s + block_length, n)))
    return np.asarray(out[:n], dtype=int)


def bootstrap_mean_ci(values: np.ndarray, replicates: int = 200, block_length: int = 20, seed: int = 42) -> tuple[float, float, float]:
    """Moving-block confidence interval for a mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = [float(values[moving_block_indices(len(values), block_length, rng)].mean()) for _ in range(replicates)]
    return float(values.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def bootstrap_difference_frame(rows: list[dict[str, object]], replicates: int, block_length: int, seed: int = 42) -> pd.DataFrame:
    out = []
    for row in rows:
        mean, lo, hi = bootstrap_mean_ci(row["diff"], replicates=replicates, block_length=block_length, seed=seed)
        out.append({k: v for k, v in row.items() if k != "diff"} | {"mean_diff": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)
