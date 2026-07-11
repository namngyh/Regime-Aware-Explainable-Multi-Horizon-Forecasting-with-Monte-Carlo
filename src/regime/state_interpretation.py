"""Economic state naming from statistics, not arbitrary state indexes."""
from __future__ import annotations
import pandas as pd


def interpret_states(diagnostics: pd.DataFrame) -> dict[int, str]:
    d = diagnostics.set_index("state"); mapping = {int(i): f"State_{int(i)}" for i in d.index}
    if len(d) >= 3:
        stress = int((d["volatility"].rank(pct=True) - d["mean_return"].rank(pct=True)).idxmax())
        bull = int(d.drop(index=stress)["mean_return"].idxmax()); bear = int(d.drop(index=[stress, bull], errors="ignore")["mean_return"].idxmin())
        mapping[stress] = "Stress"; mapping[bull] = "Bull"; mapping[bear] = "Bear"
        remaining = [i for i in d.index if mapping[int(i)].startswith("State_")]
        for i in remaining: mapping[int(i)] = "Sideway" if abs(d.loc[i, "mean_return"]) <= d["mean_return"].abs().median() else "Recovery"
    return mapping

