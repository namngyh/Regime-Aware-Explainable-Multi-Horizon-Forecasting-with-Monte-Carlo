import numpy as np
import pandas as pd

from src.targets import create_multihorizon_targets


def sample(n=200):
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "close": 100 * np.exp(np.arange(n) * 0.001)})


def test_target_end_dates_and_stress_override():
    out = create_multihorizon_targets(sample(), (20, 40, 60))
    assert out.loc[0, "target_end_date_20"] == out.loc[20, "date"]
    assert {"Bull", "Sideway", "Bear", "Stress"}.issuperset(set(out["target_20"].dropna().unique()))
    assert out["target_60"].notna().sum() == 120

