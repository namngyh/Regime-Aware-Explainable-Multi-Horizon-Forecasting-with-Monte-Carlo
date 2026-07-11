import numpy as np
import pandas as pd

from src.features import make_raemf_features
from src.validation import validate_no_future_feature_leakage


def test_future_mutation_does_not_change_past_features():
    n = 400
    raw = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": np.arange(n)+100., "high": np.arange(n)+101., "low": np.arange(n)+99., "close": np.arange(n)+100., "volume": np.arange(n)+1000.})
    before, columns = make_raemf_features(raw)
    changed = raw.copy(); changed.loc[301:, "close"] *= 3
    after, _ = make_raemf_features(changed)
    assert validate_no_future_feature_leakage(before, after, 300, columns)

