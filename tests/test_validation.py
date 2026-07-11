import numpy as np
import pandas as pd

from src.targets import create_multihorizon_targets
from src.validation import PurgedWalkForwardSplit, purged_train_validation_test_split


def frame(n=500):
    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    return create_multihorizon_targets(pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=n), "close": close}), (20, 40, 60))


def test_outer_boundaries_are_purged():
    data = frame()
    for horizon in (20, 40, 60):
        train, valid, test = purged_train_validation_test_split(data, horizon)
        assert train[f"target_end_date_{horizon}"].max() < valid["date"].min()
        assert valid[f"target_end_date_{horizon}"].max() < test["date"].min()


def test_inner_walk_forward_is_purged():
    data = frame().dropna(subset=["target_60"]).reset_index(drop=True)
    for train_idx, valid_idx in PurgedWalkForwardSplit(3).split(data, 60):
        assert data.loc[train_idx, "target_end_date_60"].max() < data.loc[valid_idx, "date"].min()

