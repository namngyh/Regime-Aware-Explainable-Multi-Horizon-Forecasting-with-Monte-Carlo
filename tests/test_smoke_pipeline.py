import pandas as pd

from raemf_mc.features.technical import build_features
from raemf_mc.targets.regime_targets import create_multihorizon_targets


def test_pipeline_components_on_synthetic_small():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=120),
        "open": range(100, 220),
        "high": range(101, 221),
        "low": range(99, 219),
        "close": range(100, 220),
        "volume": range(1000, 1120),
    })
    targeted = create_multihorizon_targets(df)
    features, registry = build_features(targeted)
    assert "target_20" in targeted
    assert len(registry.to_frame()) > 20
    assert len(features) == len(df)
