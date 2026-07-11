"""Purged chronological validation and explicit leakage assertions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    """Expanding walk-forward folds purged using target end dates."""

    n_splits: int = 4
    min_train_size: int | None = None

    def split(self, frame: pd.DataFrame, horizon: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = pd.to_datetime(frame["date"]).reset_index(drop=True)
        end_dates = pd.to_datetime(frame[f"target_end_date_{horizon}"]).reset_index(drop=True)
        n = len(frame)
        min_train = self.min_train_size or max(horizon * 5, n // (self.n_splits + 2))
        remaining = n - min_train
        fold_size = max(1, remaining // (self.n_splits + 1))
        for fold in range(self.n_splits):
            valid_start = min_train + fold * fold_size
            valid_end = min(valid_start + fold_size, n)
            if valid_start >= n or valid_end <= valid_start:
                continue
            boundary = dates.iloc[valid_start]
            train_idx = np.flatnonzero((np.arange(n) < valid_start) & (end_dates < boundary).to_numpy())
            valid_idx = np.arange(valid_start, valid_end)
            if len(train_idx) and len(valid_idx):
                yield train_idx, valid_idx


def purged_train_validation_test_split(
    frame: pd.DataFrame,
    horizon: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create outer splits and purge labels that cross either boundary."""
    data = frame.sort_values("date").reset_index(drop=True)
    labeled = data[data[f"target_{horizon}"].notna()].copy()
    n = len(labeled)
    train_cut = int(n * train_ratio)
    valid_cut = int(n * (train_ratio + validation_ratio))
    validation_start = pd.Timestamp(labeled.iloc[train_cut]["date"])
    test_start = pd.Timestamp(labeled.iloc[valid_cut]["date"])
    target_end = pd.to_datetime(labeled[f"target_end_date_{horizon}"])
    train = labeled.iloc[:train_cut].loc[target_end.iloc[:train_cut] < validation_start].copy()
    validation = labeled.iloc[train_cut:valid_cut].loc[target_end.iloc[train_cut:valid_cut] < test_start].copy()
    test = labeled.iloc[valid_cut:].copy()
    validate_no_label_overlap(train, validation_start, horizon)
    validate_outer_test_purge(pd.concat([train, validation]), test_start, horizon)
    return train, validation, test


def validate_no_label_overlap(train: pd.DataFrame, boundary: pd.Timestamp, horizon: int) -> bool:
    """Assert that no training target reads prices at/after the next split."""
    ends = pd.to_datetime(train[f"target_end_date_{horizon}"])
    if not bool((ends < pd.Timestamp(boundary)).all()):
        raise AssertionError("training labels overlap the validation boundary")
    return True


def validate_outer_test_purge(train_validation: pd.DataFrame, test_start: pd.Timestamp, horizon: int) -> bool:
    """Assert purge across the outer train-validation/test boundary."""
    ends = pd.to_datetime(train_validation[f"target_end_date_{horizon}"])
    if not bool((ends < pd.Timestamp(test_start)).all()):
        raise AssertionError("train-validation labels overlap the outer test")
    return True


def validate_no_future_feature_leakage(before: pd.DataFrame, after: pd.DataFrame, cutoff: int, columns: list[str]) -> bool:
    """Assert historical features are invariant to future raw-data changes."""
    pd.testing.assert_frame_equal(before.loc[:cutoff, columns], after.loc[:cutoff, columns], check_dtype=False)
    return True


def validate_scaler_scope(scaler: object, train: pd.DataFrame, columns: list[str]) -> bool:
    """Assert StandardScaler statistics equal train-only statistics."""
    expected = train[columns].mean().to_numpy()
    if not np.allclose(getattr(scaler, "mean_"), expected, rtol=1e-8, atol=1e-10):
        raise AssertionError("scaler was not fit exclusively on train")
    return True


def validate_calibration_scope(calibration_dates: pd.Series, test_start: pd.Timestamp) -> bool:
    """Assert calibrator observations precede test start."""
    if not bool((pd.to_datetime(calibration_dates) < pd.Timestamp(test_start)).all()):
        raise AssertionError("calibration includes test observations")
    return True


def validate_prediction_timestamp(signal_dates: pd.Series, applied_dates: pd.Series) -> bool:
    """Assert close-t signals are applied strictly after their timestamps."""
    if not bool((pd.to_datetime(applied_dates).to_numpy() > pd.to_datetime(signal_dates).to_numpy()).all()):
        raise AssertionError("signal was applied without a one-session lag")
    return True


def validate_no_future_feature_leakage_names(feature_columns: list[str]) -> bool:
    """Reject explicit target/future columns from a model feature matrix."""
    forbidden = ("future_", "target_", "mae_path_", "mfe_path_", "forward_log_return_")
    bad = [name for name in feature_columns if name.startswith(forbidden)]
    if bad:
        raise AssertionError(f"future-dependent columns used as features: {bad}")
    return True

