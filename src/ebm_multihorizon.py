"""Separate multiclass EBM forecaster for each horizon."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.preprocessing import LabelEncoder

@dataclass
class EBMResult:
    horizon: int
    model: ExplainableBoostingClassifier
    encoder: LabelEncoder
    raw_probabilities: np.ndarray
    feature_names: list[str]


def fit_ebm(train_x: pd.DataFrame, train_y: pd.Series, eval_x: pd.DataFrame, horizon: int, params: dict | None = None, seed: int = 42) -> EBMResult:
    """Fit one horizon-specific EBM; callers own purging and calibration scope."""
    selected = {"max_bins": 128, "max_rounds": 500, "learning_rate": .03, "min_samples_leaf": 20, "outer_bags": 4, "inner_bags": 0, "interactions": 6, "max_leaves": 3, "validation_size": .15, "early_stopping_rounds": 50}
    selected.update(params or {})
    encoder = LabelEncoder(); y = encoder.fit_transform(train_y)
    model = ExplainableBoostingClassifier(**selected, random_state=seed, n_jobs=-1)
    model.fit(train_x, y)
    return EBMResult(horizon, model, encoder, model.predict_proba(eval_x), list(train_x.columns))
