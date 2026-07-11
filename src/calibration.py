"""Validation-only probability calibration and calibration metrics."""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class MulticlassCalibrator:
    def __init__(self, method: str = "sigmoid"): self.method, self.models = method, []
    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "MulticlassCalibrator":
        self.models = []
        for k in range(probabilities.shape[1]):
            binary = (y == k).astype(int)
            model = IsotonicRegression(out_of_bounds="clip") if self.method == "isotonic" else LogisticRegression(C=1e6)
            x = probabilities[:, k]
            model.fit(x, binary) if self.method == "isotonic" else model.fit(x[:, None], binary)
            self.models.append(model)
        return self
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        cols = []
        for k, model in enumerate(self.models):
            x = probabilities[:, k]; cols.append(model.predict(x) if self.method == "isotonic" else model.predict_proba(x[:, None])[:, 1])
        out = np.column_stack(cols); return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def calibration_metrics(probabilities: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, float]:
    n, k = probabilities.shape; onehot = np.eye(k)[y]
    brier = np.mean(np.sum((probabilities-onehot)**2, axis=1)); logloss = -np.mean(np.log(np.clip(probabilities[np.arange(n), y], 1e-12, 1)))
    confidence, correct = probabilities.max(1), probabilities.argmax(1) == y; edges = np.linspace(0,1,bins+1); gaps=[]; weights=[]
    for i in range(bins):
        mask=(confidence>=edges[i]) & (confidence<(edges[i+1] if i+1<bins else 1.000001))
        if mask.any(): gaps.append(abs(correct[mask].mean()-confidence[mask].mean())); weights.append(mask.mean())
    return {"multiclass_brier": float(brier), "log_loss": float(logloss), "expected_calibration_error": float(np.sum(np.array(gaps)*weights)), "maximum_calibration_error": float(max(gaps, default=np.nan))}

