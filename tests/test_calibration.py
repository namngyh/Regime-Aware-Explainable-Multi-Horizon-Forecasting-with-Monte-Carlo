import numpy as np
from src.calibration import MulticlassCalibrator, calibration_metrics

def test_calibrated_probabilities_sum_to_one():
    rng=np.random.default_rng(1); p=rng.dirichlet(np.ones(4),300); y=rng.integers(0,4,300)
    out=MulticlassCalibrator("sigmoid").fit(p[:200],y[:200]).predict(p[200:])
    assert np.allclose(out.sum(1),1); assert np.isfinite(calibration_metrics(out,y[200:])["log_loss"])

