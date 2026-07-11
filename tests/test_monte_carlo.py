import numpy as np
from src.simulation.monte_carlo import structural_monte_carlo, summarize_scenarios
def test_monte_carlo_is_reproducible_and_finite():
    args=(100.,np.array([.6,.4]),np.array([[.9,.1],[.2,.8]]),np.array([.0005,-.0003]),np.array([.01,.02]))
    a=structural_monte_carlo(*args,paths=200,seed=3); b=structural_monte_carlo(*args,paths=200,seed=3)
    assert np.allclose(a["prices"],b["prices"]); assert np.isfinite(summarize_scenarios(a,100.)).all().all()

