import numpy as np

from raemf_mc.uncertainty.block_bootstrap import moving_block_indices


def test_bootstrap_preserves_length():
    idx = moving_block_indices(100, 10, np.random.default_rng(1))
    assert len(idx) == 100
    assert idx.min() >= 0 and idx.max() < 100
