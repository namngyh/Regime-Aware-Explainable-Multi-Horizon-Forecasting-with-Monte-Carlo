import numpy as np
from src.simulation.block_bootstrap import moving_block_indices
def test_bootstrap_uses_contiguous_blocks():
    idx=moving_block_indices(100,20,np.random.default_rng(1)); assert len(idx)==100
    assert np.mean(np.diff(idx)!=1)<.06

