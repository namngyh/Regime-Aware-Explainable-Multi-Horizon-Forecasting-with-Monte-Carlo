import numpy as np
from src.simulation.block_bootstrap import moving_block_indices
def test_seed_reproduces_bootstrap_indices():
    assert np.array_equal(moving_block_indices(200,40,np.random.default_rng(42)),moving_block_indices(200,40,np.random.default_rng(42)))

