import numpy as np
from src.regime.state_alignment import align_probabilities, align_state_profiles

def test_label_alignment_recovers_permutation():
    ref=np.array([[1,1],[5,2],[-2,4.]]); candidate=ref[[2,0,1]]
    mapping=align_state_profiles(ref,candidate); p=np.eye(3)
    assert np.allclose(align_probabilities(p,mapping).argmax(1), [2,0,1])

