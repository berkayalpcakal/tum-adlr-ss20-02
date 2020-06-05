from two_blocks_env.toy_labyrinth_env import _initial_pos, _normalize, \
    _labyrinth_upper_bound, _labyrinth_lower_bound
import numpy as np


def test_normalization():
    assert _initial_pos.shape == _normalize(_initial_pos).shape
    assert np.allclose(-np.ones(2), _normalize(_labyrinth_lower_bound))
    assert np.allclose(np.ones(2), _normalize(_labyrinth_upper_bound))
    middle = (_labyrinth_lower_bound + _labyrinth_upper_bound)/2
    assert np.allclose(np.zeros(2), _normalize(middle))
