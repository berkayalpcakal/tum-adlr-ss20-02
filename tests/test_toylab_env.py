from two_blocks_env.toy_labyrinth_env import _initial_pos, _normalize, \
    _labyrinth_upper_bound, _labyrinth_lower_bound, _are_on_same_side_of_wall
import numpy as np


def test_normalization():
    assert _initial_pos.shape == _normalize(_initial_pos).shape
    assert np.allclose(-np.ones(2), _normalize(_labyrinth_lower_bound))
    assert np.allclose(np.ones(2), _normalize(_labyrinth_upper_bound))
    middle = (_labyrinth_lower_bound + _labyrinth_upper_bound)/2
    assert np.allclose(np.zeros(2), _normalize(middle))


def test_are_on_same_side_of_wall():
    below_wall = np.array([1, -1])
    above_wall = np.array([1, 1])
    assert _are_on_same_side_of_wall(below_wall, below_wall)
    assert _are_on_same_side_of_wall(above_wall, above_wall)
    assert not _are_on_same_side_of_wall(below_wall, above_wall)
