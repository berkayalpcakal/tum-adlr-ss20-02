from two_blocks_env.toy_labyrinth_env import _initial_pos, _normalize, \
    _labyrinth_upper_bound, _labyrinth_lower_bound, _are_on_same_side_of_wall, ToyLab, \
    _denormalize
import numpy as np

def test_compute_reward():
    for _ in range(5):
        g = ToyLab.observation_space["desired_goal"].sample()
        assert ToyLab.compute_reward(g, g, None) == max(ToyLab.reward_range)


def test_normalization():
    assert _initial_pos.shape == _normalize(_initial_pos).shape
    assert np.allclose(-np.ones(2), _normalize(_labyrinth_lower_bound))
    assert np.allclose(np.ones(2), _normalize(_labyrinth_upper_bound))
    middle = (_labyrinth_lower_bound + _labyrinth_upper_bound)/2
    assert np.allclose(np.zeros(2), _normalize(middle))

    for _ in range(10):
        pos = ToyLab.observation_space["desired_goal"].sample()
        assert np.allclose(_normalize(_denormalize(pos)), pos)


def test_are_on_same_side_of_wall():
    below_wall = np.array([1, -1])
    above_wall = np.array([1, 1])
    assert _are_on_same_side_of_wall(below_wall, below_wall)
    assert _are_on_same_side_of_wall(above_wall, above_wall)
    assert not _are_on_same_side_of_wall(below_wall, above_wall)


def test_setting_goals_at_runtime():
    my_goals = [tuple(ToyLab.observation_space["desired_goal"].sample()) for _ in range(3)]
    env = ToyLab()
    for _ in range(3):
        assert tuple(env.reset().desired_goal) not in my_goals

    env.set_possible_goals(np.array(my_goals))
    for _ in range(3):
        assert tuple(env.reset().desired_goal) in my_goals


def test_get_goal_successes():
    env = ToyLab()
    assert all(len(successes) == 0 for successes in env.get_successes_of_goals().values())
    my_goals = np.array([env.starting_obs, env.observation_space["desired_goal"].high])
    env.set_possible_goals(my_goals)

    null_action = np.zeros(shape=env.action_space.shape)
    for _ in range(2):
        env.reset()
        for _ in range(3):
            env.step(null_action)

    successes = env.get_successes_of_goals()
    assert successes[tuple(my_goals[0])][0]
    assert len(successes[tuple(my_goals[1])]) == 0


def test_moving_one_step_away_from_goal_still_success():
    env = ToyLab()
    env.set_possible_goals(env.starting_obs[np.newaxis])
    obs = env.reset()
    obs, r, done, info = env.step(env.action_space.high)
    assert info["is_success"] == 1
    assert env.compute_reward(obs.achieved_goal, obs.desired_goal, None) == env.reward_range[1]

def test_seed_determines_trajectories():
    assert ToyLab(seed=0).reset() == ToyLab(seed=0).reset()
    assert ToyLab(seed=0).reset() != ToyLab(seed=1).reset()

    env = ToyLab(seed=0)
    env.reset()

    mk_actions = lambda: [env.action_space.sample() for _ in range(10)]
    mk_obs = lambda: [env.reset() for _ in range(10)]

    actions = mk_actions()
    obss = mk_obs()
    trajectory = [env.step(a) for a in actions]

    env.seed(0)
    env.reset()
    assert np.allclose(actions, mk_actions())
    assert obss == mk_obs()
    assert trajectory == [env.step(a) for a in actions]
