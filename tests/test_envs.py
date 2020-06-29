from itertools import combinations

import gym
import pytest

from multi_goal.GenerativeGoalLearning import trajectory, null_agent
from multi_goal.envs import Observation
from multi_goal.envs.pybullet_labyrinth_env import Labyrinth
from multi_goal.envs.toy_labyrinth_env import normalizer, ToyLabSimulator, ToyLab
import numpy as np

env_fns = [Labyrinth, ToyLab]


@pytest.mark.parametrize("env_fn", env_fns)
def test_compute_reward(env_fn):
    env = env_fn()
    for _ in range(5):
        g = env.observation_space["desired_goal"].sample()
        assert env.compute_reward(g, g, None) == max(env.reward_range)


@pytest.mark.parametrize("env_fn", env_fns)
def test_env_normalization(env_fn):
    env = env_fn()
    space = env.observation_space["desired_goal"]
    assert env.reward_range == (-1, 0)
    assert np.allclose(np.ones(space.shape), space.high)
    assert np.allclose(-np.ones(space.shape), space.low)

    env.reset()
    low = env.observation_space["achieved_goal"].low
    high = env.observation_space["achieved_goal"].high
    for _ in range(100):
        obs: Observation = env.step(env.action_space.high)[0]
        assert all(low <= obs.achieved_goal) and all(obs.achieved_goal <= high)


def test_normalizer():
    rand = np.random.randn(2, 10)
    low, high = rand.max(axis=1), rand.min(axis=1)
    norm, denorm = normalizer(low, high)
    for _ in range(10):
        goal = np.random.uniform(low, high)
        assert np.allclose(norm(denorm(goal)), goal)


def test_are_on_same_side_of_wall():
    below_wall = np.array([1, -1])
    above_wall = np.array([1, 1])
    c = ToyLabSimulator
    assert c._are_on_same_side_of_wall(below_wall, below_wall)
    assert c._are_on_same_side_of_wall(above_wall, above_wall)
    assert not c._are_on_same_side_of_wall(below_wall, above_wall)


@pytest.mark.parametrize("env_fn", env_fns)
def test_setting_goals_at_runtime(env_fn):
    env = env_fn()
    my_goals = [tuple(env.observation_space["desired_goal"].sample()) for _ in range(3)]
    for _ in range(3):
        assert tuple(env.reset().desired_goal) not in my_goals

    env.set_possible_goals(np.array(my_goals))
    for _ in range(3):
        obs = env.reset()
        assert any(np.allclose(obs.desired_goal, g) for g in my_goals), f"{obs.desired_goal} not in {my_goals}"

    env.set_possible_goals(None, entire_space=True)
    for _ in range(3):
        assert tuple(env.reset().desired_goal) not in my_goals


@pytest.mark.parametrize("use_random_starting_pos", [True, False])
def test_get_goal_successes(use_random_starting_pos: bool):
    env = ToyLab(use_random_starting_pos=use_random_starting_pos)
    assert all(len(successes) == 0 for successes in env.get_successes_of_goals().values())
    difficult_goal = env.observation_space["desired_goal"].high
    my_goals = np.array([env.starting_agent_pos, difficult_goal])
    env.set_possible_goals(my_goals)

    null_action = np.zeros(shape=env.action_space.shape)
    for _ in range(2):
        env.reset()
        for _ in range(3):
            env.step(null_action)

    successes = env.get_successes_of_goals()

    if not use_random_starting_pos:
        assert successes[tuple(env.starting_agent_pos)][0]
    assert len(successes[tuple(difficult_goal)]) == 0


@pytest.mark.parametrize("env_fn", env_fns)
def test_moving_one_step_away_from_goal_still_success(env_fn):
    env = env_fn()
    env.set_possible_goals(env.starting_agent_pos[np.newaxis])
    env.reset()
    obs, r, done, info = env.step(env.action_space.high)
    assert np.allclose(obs.desired_goal, env.starting_agent_pos)
    assert info["is_success"] == 1
    assert env.compute_reward(obs.achieved_goal, obs.desired_goal, None) == env.reward_range[1]


@pytest.mark.parametrize("env_fn", env_fns)
def test_seed_determines_trajectories(env_fn):
    null = np.zeros(shape=env_fn().action_space.shape)
    assert env_fn(seed=0).step(null)[0] == env_fn(seed=0).step(null)[0]
    assert env_fn(seed=0).step(null)[0] != env_fn(seed=1).step(null)[0]

    env = env_fn(seed=0)
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


@pytest.mark.parametrize("env_fn", env_fns)
def test_with_random_starting_states(env_fn):
    env = env_fn(use_random_starting_pos=True)
    o1: Observation
    starting_obss = [env.reset() for _ in range(5)]
    for o1, o2 in combinations(starting_obss, 2):
        assert not np.allclose(o1.achieved_goal, o2.achieved_goal)

    for obs in starting_obss:
        assert not np.allclose(obs.achieved_goal, obs.desired_goal)


@pytest.mark.parametrize("env_name", ["ToyLab-v0", "Labyrinth-v0", "HardLabyrinth-v0"])
def test_gym_registration_succeded(env_name):
    assert gym.make(env_name) is not None, "The gym could not be loaded with gym.make." \
                                           "Check the env registration string."


@pytest.mark.parametrize("env_fn", env_fns)
def test_multiple_envs_can_be_instantiated(env_fn):
    envs = [env_fn() for _ in range(3)]
    assert envs is not None


def done(env_res):
    return env_res[2]


@pytest.mark.parametrize("env_fn", env_fns)
def test_max_episode_len(env_fn):
    env = env_fn(max_episode_len=7)
    null_action = np.zeros(shape=env.action_space.shape)
    dones = [done(env.step(null_action)) for _ in range(6)]
    assert not dones[-1]
    assert done(env.step(null_action))


@pytest.mark.parametrize("env_fn", env_fns)
def test_restart_reset_steps(env_fn):
    env = env_fn(max_episode_len=5)
    null_action = np.zeros(shape=env.action_space.shape)
    env.seed(0)
    while not done(env.step(null_action)):
        pass

    env.reset()
    env.set_possible_goals(np.array([[1, 1]]))
    assert not done(env.step(null_action))


@pytest.mark.parametrize("env_fn", env_fns)
def test_env_trajectory(env_fn):
    env = env_fn(max_episode_len=10)
    agent = null_agent(action_space=env.action_space)
    assert len(list(trajectory(pi=agent, env=env))) == 10

    goal = env.observation_space["desired_goal"].high
    assert len(list(trajectory(pi=agent, env=env, goal=goal))) == 10


@pytest.mark.parametrize("env_fn", env_fns)
def test_random_goals_cover_space(env_fn):
    env = env_fn()
    null_step = np.zeros(shape=env.action_space.shape)
    instantiation_goals = np.array([env_fn(seed=i).step(null_step)[0].desired_goal for i in range(100)])
    assert cover_space(instantiation_goals)

    reset_goals = np.array([env.reset().desired_goal for _ in range(100)])
    assert cover_space(reset_goals)


def cover_space(samples: np.ndarray, tolerance=0.03) -> bool:
    return (np.allclose(samples.min(axis=0), -1, atol=tolerance) and
            np.allclose(samples.max(axis=0), 1, atol=tolerance))


@pytest.mark.parametrize("env_fn,obs_size", [(Labyrinth, 2), (ToyLab, 0)])
def test_obs_size_as_expected(env_fn, obs_size):
    env = env_fn()
    assert env.observation_space["observation"].shape[0] == obs_size
    assert env.reset().observation.size == obs_size
    null_action = np.zeros(shape=env.action_space.shape)
    assert env.step(null_action)[0].observation.size == obs_size
