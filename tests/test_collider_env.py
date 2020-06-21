import numpy as np
import pytest

from multi_goal.GenerativeGoalLearning import trajectory, null_agent
from multi_goal.envs.collider_env import ColliderEnv
from multi_goal.envs.toy_labyrinth_env import ToyLab

NULL_ACTION = np.zeros(shape=ColliderEnv.action_space.shape)


def test_multiple_envs_can_be_instantiated():
    envs = [ColliderEnv(visualize=False) for _ in range(3)]
    assert envs is not None


def done(env_res):
    return env_res[2]


def test_max_episode_len():
    env = ColliderEnv(visualize=False, max_episode_len=7)
    dones = [done(env.step(NULL_ACTION)) for _ in range(6)]
    assert not dones[-1]
    assert done(env.step(NULL_ACTION))


def test_restart_reset_steps():
    env = ColliderEnv(visualize=False, max_episode_len=5)
    env.seed(0)
    while not done(env.step(NULL_ACTION)):
        pass

    env.reset()
    env.set_goal(np.array([1, 1]))
    assert not done(env.step(NULL_ACTION))


@pytest.mark.parametrize("env", [ColliderEnv(visualize=False, max_episode_len=10),
                                 ToyLab(max_episode_len=10, seed=1)])
def test_env_trajectory(env):
    agent = null_agent(action_space=env.action_space)
    assert len(list(trajectory(π=agent, env=env))) == 10

    goal = env.observation_space["desired_goal"].high
    assert len(list(trajectory(π=agent, env=env, goal=goal))) == 10

