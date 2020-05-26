import random
import time
from typing import Sequence, Callable, Tuple

import gym
import numpy as np

from two_blocks_env.collider_env import Observation, SettableGoalEnv, distance

#### PARAMETERS ####
num_old_goals = 100
Rmin = 0.1
Rmax = 0.9
max_episode_length = 1000
####################


class Agent:
    def act(self, obs: Observation) -> np.ndarray:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, action_space: gym.spaces.Space):
        self._action_space = action_space

    def act(self, obs: Observation) -> np.ndarray:
        return self._action_space.sample()


def sample(goals_old: Sequence[np.ndarray]):
    return random.sample(goals_old, k=min(num_old_goals, len(goals_old)))


Network = Callable
Goals = Sequence[np.ndarray]
Returns = Sequence[float]


def initialize_GAN(obs_space: gym.spaces.Dict) -> Tuple[Network, Network]:
    """Placeholders for now"""

    def G(noise_vector: np.ndarray):
        return [obs_space['desired_goal'].sample() for _ in noise_vector]

    def D():
        pass

    return G, D


def update_policy(goals: Goals, π: Agent, env: SettableGoalEnv) -> Agent:
    """Placeholder update. Exemplary use of trajectory() below"""
    for g in goals:
        τ = trajectory(π=π, env=env, goal=g)
        for (s, a, r, s2, done) in τ:
            pass
    return π


def evaluate_policy(goals: Goals, π: Agent, env: SettableGoalEnv) -> Returns:
    """Placeholder evaluation"""
    return [random.random() for _ in goals]


def label_goals(returns: Returns) -> Sequence[int]:
    return [int(Rmin <= r <= Rmax) for r in returns]


def train_GAN(goals: Goals, labels: Sequence[int], G: Network, D: Network) -> Tuple[Network, Network]:
    """Placeholder training"""
    return G, D


def update_replay(goals: Goals) -> Goals:
    """Placeholder update"""
    return goals


def trajectory(π: Agent, env: SettableGoalEnv, goal: np.ndarray):
    obs = env.reset()
    env.set_goal(goal)
    for t in range(max_episode_length):
        action = π.act(obs)
        next_obs, reward, done, _ = env.step(action)
        time.sleep(1/24)

        if t % 100 == 0:
            print(f"achieved goal: {obs.achieved_goal.T},"
                  f" desired goal: {obs.desired_goal.T},"
                  f" distance: {distance(obs.achieved_goal, obs.desired_goal)},")
        if done:
            print("SUCCESS!")
        if t == (max_episode_length - 1):
            print("REACHED MAXIMUM EPISODE LEN")
            done = True

        yield obs, action, reward, next_obs, done

        obs = next_obs
        if done:
            break
