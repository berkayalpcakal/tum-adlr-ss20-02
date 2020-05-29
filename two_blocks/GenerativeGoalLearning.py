import random
import time
from typing import Sequence, Tuple

import gym
import numpy as np
import torch
from torch import nn, Tensor

from two_blocks_env.collider_env import Observation, SettableGoalEnv, distance, dim_goal
from LSGAN import LSGAN

#### PARAMETERS ####
Rmin = 0.1
Rmax = 0.9
max_episode_length = 500

G_Input_Size  = 4       # noise dim, somehow noise size is defined as 4 in their implementation for ant_gan experiment
G_Hidden_Size = 256
D_Hidden_Size = 128
####################


class Agent:
    def act(self, obs: Observation) -> np.ndarray:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, action_space: gym.spaces.Space):
        self._action_space = action_space

    def act(self, obs: Observation) -> np.ndarray:
        return self._action_space.sample()


def sample_from_list(list_to_sample: Sequence[np.ndarray], out_length):
    return random.sample(list_to_sample, k=out_length)


def sample(t: Tensor, k: int) -> Tensor:
    """
    https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    Implemented according to Appendix A.1: 2/3 from gan generated goals, 1/3 from old goals
    TODO: To avoid concentration of goals, concatinate only the goals which are away from old_goals
    """
    num_samples = min(len(t), k)
    indices = torch.randperm(len(t))[:num_samples]
    return t[indices]


Goals = Tensor
Returns = Sequence[float]


def initialize_GAN(env: gym.GoalEnv) -> Tuple[nn.Module, nn.Module]:
    goalGAN = LSGAN(generator_input_size=G_Input_Size,
                    generator_hidden_size=G_Hidden_Size,
                    generator_output_size=dim_goal(env),
                    discriminator_input_size=dim_goal(env),
                    discriminator_hidden_size=D_Hidden_Size,
                    discriminator_output_size=1) # distinguish whether g is in GOID or not
    return goalGAN.Generator, goalGAN.Discriminator


def update_policy(goals: Goals, π: Agent, env: SettableGoalEnv) -> Agent:
    """Placeholder update. Exemplary use of trajectory() below"""
    for g in goals.numpy():
        τ = trajectory(π=π, env=env, goal=g)
        for (s, a, r, s2, done) in τ:
            pass
    return π


def evaluate_policy(goals: Goals, π: Agent, env: SettableGoalEnv) -> Returns:
    """Placeholder evaluation"""
    return [random.random() for _ in goals]


def label_goals(returns: Returns) -> Sequence[int]:
    return [int(Rmin <= r <= Rmax) for r in returns]


def train_GAN(goals: Goals, labels: Sequence[int], G: nn.Module, D: nn.Module) -> Tuple[nn.Module, nn.Module]:
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
        time.sleep(1/240)

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
