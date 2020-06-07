import time
from itertools import count
from typing import Callable
from typing import Sequence, Tuple

import gym
import numpy as np
import torch
from torch import Tensor

from two_blocks_env.collider_env import Observation, SettableGoalEnv, distance, dim_goal
from LSGAN import LSGAN

#### PARAMETERS ####
from utils import print_message

Rmin = 0.1
Rmax = 0.9

G_Input_Size  = 4       # noise dim, somehow noise size is defined as 4 in their implementation for ant_gan experiment
G_Hidden_Size = 256
D_Hidden_Size = 128
####################

Goals = Tensor
Returns = Sequence[float]


class Agent:
    def __call__(self, obs: Observation) -> np.ndarray:
        raise NotImplementedError

    def train(self, timesteps: int, callback: Callable = None) -> None:
        raise NotImplementedError


def random_agent(action_space: gym.spaces.Space) -> Agent:
    return lambda obs: action_space.sample()


def null_agent(action_space: gym.spaces.Space) -> Agent:
    return lambda obs: np.zeros(shape=action_space.shape)


def sample(t: Tensor, k: int) -> Tensor:
    """
    https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    Implemented according to Appendix A.1: 2/3 from gan generated goals, 1/3 from old goals
    TODO: To avoid concentration of goals, concatinate only the goals which are away from old_goals
    """
    num_samples = min(len(t), k)
    indices = torch.randperm(len(t))[:num_samples]
    return t[indices]


def initialize_GAN(env: gym.GoalEnv) -> LSGAN:
    goalGAN = LSGAN(generator_input_size=G_Input_Size,
                    generator_hidden_size=G_Hidden_Size,
                    generator_output_size=dim_goal(env),
                    discriminator_input_size=dim_goal(env),
                    discriminator_hidden_size=D_Hidden_Size,
                    discriminator_output_size=1) # distinguish whether g is in GOID or not
    return goalGAN


@print_message("Training the policy on current goals")
def update_and_eval_policy(goals: Goals, π: Agent, env: SettableGoalEnv) -> Tuple[Agent, Returns]:
    env.set_possible_goals(goals.numpy())
    env.reset()
    π.train(timesteps=env.max_episode_len*len(goals)*5)
    episode_successes_per_goal = env.get_successes_of_goals()
    assert all(len(sucs) > 0 for g, sucs in episode_successes_per_goal.items()),\
        "More steps are necessary to eval each goal at least once."
    returns = [np.mean(episode_successes_per_goal[tuple(g)]) for g in goals.numpy()]
    return π, returns


def label_goals(returns: Returns) -> Sequence[int]:
    return [int(Rmin <= r <= Rmax) for r in returns]

@print_message("Training GAN on current goals")
def train_GAN(goals: Goals, labels: Sequence[int], goalGAN):
    y: Tensor = torch.Tensor(labels).reshape(len(labels), 1)
    D = goalGAN.Discriminator.forward
    G = goalGAN.Generator.forward

    def D_loss_vec(z: Tensor) -> Tensor:
        return y*(D(goals)-1)**2 + (1-y)*(D(goals)+1)**2 +(D(G(z))+1)**2

    gradient_steps = 200
    for _ in range(gradient_steps):
        ### Train Discriminator ###
        zs = torch.randn(len(labels), goalGAN.Generator.noise_size)
        goalGAN.Discriminator.zero_grad()
        D_loss = torch.sum(D_loss_vec(zs))
        D_loss.backward()
        goalGAN.D_Optimizer.step()

        ### Train Generator ###
        zs = torch.randn(len(labels), goalGAN.Generator.noise_size)
        goalGAN.Generator.zero_grad()
        G_loss = torch.sum(D(G(zs))**2)
        G_loss.backward()
        goalGAN.G_Optimizer.step()

    return goalGAN


def trajectory(π: Agent, env: SettableGoalEnv, goal: np.ndarray = None,
               sleep_secs: float = 1/240):
    obs = env.reset()
    if goal is not None:
        env.set_possible_goals(goal[np.newaxis])

    for t in count():
        action = π(obs)
        next_obs, reward, done, info = env.step(action)
        time.sleep(sleep_secs)
        env.render()

        if t % 10 == 0:
            print(f"achieved goal: {obs.achieved_goal.T},"
                  f" desired goal: {obs.desired_goal.T},"
                  f" distance: {distance(obs.achieved_goal, obs.desired_goal)},")
        if info.get("is_success"):
            print("SUCCESS!")

        yield obs, action, reward, next_obs, done

        obs = next_obs
        if done:
            break
