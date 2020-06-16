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
D_Hidden_Size = 256
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

def eval_policy(goals: Goals, π: Agent, env: SettableGoalEnv):
    for g in goals:
        for obs, action, reward, next_obs, done, info in trajectory(π, env, goal=g):
            if info.get("is_success"):
                break
    import pdb; pdb.set_trace()
    episode_successes_per_goal = env.get_successes_of_goals()
    returns = [np.mean(episode_successes_per_goal[tuple(g)]) for g in goals.numpy()]
    return returns

def label_goals(returns: Returns) -> Sequence[int]:
    return [int(Rmin <= r <= Rmax) for r in returns]

@print_message("Training GAN on current goals")
def train_GAN(goals: Goals, labels: Sequence[int], goalGAN):
    y: Tensor = torch.Tensor(labels).reshape(len(labels), 1)
    D = goalGAN.Discriminator.forward
    G = goalGAN.Generator.forward

    def D_loss_vec(z: Tensor) -> Tensor:
        return y*(D(goals)-1)**2 + (1-y)*(D(goals)+1)**2 +(D(G(z))+1)**2

    iterations = 10
    for _ in range(iterations):
        ### Train Discriminator ###
        gradient_steps = 1
        for _ in range(gradient_steps):
            zs = torch.randn(len(labels), goalGAN.Generator.noise_size)
            goalGAN.Discriminator.zero_grad()
            D_loss = torch.mean(D_loss_vec(zs))
            D_loss.backward()
            goalGAN.D_Optimizer.step()

        ### Train Generator ###
        gradient_steps = 1
        β = goalGAN.Generator.variance_coeff
        for _ in range(gradient_steps):
            zs = torch.randn(len(labels), goalGAN.Generator.noise_size)
            goalGAN.Generator.zero_grad()
            G_loss = torch.mean(D(G(zs))**2) + β/torch.var(G(zs), dim=0).mean()
            G_loss.backward()
            goalGAN.G_Optimizer.step()

    return goalGAN

@print_message("Updating the regularized replay buffer")
def update_replay(goals: Tensor, goals_old: Tensor):
    eps = 0.1
    for g in goals:
        g_is_close_to_goals_old = min((torch.dist(g, g_old) for g_old in goals_old)) < eps
        if not g_is_close_to_goals_old:
            goals_old = torch.cat((g[None], goals_old))
    return goals_old

def trajectory(π: Agent, env: SettableGoalEnv, goal: np.ndarray = None,
               sleep_secs: float = 0, render=False, print_every: int = None):
    if goal is not None:
        env.set_possible_goals(np.array(goal)[np.newaxis])
    obs = env.reset()

    for t in count():
        action = π(obs)
        next_obs, reward, done, info = env.step(action)

        if sleep_secs > 0:
            time.sleep(sleep_secs)
        if render:
            env.render()

        if print_every is not None and t % print_every == 0:
            print(f"achieved goal: {obs.achieved_goal.T},"
                  f" desired goal: {obs.desired_goal.T},"
                  f" distance: {distance(obs.achieved_goal, obs.desired_goal)},")
        if print_every is not None and info.get("is_success"):
            print("SUCCESS!")

        yield obs, action, reward, next_obs, done, info

        obs = next_obs
        if done:
            break
