import random
from typing import Sequence, Tuple, Callable, List

import gym
import numpy as np
import torch
from gym.wrappers import FlattenObservation
from stable_baselines import HER
from torch import Tensor

import torch.nn as nn
from torch.distributions import Distribution, MultivariateNormal as MVNormal
from torch.distributions.kl import kl_divergence as KL
import torch.nn.functional as F

from GenerativeGoalLearning import Agent
from two_blocks_env.collider_env import Observation

ObservationSeq = Sequence[Observation]
GaussianPolicy = Callable[[ObservationSeq], List[Distribution]]


def sac_agent(model: HER, env: type(gym.Env)) -> Tuple[Agent, GaussianPolicy]:
    model.env.reward_range = env.reward_range
    flattener = FlattenObservation(env)

    def agent(o: Observation) -> np.ndarray:
        flat_obs = flattener.observation(o)
        action, _ = model.predict(flat_obs)
        return action

    def gaussian_pi(many_obs: ObservationSeq) -> List[Distribution]:
        many_flat_obs = np.array([flattener.observation(o) for o in many_obs])
        mus, sigma_diags = model.model.policy_tf.proba_step(many_flat_obs)
        t = Tensor
        return [MVNormal(t(mu), torch.diag(t(sig))) for mu, sig in zip(mus, sigma_diags)]

    return agent, gaussian_pi

def copy_and_replace_goal(obss: ObservationSeq, goal: np.ndarray) -> ObservationSeq:
    new_obss = [Observation(**o) for o in obss]
    for obs in new_obss:
        obs["desired_goal"] = goal
    return new_obss

def Dact(g1: np.ndarray, g2: np.ndarray, D_: ObservationSeq, pi: GaussianPolicy) -> Tensor:
    num_s_samples = len(D_)
    obss_s1 = copy_and_replace_goal(obss=random.sample(D_, k=num_s_samples), goal=g1)
    N1s = pi(obss_s1)

    obss_s2 = copy_and_replace_goal(obss=random.sample(D_, k=num_s_samples), goal=g2)
    N2s = pi(obss_s2)

    kls = [KL(N1, N2) + KL(N2, N1) for N1, N2 in zip(N1s, N2s)]
    return torch.mean(Tensor(kls))


class ActionableRep(nn.Module):
    def __init__(self, input_size: int = 4):
        super().__init__()
        hidden_layer_size = 128
        acr_size = 2

        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l3 = nn.Linear(hidden_layer_size, acr_size)

    def forward(self, x: Tensor) -> Tensor:
        res = F.relu(self.l1(x))
        res = F.relu(self.l2(res))
        return self.l3(res)
