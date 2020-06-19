import random
from typing import Sequence, Callable, List

import numpy as np
import torch
from stable_baselines.her import HERGoalEnvWrapper
from torch import Tensor

import torch.nn as nn
from torch.distributions import Distribution, MultivariateNormal as MVNormal
from torch.distributions.kl import kl_divergence as KL

from agents import HERSACAgent
from two_blocks_env.collider_env import Observation, SettableGoalEnv

ObservationSeq = Sequence[Observation]
GaussianPolicy = Callable[[ObservationSeq], List[Distribution]]


def get_gaussian_pi(agent: HERSACAgent, env: SettableGoalEnv) -> GaussianPolicy:
    flattener = HERGoalEnvWrapper(env)

    def gaussian_pi(many_obs: ObservationSeq) -> List[Distribution]:
        many_flat_obs = np.array([flattener.convert_dict_to_obs(o) for o in many_obs])
        mus, sigma_diags = agent._model.policy_tf.proba_step(many_flat_obs)
        t = Tensor
        return [MVNormal(t(mu), torch.diag(t(sig))) for mu, sig in zip(mus, sigma_diags)]

    return gaussian_pi


def copy_and_replace_goal(obss: ObservationSeq, goal: np.ndarray) -> ObservationSeq:
    new_obss = [Observation(**o) for o in obss]
    for obs in new_obss:
        obs["desired_goal"] = goal
    return new_obss


def Dact(g1: np.ndarray, g2: np.ndarray, D_: ObservationSeq, pi: GaussianPolicy) -> Tensor:
    obs_samples = random.sample(D_, k=7)
    N1s = pi(copy_and_replace_goal(obss=obs_samples, goal=g1))
    N2s = pi(copy_and_replace_goal(obss=obs_samples, goal=g2))
    kls = [KL(N1, N2) + KL(N2, N1) for N1, N2 in zip(N1s, N2s)]
    return torch.mean(Tensor(kls))


class ActionableRep(nn.Module):
    def __init__(self, input_size: int = 2):
        super().__init__()
        hidden_layer_size = 128
        acr_size = 2

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, acr_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
