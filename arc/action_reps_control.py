import os
import random
from typing import Sequence, Callable, List

import numpy as np
import torch
from stable_baselines.her import HERGoalEnvWrapper
from torch import Tensor

import torch.nn as nn
from torch.distributions import Distribution, MultivariateNormal as MVNormal
from torch.distributions.kl import kl_divergence as KL

from multi_goal.GenerativeGoalLearning import Agent, trajectory
from multi_goal.agents import HERSACAgent
from multi_goal.envs import Observation, ISettableGoalEnv
from multi_goal.envs.toy_labyrinth_env import ToyLab

ObservationSeq = Sequence[Observation]
GaussianPolicy = Callable[[ObservationSeq], List[Distribution]]


def get_gaussian_pi(agent: HERSACAgent, env: ISettableGoalEnv) -> GaussianPolicy:
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
        acr_size = 3

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


class ARCAgent(Agent):
    _fpath = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    _arc_fpath = os.path.join(_fpath, "arc.pt")

    def __init__(self, phi: ActionableRep = None):
        if phi is None:
            self._phi = ActionableRep(input_size=2)
            self._phi.load_state_dict(torch.load(self._arc_fpath))
        else:
            self._phi = phi
        self._x = torch.zeros(2, requires_grad=True)
        self._opt = self._make_opt()

    def _make_opt(self):
        return torch.optim.Adam([self._x], lr=0.8)

    def __call__(self, obs: Observation) -> np.ndarray:
        self._opt.zero_grad()
        cur_x = torch.from_numpy(obs.achieved_goal).float()
        self._x.data = cur_x.clone()
        goal = torch.from_numpy(obs.desired_goal).float()
        loss = torch.dist(self._phi(goal), self._phi(self._x))
        loss.backward()
        self._opt.step()
        direction = self._x - cur_x
        norming = max(1, torch.norm(direction))
        return (direction/norming).detach().numpy()

    def train(self, timesteps: int) -> None:
        raise NotImplementedError("The ARC agent is already trained.")

    def reset_momentum(self):
        if self._x.grad is not None:
            self._x.grad.zero_()
        self._opt = self._make_opt()


if __name__ == '__main__':
    agent = ARCAgent()
    env = ToyLab(max_episode_len=140)
    # evaluate(agent, env, very_granular=False)
    # input("exit")

    goals = np.mgrid[-1:0:5j, 0:1:5j].reshape((2, -1)).T
    env.set_possible_goals(goals)
    while True:
        traj_len = sum(1 for _ in trajectory(agent, env, sleep_secs=0.05, render=True))
        print(f"Trajectory length: {traj_len}")
        agent.reset_momentum()
