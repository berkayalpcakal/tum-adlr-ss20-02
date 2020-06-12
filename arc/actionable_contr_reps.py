import random
from itertools import islice, count, tee
from typing import Iterable, Sequence, Callable, Tuple

import matplotlib.pyplot as plt
import gym
import torch
from gym.wrappers import FlattenObservation
from stable_baselines import HER, SAC
import numpy as np
from stable_baselines.common import BaseRLModel
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

from GenerativeGoalLearning import trajectory, Agent
from app_utils import Dirs, pairwise
from two_blocks_env.collider_env import Observation
from two_blocks_env.toy_labyrinth_env import ToyLab

from torch import nn
from torch import Tensor
from torch.distributions import MultivariateNormal, Distribution
from torch.distributions.kl import kl_divergence as KL
import torch.nn.functional as F

def sac_agent(model: HER, env: gym.Env) -> Tuple[Agent, Callable]:
    model.env.reward_range = env.reward_range
    flattener = FlattenObservation(ToyLab)

    def agent(o: Observation) -> np.ndarray:
        flat_obs = flattener.observation(o)
        action, _ = model.predict(flat_obs)
        return action

    def gaussians_at(many_obs: Sequence[Observation]) -> Sequence[Distribution]:
        many_flat_obs = np.array([flattener.observation(o) for o in many_obs])
        mus, sigma_diags = model.model.policy_tf.proba_step(many_flat_obs)
        t = Tensor
        return [MultivariateNormal(t(mu), torch.diag(t(sig))) for mu, sig in zip(mus, sigma_diags)]

    return agent, gaussians_at


def take(n: int, it: Iterable):
    return list(islice(it, n))


class ActionableRep(nn.Module):
    def __init__(self, input_size: int = 4):
        super().__init__()
        hidden_layer_size = 16
        acr_size = 2

        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l3 = nn.Linear(hidden_layer_size, acr_size)

    def forward(self, x: Tensor) -> Tensor:
        res = F.relu(self.l1(x))
        res = F.relu(self.l2(res))
        return self.l3(res)


Dataset = Sequence[Observation]

def copy_and_replace_goal(obss: Dataset, goal: np.ndarray):
    new_obss = [Observation(**o) for o in obss]
    for obs in new_obss:
        obs["desired_goal"] = goal
    return new_obss

def Dact(g1: np.ndarray, g2: np.ndarray, D_: Dataset) -> Tensor:
    num_s_samples = 5
    obss_s1 = copy_and_replace_goal(obss=random.sample(D_, k=num_s_samples), goal=g1)
    N1s = gaussians_at(obss_s1)

    obss_s2 = copy_and_replace_goal(obss=random.sample(D_, k=num_s_samples), goal=g2)
    N2s = gaussians_at(obss_s2)

    kls = [KL(N1, N2) + KL(N2, N1) for N1, N2 in zip(N1s, N2s)]
    return torch.mean(Tensor(kls))


if __name__ == '__main__':
    env = ToyLab()
    model_fpath = Dirs("her-sac-toylab").best_model
    model = HER.load(load_path=model_fpath, env=env)
    agent, gaussians_at = sac_agent(model, env)

    traj_lens = [len(list(trajectory(agent, env))) for _ in range(100)]
    bins = np.bincount(traj_lens)
    plt.bar(range(len(bins)), bins)
    plt.show()
    env = ToyLab(max_episode_len=int(2*np.mean(traj_lens)))

    steps_gen = (step for _ in count() for step in trajectory(agent, env))
    obs_gen = (step[0] for step in steps_gen)

    phi = ActionableRep(input_size=env.observation_space["achieved_goal"].shape[0])
    optimizer = Adam(phi.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    num_epochs = 10
    approx_num_diff_goals = 100
    D: Dataset = take(approx_num_diff_goals*int(np.mean(traj_lens)), obs_gen)

    goals = np.array(list(set(tuple(o.desired_goal) for o in D)))
    states = np.array([o.achieved_goal for o in D])
    plt.scatter(*goals.T)
    plt.show()
    plt.scatter(*states.T)
    plt.show()

    for epoch in range(num_epochs):
        losses = []
        for batch in DataLoader(D, batch_size=2, shuffle=True):
            s1, s2 = batch["achieved_goal"].float()

            optimizer.zero_grad()
            loss = loss_fn(torch.dist(phi(s1), phi(s2)), Dact(s1, s2, D))
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        print(f"Epoch finihsed: {epoch}, loss: {np.mean(losses):.2f}", flush=True)

    torch.save(phi.state_dict(), "arc.pt")

    X, Y = np.mgrid[-1:1:0.1, -1:1:0.1]
    Xs = np.array([X.flatten(), Y.flatten()])
    out = phi(Tensor(Xs.T)).detach().numpy()
    plt.scatter(*out.T)
    plt.show()

    print("")