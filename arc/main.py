from itertools import islice, count
from typing import Iterable

import matplotlib.pyplot as plt
import torch
from stable_baselines import HER
import numpy as np
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

from GenerativeGoalLearning import trajectory
from app_utils import Dirs
from action_reps_control import sac_agent, ActionableRep, ObservationSeq, Dact
from two_blocks_env.toy_labyrinth_env import ToyLab

from torch import nn
from torch import Tensor


def take(n: int, it: Iterable):
    return list(islice(it, n))

wall = np.array([[x1, 0] for x1 in np.linspace(-1, 0.5, 30)])

def init_viz():
    X, Y = np.mgrid[-1:1:0.1, -1:1:0.1]
    Xs = np.array([X.flatten(), Y.flatten()])
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(111)
    ax.set_autoscale_on(True)
    scatter = ax.scatter(*Xs)
    scatter2 = ax.scatter(*wall.T, c="red")
    starting_pos = ToyLab.starting_obs
    sactter3 = ax.scatter(*starting_pos, c="orange")
    t = count()

    def viz(phi: ActionableRep):
        out = phi(Tensor(Xs.T)).detach().numpy()
        scatter.set_offsets(out)
        wall_out = phi(Tensor(wall)).detach().numpy()
        scatter2.set_offsets(wall_out)
        sactter3.set_offsets(phi(Tensor(starting_pos)).detach().numpy())
        ax.ignore_existing_data_limits = True
        ax.update_datalim(scatter.get_datalim(ax.transData))
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        #fig.savefig(f"./figs/arc-fig-{next(t)}.png")
    return viz


if __name__ == '__main__':
    plt.ion()
    env = ToyLab()
    model_fpath = Dirs("her-sac-toylab").best_model
    model = HER.load(load_path=model_fpath, env=ToyLab)
    agent, gaussian_pi = sac_agent(model, ToyLab)

    traj_lens = [len(list(trajectory(agent, env))) for _ in range(100)]
    bins = np.bincount(traj_lens)
    fig, ax = plt.subplots()
    ax.bar(range(len(bins)), bins)
    fig.show()
    env = ToyLab(max_episode_len=int(2*np.mean(traj_lens)))

    steps_gen = (step for _ in count() for step in trajectory(agent, env))
    obs_gen = (step[0] for step in steps_gen)

    phi = ActionableRep(input_size=env.observation_space["achieved_goal"].shape[0])
    optimizer = Adam(phi.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    num_epochs = 300
    approx_num_diff_goals = 2
    D: ObservationSeq = take(approx_num_diff_goals*int(np.mean(traj_lens)), obs_gen)

    goals = np.array(list(set(tuple(o.desired_goal) for o in D)))
    states = np.array([o.achieved_goal for o in D])
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(*goals.T); axs[0].scatter(*wall.T, c="red")
    axs[1].scatter(*states.T); axs[1].scatter(*wall.T, c="red")
    fig.show()
    viz = init_viz()
    viz(phi)

    for epoch in range(num_epochs):
        losses = []
        for batch in DataLoader(D, batch_size=2, shuffle=True):
            s1, s2 = batch["achieved_goal"].float()

            optimizer.zero_grad()
            loss = loss_fn(torch.dist(phi(s1), phi(s2)), Dact(s1, s2, D, pi=gaussian_pi))
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        print(f"Epoch finihsed: {epoch}, loss: {np.mean(losses):.2f}", flush=True)
        viz(phi)

    torch.save(phi.state_dict(), "arc.pt")

    print("")