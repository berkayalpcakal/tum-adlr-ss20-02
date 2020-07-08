import os
from datetime import datetime
from itertools import islice, count
from typing import Iterable

import matplotlib.pyplot as plt
import torch
import numpy as np

from multi_goal.GenerativeGoalLearning import trajectory
from multi_goal.agents import HERSACAgent
from arc.action_reps_control import get_gaussian_pi, ActionableRep, ObservationSeq, train_arc

from torch import Tensor

from multi_goal.envs.toy_labyrinth_env import ToyLab


def take(n: int, it: Iterable):
    return list(islice(it, n))


wall = np.array([[x1, 0] for x1 in np.linspace(-1, 0.5, 30)])
starting_pos = np.array([-.75, -.5])


def init_viz():
    X, Y = np.mgrid[-1:1:0.1, -1:1:0.1]
    Xs = np.array([X.flatten(), Y.flatten()])
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(111)
    ax.set_autoscale_on(True)
    scatter = ax.scatter(*Xs)
    wall_line, = ax.plot(*wall.T, c="red")
    scatter2 = ax.scatter(*starting_pos, c="orange")
    t = count()
    viztime = {"last": datetime.now()}

    def viz(phi: ActionableRep):
        return
        if (datetime.now() - viztime["last"]).seconds < 2:
            return
        out = phi(Tensor(Xs.T)).detach().numpy()
        scatter.set_offsets(out)
        wall_out = phi(Tensor(wall)).detach().numpy()
        wall_line.set_data(wall_out.T)
        scatter2.set_offsets(phi(Tensor(starting_pos)).detach().numpy())
        ax.ignore_existing_data_limits = True
        ax.update_datalim(scatter.get_datalim(ax.transData))
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        viztime["last"] = datetime.now()
        #fig.savefig(f"./figs/arc-fig-{next(t)}.png")
    return viz


if __name__ == '__main__':
    # plt.ion()
    env = ToyLab(use_random_starting_pos=True)
    agent = HERSACAgent(env=env, rank=1)
    gaussian_pi = get_gaussian_pi(agent, env)

    traj_lens = [len(list(trajectory(agent, env))) for _ in range(100)]
    # bins = np.bincount(traj_lens)
    # fig, ax = plt.subplots()
    # ax.bar(range(len(bins)), bins)

    env = ToyLab(max_episode_len=int(2 * np.mean(traj_lens)), use_random_starting_pos=True)
    traj_gen = ([step[3] for step in trajectory(agent, env)] for _ in count())
    phi = ActionableRep(input_size=env.observation_space["achieved_goal"].shape[0])
    fname = "arc.pt"
    if os.path.isfile(fname):
        phi.load_state_dict(torch.load(fname))
        print("loaded previous state")

    num_trajectories = 50
    trajectories = take(num_trajectories, traj_gen)
    D: ObservationSeq = [obs for t in trajectories for obs in t]

    # goals = np.array([t[0].desired_goal for t in trajectories])
    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # axs[0].set_ylim((-1, 1)); axs[0].set_xlim((-1, 1))
    # axs[0].scatter(*goals.T)
    # for t in trajectories:
    #     states = np.array([o.achieved_goal for o in t])
    #     axs[1].plot(*states.T)
    # for ax in axs:
    #     ax.plot(*wall.T, c="red")
    #     ax.scatter(*starting_pos, c="orange")

    train_arc(D=D, gaussian_pi=gaussian_pi, phi=phi)
