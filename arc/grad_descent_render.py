import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import torch

from arc.action_reps_control import ActionableRep, ARCAgent
from multi_goal.GenerativeGoalLearning import trajectory
from multi_goal.envs.toy_labyrinth_env import ToyLab

goal = np.array([-0.7, 0.7])
phi = ActionableRep(2)
phi.load_state_dict(torch.load("arc.pt"))
phi.eval()
#phi = lambda x: x

def phi_dist(pos1, pos2):
    return torch.dist(*phi(torch.Tensor([pos1, pos2])).detach())


def gradient_descent_rendering():
    fig, ax = plt.subplots()

    wall = np.array([[-1, 0], [0.5, 0]])
    ax.plot(*wall.T, c="red")

    ax.scatter(*goal, c="orange")

    space2d = np.mgrid[-1:1:30j, -1:1:30j]
    isolines = 20
    dists = np.array([phi_dist(goal, pt).numpy() for pt in space2d.reshape((2, -1)).T])
    ax.contour(*space2d, dists.reshape((30, 30)), isolines, cmap=viridis)

    fig.tight_layout()

    return fig, ax


fig, ax = gradient_descent_rendering()
env = ToyLab(max_episode_len=60, use_random_starting_pos=True, seed=1)
agent = ARCAgent(phi=phi, env=env)
while True:
    agent.reset_momentum()
    t = trajectory(pi=agent, env=env, goal=goal, print_every=100)
    xs = np.array([step[3].achieved_goal for step in t])

    lines, = ax.plot([], [], c="blue", alpha=0.8)
    h = ax.scatter([], [], c="blue")

    for idx, head in enumerate(xs):
        curve = xs[:idx+1]
        lines.set_data(*curve.T)
        h.set_offsets(head)
        plt.pause(0.05)
