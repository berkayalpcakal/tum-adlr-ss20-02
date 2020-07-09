import functools
import os
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple
import numpy as np
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
from scipy import stats as st

from multi_goal.envs import ISettableGoalEnv


def print_message(msg: str):
    def decorator(func):
        @functools.wraps(func)
        def printer(*args, **kwargs):
            start = time.time()
            print(msg, end=" ", flush=True)
            res = func(*args, **kwargs)
            duration = time.time() - start
            print(f"DONE. TIME: {duration:.2f} [s]")
            return res
        return printer
    return decorator


def get_updateable_scatter():
    plt.ion()
    fig, ax = plt.subplots()
    scatters: Dict[str, PathCollection] = dict()

    def scatter(name: str, pts: np.ndarray, *args, **kwargs):
        if pts is None or len(pts) == 0:
            if name in scatters:
                scatters.pop(name).remove()

        else:
            if pts.size == 2:
                pts = pts[np.newaxis]
            assert pts.shape[1] == 2, "Inputs pts must have shape (N, 2)"

            if name in scatters:
                scatters[name].set_offsets(pts)
            else:
                scatters[name] = ax.scatter(*pts.T, *args, **kwargs)

    return fig, ax, scatter


def get_updateable_contour(xlim=(-12, 4), ylim=(-4, 4)):
    xmin, xmax = xlim
    ymin, ymax = ylim

    X, Y = S = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = S.reshape((2, -1)).T
    last_contours = []

    def contour_fn(unnormed_data, ax):
        if len(last_contours) > 0:
            for c in last_contours.pop().collections:
                c.remove()

        kernel = st.gaussian_kde(unnormed_data.T)
        z = np.reshape(kernel(positions.T), X.shape)

        contours = ax.contourf(X, Y, z, cmap='Blues', zorder=-1)
        last_contours.append(contours)
        #cset = ax.contour(X, Y, z, colors='k')
        #ax.clabel(cset, inline=1, fontsize=10)

    return contour_fn


def display_goals(goals: np.ndarray, returns, idx, env: ISettableGoalEnv, fileNamePrefix ='goals', gan_goals=None):
    rewards = np.array(returns)
    low_reward_idx  = np.argwhere(0.1>rewards).reshape(-1,)
    high_reward_idx = np.argwhere(0.9<rewards).reshape(-1,)
    goid_reward_idx = np.argwhere(np.array([int(0.1 <= r <= 0.9) for r in returns])==1).reshape(-1,)

    low_reward_goals  = goals[low_reward_idx]
    high_reward_goals = goals[high_reward_idx]
    goid_reward_goals = goals[goid_reward_idx]

    colors = {"red": low_reward_goals,
              "green": high_reward_goals,
              "blue": goid_reward_goals,
              "orange": env.starting_agent_pos}
    fig, ax = env.render(other_positions=colors,
                         show_agent_and_goal_pos=False,
                         positions_density=gan_goals)

    fig.savefig("./figs/{}_{}.png".format(fileNamePrefix, idx))


class Dirs:
    def __init__(self, experiment_name: str, rank=0):
        self.prefix = experiment_name
        this_fpath = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.root = Path(this_fpath)/"../all-results"/experiment_name
        self.models = str(self.root/f"ckpts-{rank}")
        self.tensorboard = str(self.root/"tensorboard")

    @property
    def best_model(self):
        return str(Path(self.models)/latest_model(self.models))


def latest_model(foldername: str):
    model_names = os.listdir(foldername)
    assert len(model_names) > 0, f"The folder {foldername} exists, but contains no model files."
    if len(model_names) == 1:
        return model_names[0]
    prefix_less, prefix = remove_common_prefix(model_names)
    nums_only, suffix = remove_common_suffix(prefix_less)
    assert "" not in nums_only, "Can't find best model, because step nums are interpreted as name prefix."
    latest_num = max(int(n) for n in nums_only)
    return f"{prefix}{latest_num}{suffix}"


def remove_common_prefix(strs: Sequence[str]) -> Tuple[Sequence[str], str]:
    prefix = os.path.commonprefix(strs)
    return [s.replace(prefix, "") for s in strs], prefix


def remove_common_suffix(strs: Sequence[str]) -> Tuple[Sequence[str], str]:
    rev_strs = [reverse(s) for s in strs]
    prefix_less, prefix = remove_common_prefix(rev_strs)
    return [reverse(s) for s in prefix_less], reverse(prefix)


def reverse(s: str) -> str:
    return "".join(reversed(s))