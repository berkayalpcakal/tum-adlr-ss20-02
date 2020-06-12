import os
from itertools import tee
from pathlib import Path
from typing import Sequence, Tuple, Callable
import gym
import numpy as np
from stable_baselines.common import BaseRLModel
from two_blocks_env.collider_env import Observation


def latest_model(foldername: str):
    model_names = os.listdir(foldername)
    assert len(model_names) > 0, model_names
    if len(model_names) == 1:
        return model_names[0]
    prefix_less, prefix = remove_common_prefix(model_names)
    nums_only, suffix = remove_common_suffix(prefix_less)
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


def vf_for_model(model: BaseRLModel, currentObs: Observation):

    def func(x, y):
        new_obs = Observation(**currentObs)
        new_obs["desired_goal"] = np.array([x, y])
        flat_obs = gym.spaces.flatten(model.env.env.observation_space, new_obs)
        return model.predict(flat_obs)[0]

    return vector_field(func, space=model.env.env.observation_space["desired_goal"])


def vector_field(func: Callable[[float, float], np.ndarray], space: gym.spaces.Box):
    assert space.shape == (2,)
    low_x, low_y = space.low
    high_x, high_y = space.high
    X, Y = np.mgrid[low_x:high_x:(high_x-low_x)/20, low_y:high_y:(high_y-low_y)/20]
    U, V = np.array([func(*k) for c in zip(X, Y) for k in zip(*c)]).T
    return X, Y, U.reshape(X.shape), V.reshape(X.shape)


class Dirs:
    def __init__(self, experiment_name: str):
        self.prefix = experiment_name
        this_fpath = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        results = Path(this_fpath)/"../all-results"/experiment_name
        self.models = str(results/"ckpts")
        self.tensorboard = str(results/"tensorboard")

    @property
    def best_model(self):
        return str(Path(self.models)/latest_model(self.models))


# For comparison
def perfect_action(obs) -> np.ndarray:
    ball_pos = obs[4:6]
    target_pos = obs[6:8]
    return (target_pos - ball_pos) / np.linalg.norm(target_pos - ball_pos)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd

def has_len_1(e):
    return len(e.summary.value) == 1

def to_dict(e):
    return {"time": e.wall_time, "step": e.step, "episode_reward": e.summary.value[0].simple_value}

def dump_tb_rewards_to_csv(tb_fpath: str, csv_fpath: str) -> None:
    gen = summary_iterator(tb_fpath)
    rews = filter(has_len_1, gen)
    dicts = map(to_dict, rews)
    df = pd.DataFrame.from_dict(dicts)
    df.to_csv(csv_fpath, index=False)
