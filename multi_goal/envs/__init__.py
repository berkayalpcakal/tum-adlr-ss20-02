from abc import ABC
from collections import OrderedDict
from typing import Tuple, Optional, Mapping, List, Sequence

import gym
import numpy as np
from sklearn.preprocessing import MinMaxScaler

GoalHashable = Tuple[float]


class Observation(OrderedDict):
    def __init__(self, observation: np.ndarray, achieved_goal: np.ndarray,
                 desired_goal: np.ndarray) -> None:
        super().__init__(observation=observation, achieved_goal=achieved_goal,
                         desired_goal=desired_goal)
        self.observation = observation
        self.achieved_goal = achieved_goal
        self.desired_goal = desired_goal

    def __eq__(self, other):
        return all(np.allclose(other[k], v) for k, v in self.items())

    def __ne__(self, other):
        return not self.__eq__(other)


class SettableGoalEnv(ABC, gym.GoalEnv):
    max_episode_len: int
    starting_obs: np.ndarray

    def set_possible_goals(self, goals: Optional[np.ndarray], entire_space=False) -> None:
        raise NotImplementedError

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        raise NotImplementedError


def normalizer(low, high):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit([low, high])

    def normalize(goal: Sequence[float]) -> np.ndarray:
        return scaler.transform(goal[np.newaxis])[0]

    def denormalize(norm_goals: Sequence[Sequence[float]]) -> np.ndarray:
        if not isinstance(norm_goals, np.ndarray):
            norm_goals = np.array(list(norm_goals))

        is_single_goal = norm_goals.size == 2
        if is_single_goal:
            norm_goals = norm_goals.reshape((1, 2))

        res = scaler.inverse_transform(norm_goals)
        if is_single_goal:
            res = res[0]

        return res

    return normalize, denormalize
