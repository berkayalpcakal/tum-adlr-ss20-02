import time
from itertools import cycle
from typing import Sequence, List, Mapping

import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typeguard import typechecked

from two_blocks_env.collider_env import Observation, SettableGoalEnv, Goal, GoalHashable

middle_wall_len = 12
sidewall_height = 8
labyrinth_corners = np.array([
    (0, 0),
    (-middle_wall_len, 0),
    (-middle_wall_len, -sidewall_height/2),
    (sidewall_height/2, -sidewall_height/2),
    (sidewall_height/2, sidewall_height/2),
    (-middle_wall_len, sidewall_height/2),
    (-middle_wall_len, 0)
])

_initial_pos = np.array([-middle_wall_len+2, -2])
_labyrinth_lower_bound = np.array([-middle_wall_len, -sidewall_height / 2])
_labyrinth_upper_bound = np.array([sidewall_height / 2, sidewall_height / 2])

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit([_labyrinth_lower_bound, _labyrinth_upper_bound])


def _normalize(goal: Sequence[float]) -> np.ndarray:
    return scaler.transform(goal[np.newaxis])[0]

def _denormalize(norm_goal: Sequence[float]) -> np.ndarray:
    return scaler.inverse_transform(norm_goal[np.newaxis])[0]


class ToyLab(SettableGoalEnv):

    observation_space = gym.spaces.Dict(spaces={
        "observation": gym.spaces.Box(low=0, high=0, shape=(0,)),
        "achieved_goal": gym.spaces.Box(low=-1, high=1, shape=(2,)),
        "desired_goal": gym.spaces.Box(low=-1, high=1, shape=(2,))
    })
    action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
    reward_range = (-1, 0)

    def __init__(self, max_episode_len: int = 80):
        super().__init__()
        self.max_episode_len = max_episode_len
        self.starting_obs = _normalize(_initial_pos)  # normalized because public
        self._cur_pos = _initial_pos
        self._step_num = 0
        self._possible_normalized_goals = None
        self._normalized_goal = self._sample_new_goal()
        self._successes_per_goal: Mapping[GoalHashable, List[bool]] = dict()
        self._plot = None

    def _sample_new_goal(self) -> Goal:
        if self._possible_normalized_goals is None:
            return self.observation_space["desired_goal"].sample()
        return next(self._possible_normalized_goals)

    def step(self, action: np.ndarray):
        action = np.array(action)
        assert self.action_space.contains(action), action
        self._step_num += 1
        self._cur_pos = simulation_step(cur_pos=self._cur_pos, action=action)
        obs = self._make_obs()
        reward = self.compute_reward(obs.achieved_goal, obs.desired_goal, {})
        is_success = reward == max(self.reward_range)
        done = (is_success or self._step_num % self.max_episode_len == 0)
        if done:
            self._successes_per_goal[tuple(self._normalized_goal)].append(is_success)
        return obs, reward, done, {"is_success": float(is_success)}

    @classmethod
    def compute_reward(cls, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """
        This function can take inputs from outside, so the inputs are the normalized
        versions of the goals (to [-1, 1]).
        """
        is_success = _is_success(_denormalize(achieved_goal), _denormalize(desired_goal))
        return max(cls.reward_range) if is_success else min(cls.reward_range)

    def reset(self) -> Observation:
        super().reset()
        self._cur_pos = _initial_pos
        self._step_num = 0
        self._normalized_goal = self._sample_new_goal()
        return self._make_obs()

    def _make_obs(self) -> Observation:
        return Observation(observation=np.empty(0),
                           achieved_goal=_normalize(self._cur_pos),
                           desired_goal=self._normalized_goal)

    @typechecked
    def set_possible_goals(self, goals: np.ndarray) -> None:
        assert goals.shape[1] == self.observation_space["desired_goal"].shape[0]
        self._possible_normalized_goals = cycle(np.random.permutation(goals))
        self._successes_per_goal = {tuple(g): [] for g in goals}

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        return dict(self._successes_per_goal)

    def render(self, mode='human'):
        if self._plot is None:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(*labyrinth_corners.T)
            s1 = ax.scatter(*self._cur_pos, c="orange")
            s2 = ax.scatter(*self._normalized_goal, c="green")
            fig.show()
            self._plot = fig, s1, s2
        else:
            fig, s1, s2 = self._plot
            s1.set_offsets(self._cur_pos)
            s2.set_offsets(self._normalized_goal)
        fig.canvas.draw()
        fig.canvas.flush_events()


_step_len = 0.5
def simulation_step(cur_pos: np.ndarray, action: np.ndarray) -> np.ndarray:
    assert cur_pos.shape == action.shape
    x1, x2 = cur_pos + _step_len * action

    # no pass through 0
    if all(cur_pos <= 0):
        x2 = min(0, x2)
    if cur_pos[0] <= 0 and cur_pos[1] >= 0:
        x2 = max(0, x2)

    return np.clip(np.array([x1, x2]), a_min=_labyrinth_lower_bound, a_max=_labyrinth_upper_bound)


def _is_success(achieved_pos: np.ndarray, desired_pos: np.ndarray) -> bool:
    return (_are_on_same_side_of_wall(achieved_pos, desired_pos) and
            _are_close(achieved_pos, desired_pos))


def _are_on_same_side_of_wall(pos1: np.ndarray, pos2: np.ndarray) -> bool:
    return _is_above_wall(pos1) == _is_above_wall(pos2)


def _is_above_wall(pos: np.ndarray) -> bool:
    return pos[1] > 0


_max_single_action_dist = np.linalg.norm(ToyLab.action_space.high) * _step_len
def _are_close(x1: np.ndarray, x2: np.ndarray) -> bool:
    return np.linalg.norm(x1 - x2)**2 < _max_single_action_dist


if __name__ == '__main__':
    env = ToyLab()
    env.reset()
    env.render()
    for _ in range(100):
        time.sleep(0.2)
        action = env.action_space.sample()
        env.step(action)
        env.render()
