import time
from typing import Sequence

import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from two_blocks_env.collider_env import Observation, SettableGoalEnv

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
_labyrinth_space = gym.spaces.Box(low=_labyrinth_lower_bound, high=_labyrinth_upper_bound)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit([_labyrinth_lower_bound, _labyrinth_upper_bound])


def _normalize(goal: Sequence[float]) -> np.ndarray:
    return scaler.transform(goal[np.newaxis])[0]


class ToyLab(SettableGoalEnv):

    observation_space = gym.spaces.Dict(spaces={
        "observation": gym.spaces.Box(low=0, high=0, shape=(0,)),
        "achieved_goal": _labyrinth_space,
        "desired_goal": _labyrinth_space
    })
    action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    def __init__(self, max_episode_len: int = 40):
        super().__init__()
        self._max_episode_len = max_episode_len
        self._cur_pos = _initial_pos
        self._step_num = 0
        self._goal = self.observation_space["desired_goal"].sample()

        self._plot = None

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), action
        self._step_num += 1
        self._cur_pos = sim_step(cur_pos=self._cur_pos, action=action)
        obs = self._make_obs()
        is_success = _is_success(obs.achieved_goal, obs.desired_goal)
        done = (is_success or self._step_num % self._max_episode_len == 0)
        reward = self.compute_reward(obs.achieved_goal, obs.desired_goal, {})
        return obs, reward, done, {"is_success": float(is_success)}

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        return 0 if _is_success(achieved_goal, desired_goal) else -1

    def reset(self) -> Observation:
        super().reset()
        self._cur_pos = _initial_pos
        self._step_num = 0
        self.set_goal(goal=self.observation_space["desired_goal"].sample())
        return self._make_obs()

    def _make_obs(self) -> Observation:
        return Observation(observation=np.empty(0),
                           achieved_goal=_normalize(self._cur_pos),
                           desired_goal=_normalize(self._goal))

    def set_goal(self, goal: np.ndarray) -> None:
        assert isinstance(goal, np.ndarray)
        assert self.observation_space["desired_goal"].contains(goal)
        self._goal = goal

    def render(self, mode='human'):
        if self._plot is None:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(*labyrinth_corners.T)
            s1 = ax.scatter(*self._cur_pos, c="orange")
            s2 = ax.scatter(*self._goal, c="green")
            fig.show()
            self._plot = fig, s1, s2
        else:
            fig, s1, s2 = self._plot
            s1.set_offsets(self._cur_pos)
            s2.set_offsets(self._goal)
        fig.canvas.draw()
        fig.canvas.flush_events()


def sim_step(cur_pos: np.ndarray, action: np.ndarray) -> np.ndarray:
    assert cur_pos.shape == action.shape
    x1, x2 = cur_pos + action

    # no pass through 0
    if all(cur_pos <= 0):
        x2 = min(0, x2)
    if cur_pos[0] <= 0 and cur_pos[1] >= 0:
        x2 = max(0, x2)

    return np.clip(np.array([x1, x2]), a_min=_labyrinth_lower_bound, a_max=_labyrinth_upper_bound)


def _is_success(g1: np.ndarray, g2: np.ndarray) -> bool:
    return _are_on_same_side_of_wall(g1, g2) and _are_close(g1, g2)


def _are_on_same_side_of_wall(pos1: np.ndarray, pos2: np.ndarray) -> bool:
    return _is_above_wall(pos1) == _is_above_wall(pos2)


def _is_above_wall(pos: np.ndarray) -> bool:
    return pos[1] > 0


def _are_close(x1: np.ndarray, x2: np.ndarray) -> bool:
    return np.linalg.norm(x1 - x2)**2 < 0.1


if __name__ == '__main__':
    env = ToyLab()
    env.reset()
    env.render()
    for _ in range(100):
        time.sleep(0.2)
        action = env.action_space.sample()
        env.step(action)
        env.render()
