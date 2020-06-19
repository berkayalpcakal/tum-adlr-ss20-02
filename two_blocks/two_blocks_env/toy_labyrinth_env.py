import time
from itertools import cycle
from typing import Sequence, List, Mapping, Optional
import gym
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typeguard import typechecked

from two_blocks_env.collider_env import Observation, SettableGoalEnv, Goal, GoalHashable
from utils import get_updateable_scatter

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

def _denormalize(norm_goals: Sequence[Sequence[float]]) -> np.ndarray:
    if not isinstance(norm_goals, np.ndarray):
        norm_goals = np.array(list(norm_goals))

    is_single_goal = norm_goals.size == 2
    if is_single_goal:
        norm_goals = norm_goals.reshape((1, 2))

    res = scaler.inverse_transform(norm_goals)
    if is_single_goal:
        res = res[0]

    return res


class ToyLab(SettableGoalEnv):

    observation_space = gym.spaces.Dict(spaces={
        "observation": gym.spaces.Box(low=0, high=0, shape=(0,)),
        "achieved_goal": gym.spaces.Box(low=-1, high=1, shape=(2,)),
        "desired_goal": gym.spaces.Box(low=-1, high=1, shape=(2,))
    })
    action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
    reward_range = (-1, 0)
    starting_obs = _normalize(_initial_pos)  # normalized because public

    def __init__(self, max_episode_len: int = 80, seed=0):
        super().__init__()
        self.max_episode_len = max_episode_len
        self._cur_pos = _initial_pos
        self._step_num = 0
        self._possible_normalized_goals = None
        self._normalized_goal = self._sample_new_goal()
        self.seed(seed)
        self._successes_per_goal: Mapping[GoalHashable, List[bool]] = dict()
        self._plot = None
        self._labyrinth_corners = labyrinth_corners

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
        if done and len(self._successes_per_goal) > 0:
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
    def set_possible_goals(self, goals: Optional[np.ndarray], entire_space=False) -> None:
        if goals is None and entire_space:
            self._possible_normalized_goals = None
            self._successes_per_goal = dict()
            return

        assert goals.shape[1] == self.observation_space["desired_goal"].shape[0]
        self._possible_normalized_goals = cycle(np.random.permutation(goals))
        self._successes_per_goal = {tuple(g): [] for g in goals}

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        return dict(self._successes_per_goal)

    def seed(self, seed=None):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def render(self, mode="human", other_positions: Mapping[str, np.ndarray] = None,
               show_cur_agent_and_goal_pos=True):
        if self._plot is None:
            self._plot = fig, ax, scatter_fn = get_updateable_scatter()
            ax.plot(*labyrinth_corners.T)
            fig.show()

        fig, ax, scatter_fn = self._plot

        if other_positions is not None:
            for color, positions in other_positions.items():
                scatter_fn(name=color, pts=None)  # clear previous
                if len(positions) > 0:
                    scatter_fn(name=color, pts=_denormalize(positions), c=color)

        agent_pos = None if not show_cur_agent_and_goal_pos else self._cur_pos
        goal = None if not show_cur_agent_and_goal_pos else _denormalize(self._normalized_goal)
        scatter_fn(name="agent_pos", pts=agent_pos)
        scatter_fn(name="goal", pts=goal, c="green")

        return fig, ax


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
    env = ToyLab(seed=1)
    env.reset()
    env.render()
    for _ in range(100):
        time.sleep(0.2)
        action = env.action_space.sample()
        env.step(action)
        env.render()
