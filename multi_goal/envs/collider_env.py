import os
import time
from abc import ABC
from itertools import cycle
from typing import Sequence, Tuple, Mapping, List, Optional
from collections import OrderedDict

import numpy as np
import pybullet
import pybullet_data
import gym

from gym import spaces
from pybullet_utils.bullet_client import BulletClient

Goal = np.ndarray
GoalHashable = Tuple[float]


class Observation(OrderedDict):
    def __init__(self, observation: np.ndarray, achieved_goal: Goal,
                 desired_goal: Goal) -> None:
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
    starting_agent_pos: np.ndarray

    def set_possible_goals(self, goals: Optional[np.ndarray], entire_space=False) -> None:
        raise NotImplementedError

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        raise NotImplementedError


class ColliderEnv(SettableGoalEnv):
    reward_range = (-1, 0)
    _action_tol = 0.01
    __filelocation__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    _red_ball_fname = os.path.join(__filelocation__, 'assets/little_ball.urdf')
    _green_ball_fname = os.path.join(__filelocation__, 'assets/immaterial_ball.urdf')

    _ball_radius = 0.3
    _green_ball_initial_pos = [2, 0, _ball_radius]
    _red_ball_initial_pos = [0, 0, _ball_radius]
    _viz_lock_taken = False
    starting_agent_pos = np.array(_red_ball_initial_pos[:2])

    def __init__(self, visualize: bool = False, max_episode_len: int = 200, seed=0):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float)
        goal_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float)
        self.observation_space = spaces.Dict(spaces={
            "observation": spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float),
            "desired_goal": goal_space,
            "achieved_goal": goal_space
        })
        self.seed(seed)
        self._visualize = visualize
        if visualize:
            assert not self._viz_lock_taken, "only one environment can be visualized simultaneously"
            ColliderEnv._viz_lock_taken = True

        self._bullet = BulletClient(connection_mode=pybullet.GUI if visualize else pybullet.DIRECT)
        self._bullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        self._bullet.setGravity(0, 0, -9.81)
        self._bullet.loadURDF("plane.urdf")
        self._agent_ball = self._bullet.loadURDF(self._red_ball_fname, self._red_ball_initial_pos)

        self._possible_goals = None
        self._goal = self._sample_new_goal()
        self._goal_ball = self._bullet.loadURDF(self._green_ball_fname, [*self._goal, self._ball_radius], useFixedBase=1)
        self.max_episode_len = max_episode_len
        self._step_num = 0
        self._successes_per_goal: Mapping[GoalHashable, List[bool]] = dict()

    def step(self, action: np.ndarray):
        assert self._action_is_valid(action), action
        self._step_num += 1
        assert self._step_num <= self.max_episode_len

        for _ in range(10):
            if self._visualize:
                time.sleep(1 / 240)
            _apply_force(self._bullet, obj=self._agent_ball, force=action)
            self._bullet.stepSimulation()

        obs = self._get_obs()
        reward = self.compute_reward(achieved_goal=obs.achieved_goal,
                                     desired_goal=obs.desired_goal, info={})
        is_success = reward == max(self.reward_range)
        done = (is_success or self._step_num % self.max_episode_len == 0)
        return obs, reward, done, {"is_success": float(is_success)}

    def _action_is_valid(self, action) -> bool:
        return (all(self.action_space.low - self._action_tol <= action) and
                all(action <= self.action_space.high + self._action_tol))

    def _get_obs(self) -> Observation:
        """This is a private method! Do not use outside of env"""
        agent_pos = _position(self._bullet.getBasePositionAndOrientation(self._agent_ball))
        goal_pos = _position(self._bullet.getBasePositionAndOrientation(self._goal_ball))

        return Observation(observation=np.empty(0), achieved_goal=agent_pos, desired_goal=goal_pos)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> float:
        return 0 if _goals_are_close(achieved_goal, desired_goal) else -1

    def reset(self):
        self._step_num = 0
        _reset_object(self._bullet, self._agent_ball, pos=self._red_ball_initial_pos)
        self._goal = self._sample_new_goal()
        _reset_object(self._bullet, self._goal_ball, pos=[*self._goal, self._ball_radius])
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def set_possible_goals(self, goals: Optional[np.ndarray], entire_space=False) -> None:
        if goals is None and entire_space:
            self._possible_goals = None
            self._successes_per_goal = dict()
            return

        assert len(goals.shape) == 2, f"Goals must have shape (N, 2), instead: {goals.shape}"
        assert goals.shape[1] == self.observation_space["desired_goal"].shape[0]
        self._possible_goals = cycle(np.random.permutation(goals))
        self._successes_per_goal = {tuple(g): [] for g in goals}

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        np.random.seed(seed)

    def _sample_new_goal(self) -> np.ndarray:
        if self._possible_goals is None:
            return self.observation_space["desired_goal"].sample()
        return next(self._possible_goals)


def distance(x1: np.ndarray, x2: np.ndarray):
    return np.linalg.norm(x1 - x2) ** 2


def _goals_are_close(achieved_goal: np.ndarray, desired_goal: np.ndarray):
    ε = ColliderEnv._ball_radius
    return distance(achieved_goal, desired_goal) < ε


def _position(position_and_orientation: Sequence[float]) -> np.ndarray:
    return np.array(position_and_orientation[0])[:2]  # [pos, quaternion]


def _reset_object(bc: BulletClient, obj, pos: Sequence[float]):
    quaternion = [0, 0, 0, 1]
    bc.resetBasePositionAndOrientation(obj, pos, quaternion)


_zforce = 0
_force_multiplier = 4  # tuned value
def _apply_force(bc: BulletClient, obj, force: Sequence[float]):
    force = _force_multiplier * np.array([*force, _zforce])
    obj_pos, _  = bc.getBasePositionAndOrientation(obj)
    bc.applyExternalForce(objectUniqueId=obj, linkIndex=-1,
                          forceObj=force, posObj=obj_pos, flags=pybullet.WORLD_FRAME)


def dim_goal(env: gym.GoalEnv):
    return env.observation_space["desired_goal"].shape[0]


if __name__ == '__main__':
    env = ColliderEnv(visualize=True)
    while True:
        env.step([0, 0])
        time.sleep(1/240)
