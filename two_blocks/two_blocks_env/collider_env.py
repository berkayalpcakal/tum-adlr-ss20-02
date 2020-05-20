from collections import UserDict
from typing import Sequence

import numpy as np
import pybullet as p
import pybullet_data
import gym
import random

from gym import spaces


class Observation(UserDict):
    def __init__(self, observation: np.ndarray, achieved_goal: np.ndarray,
                 desired_goal: np.ndarray) -> None:
        super().__init__(observation=observation, achieved_goal=achieved_goal,
                         desired_goal=desired_goal)
        self.observation = observation
        self.achieved_goal = achieved_goal
        self.desired_goal = desired_goal


class ColliderEnv(gym.GoalEnv):

    def __init__(self, visualize):
        p.connect(p.GUI if visualize else p.DIRECT)
        p.setGravity(0, 0, -9.81)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        p.loadURDF("plane.urdf")
        self._red_box = p.loadURDF('./two_blocks_env/red_block.urdf', [0, 0, 0.51])
        self._blue_box = p.loadURDF('./two_blocks_env/blue_block.urdf', [2, 0, 0.51])
        self.action_space = spaces.Box(low=-10, high=10, shape=(3, 1), dtype=np.float)

        pos_space = spaces.Box(low=-100, high=100, shape=(3, 1), dtype=np.float)
        self.observation_space = spaces.Dict(spaces={
            "observation": spaces.Box(low=-100, high=100, shape=(14, 1), dtype=np.float),
            "desired_goal": pos_space,
            "achieved_goal": pos_space
        })

        self._desired_blue_pos = np.ones(3)*5

    def step(self, action: Sequence[float]):
        _apply_force(obj=self._red_box, force=action)
        p.stepSimulation()

        obs = self._get_obs()
        done = _goals_are_close(obs.achieved_goal, obs.desired_goal)
        reward = self.compute_reward(achieved_goal=obs.achieved_goal,
                                     desired_goal=obs.desired_goal, info={})

        return obs, reward, done, {}

    def _get_obs(self) -> Observation:
        red = p.getBasePositionAndOrientation(self._red_box)  # (pos, quaternion)
        blue = p.getBasePositionAndOrientation(self._blue_box)
        blue_pos = np.array(blue[0]).reshape((3,1))

        state = np.concatenate((np.concatenate(red), np.concatenate(blue)))  # len: 14
        return Observation(observation=state, achieved_goal=blue_pos,
                           desired_goal=self._desired_blue_pos)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> float:
        return 0 if _goals_are_close(achieved_goal, desired_goal) else -1

    def reset(self):
        _reset_object(self._red_box, pos=[0, 0, 0.51], quaternion=[0, 0, 0, 1])
        _reset_object(self._blue_box, pos=[2 + random.uniform(0, 1), 0, 0.51], quaternion=[0, 0, 0, 1])
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def set_goal(self, new_goal: np.ndarray):
        assert isinstance(new_goal, np.ndarray)
        assert self.observation_space["desired_goal"].contains(new_goal)
        self._desired_blue_pos = new_goal


def distance(x1: np.ndarray, x2: np.ndarray):
    return np.linalg.norm(x1 - x2) ** 2


def _goals_are_close(achieved_goal: np.ndarray, desired_goal: np.ndarray):
    ε = 0.1
    return distance(achieved_goal, desired_goal) < ε


def _reset_object(obj, pos: Sequence[float], quaternion: Sequence[float]):
    p.resetBasePositionAndOrientation(obj, pos, quaternion)


def _apply_force(obj, force: Sequence[float]):
    at_obj_position = [0, 0, 0]
    at_obj_root = -1
    p.applyExternalForce(obj, at_obj_root, force, at_obj_position, flags=p.LINK_FRAME)
