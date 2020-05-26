import os
from collections import UserDict
from typing import Sequence
from abc import ABC

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


class SettableGoalEnv(ABC, gym.GoalEnv):
    def set_goal(self, goal: np.ndarray) -> None:
        raise NotImplementedError


class ColliderEnv(SettableGoalEnv):

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    _red_box_fname = os.path.join(__location__, 'red_block.urdf')
    _blue_box_fname = os.path.join(__location__, 'blue_block.urdf')

    def __init__(self, visualize):
        p.connect(p.GUI if visualize else p.DIRECT)
        p.setGravity(0, 0, -9.81)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        p.loadURDF("plane.urdf")
        self._box_height = 0.51
        self._red_box = p.loadURDF(self._red_box_fname, [0, 0, self._box_height])
        self._blue_box = p.loadURDF(self._blue_box_fname, [2, 0, self._box_height])
        self.action_space = spaces.Box(low=-10, high=10, shape=(3, 1), dtype=np.float)

        goal_space = spaces.Box(low=-10, high=10, shape=(2, 1), dtype=np.float)
        self.observation_space = spaces.Dict(spaces={
            "observation": spaces.Box(low=-100, high=100, shape=(14, 1), dtype=np.float),
            "desired_goal": goal_space,
            "achieved_goal": goal_space
        })

        self._desired_goal = np.ones(3) * 5

    def step(self, action: Sequence[float]):
        _apply_force(obj=self._red_box, force=action)
        p.stepSimulation()

        obs = self._get_obs()
        done = _goals_are_close(obs.achieved_goal, obs.desired_goal)
        reward = self.compute_reward(achieved_goal=obs.achieved_goal,
                                     desired_goal=obs.desired_goal, info={})

        return obs, reward, done, {}

    def _get_obs(self) -> Observation:
        """This is a private method! Do not use outside of env"""
        red = p.getBasePositionAndOrientation(self._red_box)  # (pos, quaternion)
        blue = p.getBasePositionAndOrientation(self._blue_box)

        state = np.concatenate((np.concatenate(red), np.concatenate(blue)))  # len: 14
        return Observation(observation=state, achieved_goal=_position(red),
                           desired_goal=self._desired_goal)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> float:
        return 0 if _goals_are_close(achieved_goal, desired_goal) else -1

    def reset(self):
        _reset_object(self._red_box,  pos=[0, 0, self._box_height], quaternion=[0, 0, 0, 1])
        _reset_object(self._blue_box, pos=[random.uniform(5, 10), random.uniform(-10, 10), self._box_height], quaternion=[0, 0, 0, 1])
        self.set_goal(_position(p.getBasePositionAndOrientation(self._blue_box)))
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def set_goal(self, new_goal: np.ndarray):
        assert isinstance(new_goal, np.ndarray)
        assert self.observation_space["desired_goal"].contains(new_goal)
        _reset_object(self._blue_box, pos=list(new_goal.T[0]) + [self._box_height], quaternion=[0, 0, 0, 1])
        self._desired_goal = new_goal


def distance(x1: np.ndarray, x2: np.ndarray):
    return np.linalg.norm(x1 - x2) ** 2


def _goals_are_close(achieved_goal: np.ndarray, desired_goal: np.ndarray):
    ε = 1
    return distance(achieved_goal, desired_goal) < ε


def _reset_object(obj, pos: Sequence[float], quaternion: Sequence[float]):
    p.resetBasePositionAndOrientation(obj, pos, quaternion)


def _position(position_and_orientation: Sequence[float]) -> np.ndarray:
    return np.array(position_and_orientation[0])[:2].reshape((2, 1))


def _apply_force(obj, force: Sequence[float]):
    at_obj_position = [0, 0, 0]
    at_obj_root = -1
    p.applyExternalForce(obj, at_obj_root, force, at_obj_position, flags=p.LINK_FRAME)
