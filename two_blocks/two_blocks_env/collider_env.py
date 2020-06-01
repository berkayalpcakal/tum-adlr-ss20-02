import os
import time
from abc import ABC
from typing import Sequence

import numpy as np
import pybullet
import pybullet_data
import gym

from gym import spaces
from pybullet_utils.bullet_client import BulletClient


class Observation(dict):
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

    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float)
    _action_tol = 0.01
    goal_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float)
    observation_space = spaces.Dict(spaces={
        "observation": spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float),
        "desired_goal": goal_space,
        "achieved_goal": goal_space
    })
    __filelocation__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    _red_ball_fname = os.path.join(__filelocation__, 'assets/little_ball.urdf')
    _green_ball_fname = os.path.join(__filelocation__, 'assets/immaterial_ball.urdf')

    _ball_radius = 0.3
    _green_ball_initial_pos = [2, 0, _ball_radius]
    _ball_initial_pos = [0, 0, _ball_radius]
    _viz_lock_taken = False

    def __init__(self, visualize: bool = True, max_episode_len: int = 200):
        self._visualize = visualize
        if visualize:
            assert not self._viz_lock_taken, "only one environment can be visualized simultaneously"
            ColliderEnv._viz_lock_taken = True

        self._bullet = BulletClient(connection_mode=pybullet.GUI if visualize else pybullet.DIRECT)
        self._bullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        self._bullet.setGravity(0, 0, -9.81)
        self._bullet.loadURDF("plane.urdf")
        self._ball = self._bullet.loadURDF(self._red_ball_fname, self._ball_initial_pos)
        self._target_ball = self._bullet.loadURDF(self._green_ball_fname, self._green_ball_initial_pos, useFixedBase=1)

        self._desired_goal = np.ones(2) * 5
        self._max_episode_len = max_episode_len
        self._step_num = 0

    def step(self, action: np.ndarray):
        assert self._action_is_valid(action), action
        self._step_num += 1
        assert self._step_num <= self._max_episode_len

        for _ in range(10):
            if self._visualize:
                time.sleep(1 / 240)
            _apply_force(self._bullet, obj=self._ball, force=action)
            self._bullet.stepSimulation()

        obs = self._get_obs()
        is_success = _goals_are_close(obs.achieved_goal, obs.desired_goal)
        done = (is_success or self._step_num % self._max_episode_len == 0)
        reward = self.compute_reward(achieved_goal=obs.achieved_goal,
                                     desired_goal=obs.desired_goal, info={})

        return obs, reward, done, {"is_success": float(is_success)}

    def _action_is_valid(self, action) -> bool:
        return (all(self.action_space.low - self._action_tol <= action) and
                all(action <= self.action_space.high + self._action_tol))

    def _get_obs(self) -> Observation:
        """This is a private method! Do not use outside of env"""
        ball_pos = _position(self._bullet.getBasePositionAndOrientation(self._ball))
        blue_pos = _position(self._bullet.getBasePositionAndOrientation(self._target_ball))

        state = np.concatenate((ball_pos, blue_pos))  # len=4
        return Observation(observation=state, achieved_goal=ball_pos, desired_goal=self._desired_goal)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> float:
        return 0 if _goals_are_close(achieved_goal, desired_goal) else -1

    def reset(self):
        self._step_num = 0
        _reset_object(self._bullet, self._ball, pos=self._ball_initial_pos)
        random_goal = self.observation_space["desired_goal"].sample()
        _reset_object(self._bullet, self._target_ball, pos=[*random_goal, self._ball_radius])
        self.set_goal(_position(self._bullet.getBasePositionAndOrientation(self._target_ball)))
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def set_goal(self, new_goal: np.ndarray):
        assert isinstance(new_goal, np.ndarray)
        assert self.observation_space["desired_goal"].contains(new_goal)
        _reset_object(self._bullet, self._target_ball, pos=[*new_goal, self._ball_radius])
        self._desired_goal = new_goal


def distance(x1: np.ndarray, x2: np.ndarray):
    return np.linalg.norm(x1 - x2) ** 2


def _goals_are_close(achieved_goal: np.ndarray, desired_goal: np.ndarray):
    ε = 0.3
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
    env = ColliderEnv()
    while True:
        env.step([0, 0])
        time.sleep(1/240)
