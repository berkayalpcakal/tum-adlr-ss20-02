import os
import time
from itertools import count
from typing import Sequence

import numpy as np
import pybullet
import pybullet_data

from pybullet_utils.bullet_client import BulletClient

from multi_goal.envs import Simulator, SettableGoalEnv, normalizer


class Labyrinth(SettableGoalEnv):
    def __init__(self, visualize=False, max_episode_len=100, *args, **kwargs):
        simulator = PyBullet(visualize=visualize)
        super().__init__(sim=simulator, max_episode_len=max_episode_len, *args, **kwargs)


class PyBullet(Simulator):
    normed_starting_agent_obs = np.array([-.75, -.5])
    _viz_lock_taken = False
    __filelocation__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    _red_ball_fname = os.path.join(__filelocation__, 'assets/little_ball.urdf')
    _green_ball_fname = os.path.join(__filelocation__, 'assets/immaterial_ball.urdf')
    _labyrinth_fname = os.path.join(__filelocation__, "assets/labyrinth.urdf")
    _arrow_fname = os.path.join(__filelocation__, "assets/arrow.urdf")
    _labyrinth_position = [7.5, -5/2, 1.5/2]
    _ball_radius = 0.3
    _env_lower_bound = np.array([-2.25, -2.25]) + _ball_radius -0.01
    _env_upper_bound = np.array([17.25, 7.25]) - _ball_radius + 0.01
    _norm, _denorm = normalizer(_env_lower_bound, _env_upper_bound)
    _norm, _denorm = staticmethod(_norm), staticmethod(_denorm)
    _green_ball_initial_pos = [2, 0, _ball_radius]
    _red_ball_initial_pos = [0, 0, _ball_radius]

    def __init__(self, visualize=False):
        self._visualize = visualize
        if visualize:
            assert not self._viz_lock_taken, "only one PyBullet simulation can be visualized simultaneously"
            PyBullet._viz_lock_taken = True

        self._bullet = BulletClient(connection_mode=pybullet.GUI if visualize else pybullet.DIRECT)
        self._bullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        self._bullet.setGravity(0, 0, -9.81)
        self._bullet.loadURDF("plane.urdf")
        self._agent_ball = self._bullet.loadURDF(self._red_ball_fname, self._red_ball_initial_pos)
        self._goal_ball = self._bullet.loadURDF(self._green_ball_fname, [2, 0, self._ball_radius], useFixedBase=1)
        self._bullet.loadURDF(self._labyrinth_fname, self._labyrinth_position, useFixedBase=1)
        self._arrow = self._bullet.loadURDF(self._arrow_fname, [2, 0, 2*self._ball_radius], useFixedBase=1)

    def step(self, action: np.ndarray) -> np.ndarray:
        sim_step_per_sec = 240
        agent_actions_per_sec = 10
        for _ in range(sim_step_per_sec // agent_actions_per_sec):
            if self._visualize:
                self._update_force_arrow_viz(force=action)
                time.sleep(1 / sim_step_per_sec)
            _apply_force(self._bullet, obj=self._agent_ball, force=action)
            self._bullet.stepSimulation()
        return self._norm(_position(self._bullet.getBasePositionAndOrientation(self._agent_ball)))

    def _update_force_arrow_viz(self, force: np.ndarray) -> None:
        xforce, yforce = force
        yaw = np.angle(complex(xforce, yforce))
        quaternion = self._bullet.getQuaternionFromEuler([0, 0, yaw])
        agent_pos = _position(self._bullet.getBasePositionAndOrientation(self._agent_ball))
        _reset_object(self._bullet, self._arrow, [*agent_pos, 2*self._ball_radius], quaternion=quaternion)

    def set_agent_pos(self, pos: np.ndarray) -> None:
        pos = self._denorm(pos)
        _reset_object(self._bullet, self._agent_ball, pos=[*pos, self._ball_radius])

    def set_goal_pos(self, pos: np.ndarray) -> None:
        pos = self._denorm(pos)
        _reset_object(self._bullet, self._goal_ball, pos=[*pos, self._ball_radius])

    def is_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        achieved, desired = self._denorm(achieved), self._denorm(desired)
        return _goals_are_close(achieved_goal=achieved, desired_goal=desired)

    def render(self, *args, **kwargs):
        pass


def distance(x1: np.ndarray, x2: np.ndarray):
    return np.linalg.norm(x1 - x2) ** 2


def _goals_are_close(achieved_goal: np.ndarray, desired_goal: np.ndarray):
    ε = PyBullet._ball_radius
    return distance(achieved_goal, desired_goal) < ε


def _position(position_and_orientation: Sequence[float]) -> np.ndarray:
    return np.array(position_and_orientation[0])[:2]  # [pos, quaternion]


def _reset_object(bc: BulletClient, obj, pos: Sequence[float], quaternion=None):
    quaternion = quaternion if quaternion else [0, 0, 0, 1]
    bc.resetBasePositionAndOrientation(obj, pos, quaternion)


_zforce = 0
_force_multiplier = 5  # tuned value
def _apply_force(bc: BulletClient, obj, force: Sequence[float]):
    force = _force_multiplier * np.array([*force, _zforce])
    obj_pos, _  = bc.getBasePositionAndOrientation(obj)
    bc.applyExternalForce(objectUniqueId=obj, linkIndex=-1,
                          forceObj=force, posObj=obj_pos, flags=pybullet.WORLD_FRAME)


if __name__ == '__main__':
    env = Labyrinth(visualize=True)
    obs = env.reset()
    for t in count():
        action = obs.desired_goal - obs.achieved_goal
        obs = env.step(action / np.linalg.norm(action))[0]
        print(f"step {t}")
