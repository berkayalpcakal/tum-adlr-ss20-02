import os
import time
from gym import spaces
import numpy as np

from two_blocks_env.collider_env import ColliderEnv


class Labyrinth(ColliderEnv):
    _labyrinth_fname = os.path.join(ColliderEnv.__filelocation__, "assets/labyrinth.urdf")
    _labyrinth_position = [7.5, -5/2, 1.5/2]

    goal_space = spaces.Box(low=np.array([-2, -2]), high=np.array([17, 7]), dtype=np.float)
    observation_space = spaces.Dict(spaces={
        "observation": spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float),
        "desired_goal": goal_space,
        "achieved_goal": goal_space
    })

    def __init__(self, visualize: bool = True, max_episode_len: int = 200):
        super().__init__(visualize, max_episode_len)
        self._bullet.loadURDF(self._labyrinth_fname, self._labyrinth_position, useFixedBase=1)


if __name__ == '__main__':
    env = Labyrinth()
    while True:
        time.sleep(1)
