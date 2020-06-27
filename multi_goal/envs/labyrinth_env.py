import os
from typing import Optional

from gym import spaces
import numpy as np

from multi_goal.envs import normalizer, Observation
from multi_goal.envs.collider_env import ColliderEnv

_env_lower_bound = np.array([-2.25, -2.25]) + ColliderEnv._ball_radius -0.01
_env_upper_bound = np.array([17.25, 7.25]) - ColliderEnv._ball_radius + 0.01
_norm, _denorm = normalizer(_env_lower_bound, _env_upper_bound)


class Labyrinth(ColliderEnv):
    _labyrinth_fname = os.path.join(ColliderEnv.__filelocation__, "assets/labyrinth.urdf")
    _labyrinth_position = [7.5, -5/2, 1.5/2]
    starting_agent_pos = _norm(ColliderEnv.starting_agent_pos)

    def __init__(self, visualize: bool = False, max_episode_len: int = 200, seed=0):
        super().__init__(visualize=visualize, max_episode_len=max_episode_len, seed=seed)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        goal_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Dict(spaces={
            "observation": spaces.Box(low=0, high=0, shape=(0,), dtype=np.float),
            "desired_goal": goal_space,
            "achieved_goal": goal_space
        })
        self._bullet.loadURDF(self._labyrinth_fname, self._labyrinth_position, useFixedBase=1)
        self.seed(seed)
        self._goal = _denorm(self._sample_new_goal())

    def step(self, action: np.ndarray):
        obs, r, done, info = super().step(action)
        obs = _norm_obs(obs)
        return obs, r, done, info

    def reset(self):
        return _norm_obs(super().reset())

    def set_possible_goals(self, goals: Optional[np.ndarray], entire_space=False) -> None:
        if goals is not None:
            goals = _denorm(goals)
            if goals.size == 2:
                goals = goals[None]

        super().set_possible_goals(goals, entire_space)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> float:
        return super().compute_reward(_denorm(achieved_goal), _denorm(desired_goal), info)


def _norm_obs(obs: Observation) -> Observation:
    desired_goal = _norm(obs.desired_goal)
    achieved_goal = _norm(obs.achieved_goal)
    return Observation(observation=obs.observation, achieved_goal=achieved_goal, desired_goal=desired_goal)


if __name__ == '__main__':
    env = Labyrinth(visualize=True)
    while True:
        print(env.step(env.action_space.high)[0])
