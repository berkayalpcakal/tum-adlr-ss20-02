import warnings
from typing import Callable
import numpy as np
from gym.wrappers import FlattenObservation
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    from stable_baselines import PPO2
from GenerativeGoalLearning import Agent
from two_blocks_env.collider_env import Observation, SettableGoalEnv


class PPOAgent(Agent):
    def __init__(self, env: SettableGoalEnv):
        self._env = env
        self._flat_env = FlattenObservation(env)
        self._model = PPO2("MlpPolicy", env=self._flat_env, verbose=0)

    def __call__(self, obs: Observation) -> np.ndarray:
        flat_obs = self._flat_env.observation(obs)
        action, _ = self._model.predict(flat_obs, deterministic=True)
        return action

    def train(self, timesteps: int, callback: Callable = None):
        self._model.learn(total_timesteps=timesteps)
