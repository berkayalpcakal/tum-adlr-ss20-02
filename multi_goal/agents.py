import os
import warnings
from datetime import datetime
from typing import List, Sequence

import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.her import HERGoalEnvWrapper

from multi_goal.utils import Dirs

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    from stable_baselines import PPO2, HER, SAC
from multi_goal.GenerativeGoalLearning import Agent, evaluate
from multi_goal.envs import Observation, ISettableGoalEnv


class PPOAgent(Agent):
    def __init__(self, env: ISettableGoalEnv, verbose=0, experiment_name="ppo", rank=0):
        self._env = env
        self._dirs = Dirs(experiment_name=f"{type(env).__name__}-{experiment_name}-{rank}")
        self._flat_env = HERGoalEnvWrapper(env)
        options = {"env": DummyVecEnv([lambda: self._flat_env]), "tensorboard_log": self._dirs.tensorboard,
                   "gamma": 1, "seed": rank, "nminibatches": 1}
        if os.path.isdir(self._dirs.models):
            self._model = PPO2.load(load_path=self._dirs.best_model, **options)
            print(f"Loaded model {self._dirs.best_model}")
        else:
            self._model = PPO2("MlpPolicy", verbose=verbose, **options)

    def __call__(self, obs: Observation) -> np.ndarray:
        flat_obs = self._flat_env.convert_dict_to_obs(obs)
        action, _ = self._model.predict(flat_obs, deterministic=True)
        return action

    def train(self, timesteps: int, num_checkpoints=4, eval_env: ISettableGoalEnv = None):
        ppo_offset = 128
        env = self._env if eval_env is None else eval_env
        cb = CallbackList(make_callbacks(timesteps=timesteps, num_checkpoints=num_checkpoints,
                                         dirs=self._dirs, agent=self, env=env))
        self._model.learn(total_timesteps=timesteps+ppo_offset, callback=cb)


class HERSACAgent(Agent):
    def __init__(self, env: ISettableGoalEnv, verbose=1, rank=0, experiment_name="her-sac"):
        self._env = env
        self._dirs = Dirs(experiment_name=f"{type(env).__name__}-{experiment_name}", rank=rank)
        options = {"env": env, "tensorboard_log": self._dirs.tensorboard, "model_class": SAC,
                   "policy_kwargs": dict(layers=[128]*2), "gamma": 1}
        if os.path.isdir(self._dirs.models):
            self._model = HER.load(load_path=self._dirs.best_model, **options)
            print(f"Loaded model {self._dirs.best_model}")
        else:
            self._model = HER(policy="MlpPolicy", verbose=verbose, **options)

    def __call__(self, obs: Observation) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action

    def train(self, timesteps: int, eval_env: ISettableGoalEnv = None,
              callbacks: Sequence[BaseCallback] = None) -> None:
        num_checkpoints = 4
        env = self._env if eval_env is None else eval_env
        callbacks = [] if callbacks is None else callbacks
        cb = CallbackList([*make_callbacks(timesteps, num_checkpoints, self._dirs, self, env), *callbacks])
        self._model.learn(total_timesteps=timesteps, callback=cb)


def make_callbacks(timesteps: int, num_checkpoints: int, dirs: Dirs,
                   agent: Agent, env: ISettableGoalEnv) -> List[BaseCallback]:
    return [
        CheckpointCallback(save_freq=timesteps//num_checkpoints, save_path=dirs.models, name_prefix=dirs.prefix),
        #EvaluateCallback(agent=agent, env=env)
    ]


class EvaluateCallback(BaseCallback):
    def __init__(self, agent: Agent, env: ISettableGoalEnv):
        super().__init__()
        self._agent = agent
        self._settable_goal_env = env
        self._last_eval = datetime.now()

    def _on_step(self) -> bool:
        if (datetime.now() - self._last_eval).seconds > 20:
            evaluate(agent=self._agent, env=self._settable_goal_env)
            self._last_eval = datetime.now()
        return True
