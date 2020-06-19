import os
import warnings
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.her import HERGoalEnvWrapper

from utils import Dirs

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    from stable_baselines import PPO2, HER, SAC
from GenerativeGoalLearning import Agent, evaluate
from two_blocks_env.collider_env import Observation, SettableGoalEnv


class PPOAgent(Agent):
    def __init__(self, env: SettableGoalEnv, verbose=0, experiment_name="ppo"):
        self._dirs = Dirs(experiment_name=experiment_name + "-" + type(env).__name__)
        self._env = env
        self._flat_env = HERGoalEnvWrapper(env)
        options = {"env": DummyVecEnv([lambda: self._flat_env]), "tensorboard_log": self._dirs.tensorboard}
        if os.path.isdir(self._dirs.models):
            self._model = PPO2.load(load_path=self._dirs.best_model, **options)
            print(f"Loaded model {self._dirs.best_model}")
        else:
            self._model = PPO2("MlpPolicy", verbose=verbose, **options)

    def __call__(self, obs: Observation) -> np.ndarray:
        flat_obs = self._flat_env.convert_dict_to_obs(obs)
        action, _ = self._model.predict(flat_obs, deterministic=True)
        return action

    def train(self, timesteps: int, num_checkpoints=4):
        ppo_offset = 128
        cb = make_callback(timesteps=timesteps, num_checkpoints=num_checkpoints,
                           dirs=self._dirs, agent=self, env=self._env)
        self._model.learn(total_timesteps=timesteps+ppo_offset, callback=cb)


class HERSACAgent(Agent):
    def __init__(self, env: SettableGoalEnv):
        self._dirs = Dirs(experiment_name="her-sac-" + type(env).__name__)
        self._env = env
        options = {"env": env, "tensorboard_log": self._dirs.tensorboard, "model_class": SAC}
        if os.path.isdir(self._dirs.models):
            self._model = HER.load(load_path=self._dirs.best_model, **options)
            print(f"Loaded model {self._dirs.best_model}")
        else:
            self._model = HER(policy="MlpPolicy", verbose=1, **options)

    def __call__(self, obs: Observation) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action

    def train(self, timesteps: int) -> None:
        num_checkpoints = 4
        cb = make_callback(timesteps, num_checkpoints, self._dirs, self, self._env)
        self._model.learn(total_timesteps=timesteps, callback=cb)


def make_callback(timesteps: int, num_checkpoints: int, dirs: Dirs, agent: Agent, env: SettableGoalEnv):
    return CallbackList([
        CheckpointCallback(save_freq=timesteps//num_checkpoints, save_path=dirs.models, name_prefix=dirs.prefix),
        EvalCallback(freq=timesteps//num_checkpoints, agent=agent, env=env)
    ])


class EvalCallback(BaseCallback):
    def __init__(self, freq: int, agent: Agent, env: SettableGoalEnv):
        super().__init__()
        self._freq = freq
        self._agent = agent
        self._settable_goal_env = env

    def _on_step(self) -> bool:
        if self.n_calls % self._freq == 0:
            evaluate(agent=self._agent, env=self._settable_goal_env)
        return True
