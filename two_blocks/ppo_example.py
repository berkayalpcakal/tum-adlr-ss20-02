"""Taken from https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html"""
import os
import time
import numpy as np
from itertools import count

from gym.wrappers import FlattenObservation
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from two_blocks_env.collider_env import ColliderEnv


model_fname = "ppo2_collider"


def train_and_save_ppo():
    env = FlattenObservation(ColliderEnv(visualize=False))
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save(model_fname)


def load_and_run_ppo():
    model = PPO2.load(model_fname)
    env = FlattenObservation(ColliderEnv(visualize=True))
    while True:
        obs = env.reset()
        for t in count(1):
            action, _ = model.predict(obs)
            #action = perfect_action(obs)
            obs, _, done, _ = env.step(action)
            time.sleep(1/240)
            if done:
                break
            if t % 10 == 0:
                print(f"step {t}")


# For comparison
def perfect_action(obs) -> np.ndarray:
    ball_pos = obs[4:6]
    target_pos = obs[6:8]
    return (target_pos - ball_pos) / np.linalg.norm(target_pos - ball_pos)


if __name__ == '__main__':
    if not os.path.exists(model_fname + ".zip"):
        train_and_save_ppo()
    else:
        load_and_run_ppo()
