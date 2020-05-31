import os
import time
from itertools import count

from gym.wrappers import FlattenObservation
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import SAC, HER

from two_blocks_env.labyrinth_env import Labyrinth


model_fname = "her-sac-labyrinth"
if not os.path.isfile(model_fname + ".zip"):
    num_timesteps = 100000
    checkpoint_callback = CheckpointCallback(save_freq=num_timesteps//10, save_path='./checkpoints/',
                                             name_prefix=model_fname)
    env = Labyrinth(visualize=False)
    model = HER('MlpPolicy', env, model_class=SAC, tensorboard_log=model_fname + "_tensorboard")
    model.learn(total_timesteps=num_timesteps, callback=checkpoint_callback)
    model.save(model_fname)
    del model

env = Labyrinth(visualize=True)
model = HER.load(model_fname, env=env)
obs = env.reset()

while True:
    obs = env.reset()
    for t in count(1):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        time.sleep(1/240)
        if done:
            break
        if t % 10 == 0:
            print(action)
            print(f"step {t}")
