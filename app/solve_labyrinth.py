import os
from pathlib import Path

from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines import HER, DDPG, SAC

from GenerativeGoalLearning import trajectory
from two_blocks_env.labyrinth_env import Labyrinth
from two_blocks_env.toy_labyrinth_env import ToyLab
from utils import latest_model, vf_for_model

experiment_name = "her-sac-labyrinth"
results_dir = Path("./results")
models_dir = results_dir/"ckpts"


def train():
    num_timesteps = 100000
    num_checkpoints = 2
    env = ToyLab()
    callbacks = CallbackList([
        CheckpointCallback(save_freq=num_timesteps//num_checkpoints, save_path=models_dir, name_prefix=experiment_name),
    ])
    if os.path.isdir(models_dir):
        model_fname = latest_model(models_dir)
        model = HER.load(model_fname, env=env, tensorboard_log=results_dir/"tensorboard")
    else:
        model = HER('MlpPolicy', env, model_class=DDPG, tensorboard_log=results_dir/"tensorboard",
                    verbose=1)
    model.learn(total_timesteps=num_timesteps, callback=callbacks)


import matplotlib.pyplot as plt


def viz():
    env = ToyLab()
    model_fname = models_dir / latest_model(models_dir)
    model = HER.load(load_path=model_fname, env=env)
    print(f"Loaded model {model_fname}")
    agent = lambda obs: model.predict(obs)[0]
    run = lambda g: sum(1 for _ in trajectory(agent, env, goal=g, sleep_secs=0.1))
    obs = env.reset()
    env.render()
    vf = vf_for_model(model, obs)
    plt.quiver(*vf, units="width")
    while True:
        g = env.observation_space["desired_goal"].sample()
        print(f"Episode len: {run(g)}")


if __name__ == '__main__':
    train()
    viz()
