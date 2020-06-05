import os
from pathlib import Path

import click
from stable_baselines.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines import HER, DDPG, SAC, PPO2

from GenerativeGoalLearning import trajectory
from two_blocks_env.labyrinth_env import Labyrinth
from two_blocks_env.toy_labyrinth_env import ToyLab
from utils import latest_model

experiment_name = "her-sac-toylab"
results_dir = Path("../all-results")/experiment_name
models_dir = results_dir/"ckpts"
tensorboard_dir = results_dir/"tensorboard"
model_fpath = lambda: models_dir/latest_model(models_dir)


def train():
    num_timesteps = 100000
    num_checkpoints = 4
    env = ToyLab()
    cb = CheckpointCallback(save_freq=num_timesteps//num_checkpoints, save_path=models_dir, name_prefix=experiment_name)

    if os.path.isdir(models_dir):
        model = HER.load(model_fpath(), env=env, tensorboard_log=tensorboard_dir)
    else:
        model = HER('MlpPolicy', env, model_class=SAC, tensorboard_log=tensorboard_dir,
                    verbose=1)
    model.learn(total_timesteps=num_timesteps, callback=cb)


def viz():
    env = ToyLab()
    model = HER.load(load_path=model_fpath(), env=env)
    print(f"Loaded model {model_fpath()}")
    agent = lambda obs: model.predict(obs)[0]
    run = lambda g: sum(1 for _ in trajectory(agent, env, goal=g, sleep_secs=0.1))
    while True:
        env.render()
        g = env.observation_space["desired_goal"].sample()
        print(f"Episode len: {run(g)}")


@click.command()
@click.option('--do-train', default=False)
def main(do_train: bool):
    if do_train:
        train()
    viz()


if __name__ == '__main__':
    main()
