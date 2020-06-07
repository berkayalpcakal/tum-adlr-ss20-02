import os

import click
from stable_baselines.common import BaseRLModel

from stable_baselines.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines import HER, DDPG, SAC, PPO2

from GenerativeGoalLearning import trajectory
from two_blocks_env.labyrinth_env import Labyrinth
from two_blocks_env.toy_labyrinth_env import ToyLab
from utils import Dirs


def train(model_class: type(BaseRLModel), dirs: Dirs):
    num_timesteps = 100000
    num_checkpoints = 4
    env = ToyLab()
    cb = CheckpointCallback(save_freq=num_timesteps//num_checkpoints, save_path=dirs.models, name_prefix=dirs.prefix)
    options = {"env": env, "tensorboard_log": dirs.tensorboard}
    if os.path.isdir(dirs.models):
        model = model_class.load(load_path=dirs.best_model, **options)
    else:
        if model_class == HER:
            options.update(model_class=SAC)
        model = model_class(policy='MlpPolicy', verbose=1, **options)
    model.learn(total_timesteps=num_timesteps, callback=cb)


def viz(model_class: type(BaseRLModel), dirs: Dirs):
    env = ToyLab()
    model = model_class.load(load_path=dirs.best_model, env=env)
    print(f"Loaded model {dirs.best_model}")
    agent = lambda obs: model.predict(obs, deterministic=True)[0]
    run = lambda g: sum(1 for _ in trajectory(agent, env, goal=g, sleep_secs=0.1))
    while True:
        env.render()
        g = env.observation_space["desired_goal"].sample()
        print(f"Episode len: {run(g)}")


@click.command()
@click.option("--do-train", is_flag=True, help="Whether to train the agent")
@click.option("--alg", type=click.Choice(['her-sac', 'ppo']), required=True, help="Algorithm: 'her' (HER+SAC) or 'ppo' available.")
def main(do_train: bool, alg: str):
    model_class = HER if alg == "her-sac" else PPO2
    dirs = Dirs(experiment_name=f"{alg}-toylab")
    if do_train:
        train(model_class=model_class, dirs=dirs)
    viz(model_class=model_class, dirs=dirs)


if __name__ == '__main__':
    main()
