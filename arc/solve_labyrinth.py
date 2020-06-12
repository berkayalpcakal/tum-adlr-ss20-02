import os

import click
from stable_baselines.common import BaseRLModel

from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import HER, SAC, PPO2

from GenerativeGoalLearning import trajectory
from app_utils import Dirs
from two_blocks_env.toy_labyrinth_env import ToyLab

# TODO: Currently not fully working for PPO.

def train(model_class: type(BaseRLModel), dirs: Dirs, num_steps: int):
    num_checkpoints = 4
    env = ToyLab()
    options = {"env": env, "tensorboard_log": dirs.tensorboard}
    if os.path.isdir(dirs.models):
        model = model_class.load(load_path=dirs.best_model, **options)
    else:
        if model_class == HER:
            options.update(model_class=SAC)
        model = model_class(policy='MlpPolicy', verbose=1, **options)

    cb = CheckpointCallback(save_freq=num_steps//num_checkpoints, save_path=dirs.models, name_prefix=dirs.prefix)
    model.learn(total_timesteps=num_steps, callback=cb)


def viz(model_class: type(BaseRLModel), dirs: Dirs):
    env = ToyLab()
    model = model_class.load(load_path=dirs.best_model, env=env)
    print(f"Loaded model {dirs.best_model}")
    agent = lambda obs: model.predict(obs, deterministic=True)[0]
    run = lambda g: sum(1 for _ in trajectory(agent, env, goal=g, sleep_secs=0.1, render=True, print_every=1))
    while True:
        env.render()
        g = env.observation_space["desired_goal"].sample()
        print(f"Episode len: {run(g)}")


@click.command()
@click.option("--do-train", is_flag=True, help="Whether to train the agent")
@click.option("--alg", type=click.Choice(['her-sac', 'ppo']), default="her-sac", help="Algorithm: 'her' (HER+SAC) or 'ppo' available.")
@click.option("--num_steps", default=100000, show_default=True)
def main(do_train: bool, alg: str, num_steps: int):
    model_class = HER if alg == "her-sac" else PPO2
    dirs = Dirs(experiment_name=f"{alg}-toylab")
    if do_train:
        train(model_class=model_class, dirs=dirs, num_steps=num_steps)
    viz(model_class=model_class, dirs=dirs)


if __name__ == '__main__':
    main()
