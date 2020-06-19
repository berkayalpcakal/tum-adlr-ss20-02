import os
from typing import Tuple
import click
from stable_baselines.common import BaseRLModel
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import HER, SAC, PPO2

from GenerativeGoalLearning import trajectory, Agent, evaluate
from ppo_agent import PPOAgent
from two_blocks_env.collider_env import SettableGoalEnv
from utils import Dirs
from two_blocks_env.toy_labyrinth_env import ToyLab


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


def load_agent(model_class: type(BaseRLModel), dirs: Dirs) -> Tuple[Agent, SettableGoalEnv]:
    env = ToyLab()
    if model_class == HER:
        model = model_class.load(load_path=dirs.best_model, env=env)
        agent = lambda obs: model.predict(obs, deterministic=True)[0]
    else:
        agent = PPOAgent(env=env, loadpath=dirs.best_model)
    print(f"Loaded model {dirs.best_model}")
    return agent, env


def viz(agent: Agent, env: SettableGoalEnv):
    run = lambda g: sum(1 for _ in trajectory(agent, env, goal=g, sleep_secs=0.1, render=True, print_every=1))
    while True:
        env.render()
        g = env.observation_space["desired_goal"].sample()
        print(f"Episode len: {run(g)}")


class Mode:
    TRAIN = "train"
    VIZ   = "viz"
    EVAL  = "eval"


@click.command()
@click.option("--mode", type=click.Choice([Mode.TRAIN, Mode.VIZ, Mode.EVAL]))
@click.option("--alg", type=click.Choice(['her-sac', 'goalgan-ppo']), default="her-sac", help="Algorithm: 'her' (HER+SAC) or 'gaolgan-ppo' available.")
@click.option("--num_steps", default=100000, show_default=True)
def main(mode: str, alg: str, num_steps: int):
    model_class = HER if alg == "her-sac" else PPO2
    dirs = Dirs(experiment_name=f"{alg}-toylab")
    if mode == Mode.TRAIN:
        return train(model_class=model_class, dirs=dirs, num_steps=num_steps)

    agent, env = load_agent(model_class=model_class, dirs=dirs)
    if mode == Mode.VIZ:
        return viz(agent=agent, env=env)
    elif mode == Mode.EVAL:
        while True:
            evaluate(agent=agent, env=env)
            input("Press any key to re-compute.")


if __name__ == '__main__':
    main()
