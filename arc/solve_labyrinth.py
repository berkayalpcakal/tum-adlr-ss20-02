import click

from multi_goal.GenerativeGoalLearning import trajectory, Agent, evaluate
from multi_goal.agents import PPOAgent, HERSACAgent
from multi_goal.envs import ISettableGoalEnv
from multi_goal.envs.pybullet_labyrinth_env import Labyrinth, HardLabyrinth
from multi_goal.envs.toy_labyrinth_env import ToyLab


def viz(agent: Agent, env: ISettableGoalEnv):
    run = lambda g: sum(1 for _ in trajectory(
        agent, env, goal=g, sleep_secs=0.05, render=True, print_every=1))

    while True:
        env.render()
        g = env.observation_space["desired_goal"].sample()
        print(f"Episode len: {run(g)}")


class Algs:
    HERSAC = "her-sac"
    PPO    = "ppo"


class Mode:
    TRAIN = "train"
    VIZ   = "viz"
    EVAL  = "eval"


class Envs:
    TOY = "toy"
    BULLET = "bullet"
    BULLET_HARD = "bullet_hard"


@click.command()
@click.option("--mode", type=click.Choice([Mode.TRAIN, Mode.VIZ, Mode.EVAL]))
@click.option("--alg", type=click.Choice(['her-sac', 'ppo']), default="her-sac", help="Algorithm: 'her' (HER+SAC) or 'ppo' available.")
@click.option("--env", type=click.Choice(["toy", "bullet", "bullet_hard"]), default="toy", help="Simple Labyrinth env or Bullet one")
@click.option("--num_steps", default=100000, show_default=True)
@click.option("--random_starts", is_flag=True, default=False, show_default=True)
def cmd_main(*args, **kwargs):
    main(*args, **kwargs)


def main(mode: str, alg: str, num_steps: int, random_starts: bool, env: str):
    env_dict = {Envs.TOY: ToyLab, Envs.BULLET: Labyrinth, Envs.BULLET_HARD: HardLabyrinth}
    env_fn = env_dict[env]

    params = dict(use_random_starting_pos=random_starts)
    if env == Envs.BULLET or env == Envs.BULLET_HARD:
        params["visualize"] = mode == Mode.VIZ
    e = env_fn(**params)
    fixed_start_env = env_fn(use_random_starting_pos=False)

    if alg == Algs.HERSAC:
        agent = HERSACAgent(env=e)
    elif alg == Algs.PPO:
        agent = PPOAgent(env=e, verbose=1)
    else:
        raise NotImplementedError

    if mode == Mode.TRAIN:
        return agent.train(timesteps=num_steps, eval_env=fixed_start_env)
    elif mode == Mode.VIZ:
        return viz(agent=agent, env=e)
    elif mode == Mode.EVAL:
        while True:
            evaluate(agent=agent, env=fixed_start_env, very_granular=True)
            input("Press any key to re-compute.")
    else:
        raise NotImplementedError


if __name__ == '__main__':
    cmd_main()
