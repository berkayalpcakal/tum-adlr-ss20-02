import click

from GenerativeGoalLearning import trajectory, Agent, evaluate
from agents import PPOAgent, HERSACAgent
from two_blocks_env.collider_env import SettableGoalEnv
from two_blocks_env.toy_labyrinth_env import ToyLab


def viz(agent: Agent, env: SettableGoalEnv):
    run = lambda g: sum(1 for _ in trajectory(agent, env, goal=g, sleep_secs=0.1, render=True, print_every=1))
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


@click.command()
@click.option("--mode", type=click.Choice([Mode.TRAIN, Mode.VIZ, Mode.EVAL]))
@click.option("--alg", type=click.Choice(['her-sac', 'ppo']), default="her-sac", help="Algorithm: 'her' (HER+SAC) or 'ppo' available.")
@click.option("--num_steps", default=100000, show_default=True)
def main(mode: str, alg: str, num_steps: int):
    env_fn = ToyLab
    env = env_fn()
    if alg == Algs.HERSAC:
        agent = HERSACAgent(env=env)
    elif alg == Algs.PPO:
        agent = PPOAgent(env=env, verbose=1)
    else:
        raise NotImplementedError

    if mode == Mode.TRAIN:
        return agent.train(timesteps=num_steps, eval_env=env_fn())
    elif mode == Mode.VIZ:
        return viz(agent=agent, env=env)
    elif mode == Mode.EVAL:
        while True:
            evaluate(agent=agent, env=env)
            input("Press any key to re-compute.")


if __name__ == '__main__':
    main()
