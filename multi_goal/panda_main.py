import click
from more_itertools import consume

from multi_goal.GenerativeGoalLearning import trajectory
from multi_goal.agents import HERSACAgent, EvaluateCallback
from multi_goal.envs.pybullet_panda_robot import PandaEnv


@click.command()
@click.option("--do-train", is_flag=True, default=False)
def main(do_train: bool):
    env = PandaEnv(visualize=not do_train)
    agent = HERSACAgent(env=env)
    if do_train:
        cb = EvaluateCallback(agent=agent, eval_env=PandaEnv())
        agent.train(timesteps=50000, callbacks=[cb])
    while True:
        consume(trajectory(agent, env))


if __name__ == '__main__':
    main()
