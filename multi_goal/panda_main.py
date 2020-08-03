import click
from more_itertools import consume

from multi_goal.GenerativeGoalLearning import trajectory
from multi_goal.agents import HERSACAgent, EvaluateCallback, GoalGANAgent
from multi_goal.envs.pybullet_panda_robot import PandaEnv


@click.command()
@click.option("--do-train", is_flag=True, default=False)
def main(do_train: bool):
    env = PandaEnv(visualize=not do_train)
    agent = HERSACAgent(env=env, experiment_name="goalgan-her-sac")
    agent = GoalGANAgent(env=env, agent=agent)
    if do_train:
        cb = EvaluateCallback(agent=agent, eval_env=PandaEnv())
        agent.train(timesteps=50000, callbacks=[cb])
    else:
        while True:
            consume(trajectory(agent, env))


def continuous_viz():
    env = PandaEnv(visualize=True)
    agent = HERSACAgent(env=env, experiment_name="goalgan-her-sac")
    obs = env.reset()
    while True:
        action = agent(obs)
        obs, _, done, info = env.step(action)
        if done:
            obs = env.reset(reset_agent_pos=not info["is_success"])


if __name__ == '__main__':
    main()
