from more_itertools import consume

from multi_goal.GenerativeGoalLearning import trajectory
from multi_goal.agents import HERSACAgent
from multi_goal.envs.pybullet_panda_robot import PandaEnv

if __name__ == '__main__':
    do_train = True
    env = PandaEnv(visualize=not do_train)
    agent = HERSACAgent(env=env)
    if do_train:
        agent.train(timesteps=100000)
    while True:
        consume(trajectory(agent, env))
