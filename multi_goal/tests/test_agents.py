from multi_goal.agents import PPOAgent, HERSACAgent, EvaluateCallback
from multi_goal.envs.toy_labyrinth_env import ToyLab


def test_class_instantiation():
    env = ToyLab()
    a1 = PPOAgent(env=env)
    a2 = HERSACAgent(env=env)
    cb = EvaluateCallback(agent=a1, env=env)
