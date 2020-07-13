from multi_goal.agents import PPOAgent, GoalGANAgent
from multi_goal.envs.toy_labyrinth_env import ToyLab


if __name__ == '__main__':
    seed  = 1
    env   = ToyLab(seed=seed)
    π     = PPOAgent(env=env, experiment_name="goalgan-ppo", rank=seed)
    agent = GoalGANAgent(env=env, agent=π)
    agent.train(timesteps=1000000)

