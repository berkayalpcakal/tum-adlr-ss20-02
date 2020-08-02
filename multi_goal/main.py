from multi_goal.agents import PPOAgent, GoalGANAgent, HERSACAgent, EvaluateCallback
from multi_goal.envs.toy_labyrinth_env import ToyLab


if __name__ == '__main__':
    seed  = 0
    env   = ToyLab(seed=seed)
    π     = PPOAgent(env=env, experiment_name="goalgan-ppo", rank=seed)
    #π     = HERSACAgent(env=env, experiment_name="goalgan-her-sac", rank=seed)
    agent = GoalGANAgent(env=env, agent=π)

    callback = EvaluateCallback(agent=agent, eval_env=ToyLab(seed=seed))
    agent.train(timesteps=int(1e6), callbacks=[callback])

