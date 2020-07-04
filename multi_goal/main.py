import warnings
from multi_goal.GenerativeGoalLearning import initialize_GAN, update_and_eval_policy, \
    label_goals, train_GAN, sample, update_replay, Agent
from multi_goal.LSGAN import LSGAN
from multi_goal.agents import PPOAgent
import numpy as np
import torch
from torchsummary import summary
from multi_goal.envs import dim_goal, ISettableGoalEnv
from multi_goal.envs.toy_labyrinth_env import ToyLab
from multi_goal.utils import display_goals

"""
Algorithm in the GAN paper, Florensa 2018 

for i in iterations:
    z         = sample_noise()                     # input for goal generator network
    goals     = G(z) union goals_old               # concat old goals with the generated ones
    π         = update_policy(goals, π)            # perform policy update, paper uses TRPO, Leon suggested to use PPO as it is simpler
    returns   = evaluate_policy(goals, π)          # needed to label the goals
    labels    = label_goals(goals)                 # needed to train discriminator network
    G, D      = train_GAN(goals, labels, G, D)
    goals_old = goals                               

"""

#### PARAMETERS ####
# The rest of the params are in goalGAN.py
initial_iterations  = 5
iterations          = 1000
num_gan_goals       = 60
num_old_goals       = num_gan_goals // 2
num_rand_goals      = num_gan_goals // 2

SEED = 0
np.random.seed(SEED)
torch.random.manual_seed(SEED)
####################


def main(π: Agent, goalGAN: LSGAN, env: ISettableGoalEnv, eval_env: ISettableGoalEnv):
    # Initial training of the policy with random goals
    for i in range(initial_iterations):
        rand_goals = torch.Tensor(num_rand_goals, dim_goal(env)).uniform_(-1, 0)
        π, returns = update_and_eval_policy(rand_goals, π, env, eval_env)
        print(f"Average reward: {(sum(returns) / len(returns)):.2f}")
        labels     = label_goals(returns)
        display_goals(rand_goals.detach().numpy(), returns, i, env, fileNamePrefix='_')
        goalGAN    = train_GAN(rand_goals, labels, goalGAN)

    close_to_starting_pos = torch.Tensor([env.starting_agent_pos]) + 0.1*torch.randn(num_old_goals, dim_goal(env))
    goals_old = torch.clamp(close_to_starting_pos, min=-1, max=1)

    for i in range(initial_iterations, iterations):
        print(f"\n### BEGIN ITERATION {i} ###")
        z          = torch.randn(size=(num_gan_goals, goalGAN.Generator.noise_size))
        gan_goals  = goalGAN.Generator.forward(z).detach()
        rand_goals = torch.Tensor(num_rand_goals, dim_goal(env)).uniform_(-1, 1)
        all_goals  = torch.cat([gan_goals, sample(goals_old, k=num_old_goals), rand_goals])
        π, returns = update_and_eval_policy(all_goals, π, env, eval_env)
        display_goals(all_goals.detach().numpy(), returns, i, env, gan_goals=gan_goals.numpy())
        print(f"Average reward: {(sum(returns) / len(returns)):.2f}")
        labels     = label_goals(returns)
        if all([lab == 0 for lab in labels]): warnings.warn("All labels are 0")
        goalGAN    = train_GAN(all_goals, labels, goalGAN)
        goals_old  = update_replay(all_goals, goals_old=goals_old)


if __name__ == '__main__':
    env       = ToyLab()
    eval_env  = ToyLab()  # Must be a different instance, used for visualizations only.
    π         = PPOAgent(env=env, experiment_name="goalgan-ppo")
    goalGAN   = initialize_GAN(env=env)

    summary(goalGAN.Generator,     input_size=(1, 1, 4), device='cpu')
    summary(goalGAN.Discriminator, input_size=(1, 1, 2), device='cpu')

    main(π=π, goalGAN=goalGAN, env=env, eval_env=eval_env)
