import warnings
from GenerativeGoalLearning import initialize_GAN, update_and_eval_policy, \
    label_goals, train_GAN, sample, eval_policy, update_replay
from ppo_agent import PPOAgent
from two_blocks_env.collider_env import dim_goal
import numpy as np
import torch
from torchsummary import summary
from two_blocks_env.toy_labyrinth_env import ToyLab
from utils import display_goals

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
initial_iterations         = 5
iterations                 = 1000
num_samples_goalGAN_goals  = 60
num_samples_from_old_goals = num_samples_goalGAN_goals // 2
num_samples_random_goals   = num_samples_goalGAN_goals // 2

SEED = 1000 #random.randint(0,2**32-1)
np.random.seed(SEED)
torch.set_default_dtype(torch.float64)
####################

env  = ToyLab()
π         = PPOAgent(env=env)
goalGAN   = initialize_GAN(env=env)
goals_old = torch.Tensor([env.starting_obs]) + torch.randn(num_samples_from_old_goals, dim_goal(env))*0.1

### print model summary
goalGAN.Generator=goalGAN.Generator.float(); goalGAN.Discriminator=goalGAN.Discriminator.float()
summary(goalGAN.Generator,     input_size=(1,1,4), device='cpu')
summary(goalGAN.Discriminator, input_size=(1,1,2), device='cpu')
goalGAN.Generator=goalGAN.Generator.double(); goalGAN.Discriminator=goalGAN.Discriminator.double()
#######################

import pdb; pdb.set_trace()

## Initial training of the policy with random goals
goals_plot = None
for i in range(initial_iterations):
    rand_goals = torch.tensor(np.random.uniform(-1, 0, size=(num_samples_random_goals, 2)))
    π,returns  = update_and_eval_policy(rand_goals, π, env)
    print(f"Average reward: {(sum(returns) / len(returns)):.2f}")
    labels     = label_goals(returns)
    print(f"Percentage of 0 vs 1 labels: {[round(n, 2) for n in (np.bincount(labels) / len(labels))]}")
    display_goals(rand_goals.detach().numpy(), returns, i, env, fileNamePrefix='_')
    goalGAN    = train_GAN(rand_goals, labels, goalGAN)


for i in range(initial_iterations, iterations):
    print(f"\n### BEGIN ITERATION {i} ###")
    z          = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
    gan_goals  = goalGAN.Generator.forward(z).detach()
    rand_goals = torch.tensor(np.random.uniform(-1, 1, size=(num_samples_random_goals, 2)))
    goals      = torch.cat([gan_goals, sample(goals_old, k=num_samples_from_old_goals), rand_goals])
    π, returns = update_and_eval_policy(goals, π, env)
    display_goals(goals.detach().numpy(), returns, i, env)
    print(f"Average reward: {(sum(returns) / len(returns)):.2f}")
    labels     = label_goals(returns)
    print(f"Percentage of 0 vs 1 labels: {[round(n, 2) for n in (np.bincount(labels) / len(labels))]}")
    if all([lab == 0 for lab in labels]): warnings.warn("All labels are 0")
    goalGAN    = train_GAN(goals, labels, goalGAN)
    goals_old  = update_replay(goals, goals_old=goals_old)
