from GenerativeGoalLearning import initialize_GAN, update_and_eval_policy, \
    label_goals, train_GAN, sample
from ppo_agent import PPOAgent
from two_blocks_env.collider_env import dim_goal
import numpy as np
import torch
from two_blocks_env.toy_labyrinth_env import ToyLab
from utils import render_goals_with_env

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
iterations                 = 100 #1000
num_samples_goalGAN_goals  = 20
num_samples_from_old_goals = 10
####################

env  = ToyLab()
env.render()
π         = PPOAgent(env=env)
goalGAN   = initialize_GAN(env=env)
goals_old = torch.Tensor([env.starting_obs]) + torch.randn(10, dim_goal(env))*0.1

goals_plot = None
for i in range(iterations):
    print(f"\n### BEGIN ITERATION {i} ###")
    z          = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
    gan_goals  = goalGAN.Generator.forward(z).detach()
    goals      = torch.cat([gan_goals, sample(goals_old, k=num_samples_from_old_goals)])
    goals_plot = render_goals_with_env(goals, goals_plot, env)
    π, returns = update_and_eval_policy(goals, π, env)
    print(f"Average reward: {(sum(returns) / len(returns)):.2f}")
    labels     = label_goals(returns)
    print(f"Percentage of 0 vs 1 labels: {[round(n, 2) for n in (np.bincount(labels) / len(labels))]}")
    assert not all([lab == 0 for lab in labels]), "All labels are 0"
    goalGAN    = train_GAN(goals, labels, goalGAN)
    goals_old  = goals
