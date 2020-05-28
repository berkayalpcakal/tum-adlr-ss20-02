import numpy as np

from GenerativeGoalLearning import RandomAgent, initialize_GAN, update_policy, evaluate_policy, \
    label_goals, train_GAN, update_replay, concatenate_goals
from two_blocks_env.collider_env import ColliderEnv

from spinup import ppo_pytorch as ppo
import torch.nn as nn
import torch
import pybullet as p

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
iterations                 = 3 #1000
num_samples_goalGAN_goals  = 3 #200
num_samples_from_old_goals = 2 #100
####################

env  = ColliderEnv(visualize=True)
π    = RandomAgent(action_space=env.action_space)
G, D = initialize_GAN(obs_space=env.observation_space)
goals_old = np.array([]).reshape((-1,2,1))

for i in range(iterations):
    z         = torch.Tensor( np.random.normal(size=(num_samples_goalGAN_goals, G.noise_size)) )
    gan_goals = G.forward(z).detach().numpy().reshape(num_samples_goalGAN_goals, 2, 1) * float(env.goal_space.high[0])
    goals     = concatenate_goals(gan_goals, goals_old, num_samples_from_old_goals)
    π         = update_policy(goals, π, env)
    returns   = evaluate_policy(goals, π, env)
    labels    = label_goals(returns)
    G, D      = train_GAN(goals, labels, G, D)
    goals_old = goals
