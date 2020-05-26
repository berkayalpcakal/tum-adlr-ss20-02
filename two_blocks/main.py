import numpy as np

from goalGAN import RandomAgent, initialize_GAN, sample, update_policy, evaluate_policy, \
    label_goals, train_GAN, update_replay
from two_blocks_env.collider_env import ColliderEnv

from spinup import ppo_pytorch as ppo
import torch.nn as nn
import pybullet as p

"""
Algorithm in the GAN paper, Floarensa 2018 

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
iterations = 1
num_goalGAN_goals = 200
####################

env = ColliderEnv(visualize=True)
π = RandomAgent(action_space=env.action_space)
G, D = initialize_GAN(obs_space=env.observation_space)
goals_old = []

for i in range(iterations):
    z = np.random.normal(size=num_goalGAN_goals)
    goals = G(z) + sample(goals_old)
    π = update_policy(goals, π, env)
    returns = evaluate_policy(goals, π, env)
    labels = label_goals(returns)
    G, D = train_GAN(goals, labels, G, D)
    goals_old = update_replay(goals)
