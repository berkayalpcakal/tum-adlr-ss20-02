from GenerativeGoalLearning import initialize_GAN, update_policy, evaluate_policy, \
    label_goals, train_GAN, sample, random_agent
from two_blocks_env.collider_env import ColliderEnv, dim_goal

import torch

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
π    = random_agent(action_space=env.action_space)
goalGAN   = initialize_GAN(env=env)
goals_old = torch.empty(0, dim_goal(env))

for i in range(iterations):
    z         = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
    gan_goals = goalGAN.Generator.forward(z).detach() * float(env.observation_space["desired_goal"].high[0])
    goals     = torch.cat([gan_goals, sample(goals_old, k=num_samples_from_old_goals)])
    π         = update_policy(goals, π, env)
    returns   = evaluate_policy(goals, π, env)
    labels    = label_goals(returns)
    goalGAN   = train_GAN(goals, labels, goalGAN)
    goals_old = goals
