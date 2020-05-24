import time
import numpy as np
from itertools import count
from two_blocks_env.collider_env import ColliderEnv, distance

from spinup import ppo_pytorch as ppo
import torch.nn as nn
import pybullet as p

"""
Algorithm in the GAN paper, Floarensa 2018 

for i in iterations:
    z         = sample_noise()                      # input for goal generator network
    goals     = G(z) union goals_old                # concat old goals with the generated ones
    pi        = update_policy(goals, pi)            # perform policy update, paper uses TRPO, Leon suggested to use PPO as it is simpler
    returns   = evaluate_policy(goals, pi)          # needed to label the goals
    labels    = label_goals(goals)                  # needed to train discriminator network
    G, D      = train_GAN(goals, labels, G, D)      # train GAN
    goals_old = goals                               

"""

#### PARAMETERS ####
iterations = 1
episode_length = 1000
Rmin = 0.1
Rmax = 0.9
####################

#### Setup environment
env = ColliderEnv(visualize=True)
obs = env.reset()

goals_old = []
#TODO: G, D = initialize GAN

##### Main Loop
for i in range(iterations):
    ### Reset environment
    env.reset()

    ### Sample goals using Generator TODO
    blue_pose = env._get_obs()['observation'][7:10].reshape((3,1))
    new_goal = blue_pose #np.array([3, 3, 0.5]).reshape((3, 1))
    env.set_goal(new_goal)

    ### Run policy optimization step, update policy using sampled goals TODO
    # We need to have a for loop for itarating over goals, perform policy update for each goal   

    done = False
    action = env.action_space.sample()

    for t in range(0, episode_length):
        obs, reward, done, _ = env.step(action)
        time.sleep(1/240)

        if t % 100 == 0:
            print(f"achieved goal: {obs.achieved_goal.T},"
                  f"desired goal: {obs.desired_goal.T},"
                  f" distance to desired goal: {distance(obs.achieved_goal, obs.desired_goal)}")

        if done:
            print("SUCCESS!")
            break


    ### Evaluate policy, get the returns for the updated policy TODO


    ### Label goals, according to the equation 4 in the GAN paper TODO
    # labels[g] = 1; if R_min < R_g(pi) < R_max for every g in goals

    ### Traing GAN using labels and goals

    
