from GenerativeGoalLearning import train_GAN
from LSGAN import LSGAN

import torch
import numpy as np
import matplotlib.pyplot as plt

#### PARAMETERS ####
Rmin = 0.1
Rmax = 0.9
max_episode_length = 500

G_Input_Size  = 4
G_Hidden_Size = 256
D_Hidden_Size = 128

num_samples_goalGAN_goals  = 200
map_length = 10
target_point = torch.Tensor(np.array([-3., 3]))
eps = 1
####################


def label_goals(samples, target):
    return [int(torch.dist(s, target) <= eps) for s in samples]


def plot_goals(goals, target):
    x = [g[0] for g in goals.detach().numpy()]
    y = [g[1] for g in goals.detach().numpy()]

    fig = plt.figure()
    plt.ylim(-map_length, map_length); plt.xlim(-map_length, map_length)
    ax = plt.gca()
    ax.scatter(x,y)
    ax.scatter(target[0], target[1], c=['green'], s=[400], alpha=0.3)
    plt.show()





goalGAN = LSGAN(generator_input_size=G_Input_Size,
                    generator_hidden_size=G_Hidden_Size,
                    generator_output_size=2,
                    discriminator_input_size=2,
                    discriminator_hidden_size=D_Hidden_Size,
                    discriminator_output_size=1,
                    map_scale=map_length) 

### training
for i in range(200):
    z         = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
    gan_goals = goalGAN.Generator.forward(z).detach()    
    labels    = label_goals(gan_goals, target_point)
    
    print("Iteration: {},   Number of generated positive samples: {}/{}".format(i, np.sum(labels), num_samples_goalGAN_goals))
    if np.sum(labels) < 1:
        print(".. reinitializing GAN")
        goalGAN.reset_GAN()
        continue

    if np.sum(labels) > num_samples_goalGAN_goals * 0.95:
        print(".. training done")
        break
    
    #plot_goals(gan_goals,target_point)

    goalGAN   = train_GAN(gan_goals, labels, goalGAN)


### validation
for i in range(5):
    z         = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
    gan_goals = goalGAN.Generator.forward(z).detach()   
    labels    = label_goals(gan_goals, target_point)
        
    print("Number of generated positive samples: {}/{}".format(np.sum(labels), num_samples_goalGAN_goals))
    plot_goals(gan_goals,target_point)

