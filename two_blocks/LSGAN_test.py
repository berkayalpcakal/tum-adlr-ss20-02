from GenerativeGoalLearning import initialize_GAN, train_GAN, sample
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

num_samples_goalGAN_goals  = 500
map_length = 1
target_point = torch.Tensor(np.array([-0.2, -0.2]))
eps = 0.1
####################


def label_goals(samples, target):
    return [int(torch.dist(s, target) <= eps) for s in samples]


def plot_goals(goals):
    x = [g[0] for g in goals.detach().numpy()]
    y = [g[1] for g in goals.detach().numpy()]

    plt.ylim(-map_length, map_length); plt.xlim(-map_length, map_length)
    plt.scatter(x,y)
    plt.show()





goalGAN = LSGAN(generator_input_size=G_Input_Size,
                    generator_hidden_size=G_Hidden_Size,
                    generator_output_size=2,
                    discriminator_input_size=2,
                    discriminator_hidden_size=D_Hidden_Size,
                    discriminator_output_size=1) 

for i in range(10):
    
    z         = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
    gan_goals = goalGAN.Generator.forward(z).detach()    
    labels    = label_goals(gan_goals, target_point)
    
    print("Number of generated positive samples: {}".format(np.sum(labels)))
    plot_goals(gan_goals)

    goalGAN   = train_GAN(gan_goals, labels, goalGAN)


