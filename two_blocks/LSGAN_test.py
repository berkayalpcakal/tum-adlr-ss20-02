from GenerativeGoalLearning import train_GAN
from LSGAN import LSGAN

import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

#### PARAMETERS ####
torch.set_default_dtype(torch.float64)
SEED = random.randint(0,2**32-1)
np.random.seed(SEED)

G_Input_Size  = 4
G_Hidden_Size = 256
D_Hidden_Size = 256

num_samples_goalGAN_goals  = 150
num_sample_random_goals    = num_samples_goalGAN_goals * 2
map_length = 1
eps = 0.1
Rmin = 0.3 
Rmax = 0.6
target_point = torch.Tensor(np.random.random(size=(2)) * 2 * map_length - map_length) /2
####################


def label_goals_naive(samples, target):
    return [int(torch.dist(s, target) <= eps) for s in samples]

def label_goals_complex(samples, target):
    return [int(Rmin <= torch.dist(s, target) <= Rmax) for s in samples]


def plot_goals(gan_goals, rand_goals, target):
    fig = plt.figure()
    plt.ylim(-map_length, map_length); plt.xlim(-map_length, map_length)
    ax = plt.gca()

    # plot target circles
    circle_rmin = plt.Circle((target[0], target[1]), Rmin, color='green', alpha=0.1)
    circle_rmax = plt.Circle((target[0], target[1]), Rmax, color='red',   alpha=0.1)
    ax.add_artist(circle_rmin)
    ax.add_artist(circle_rmax)

    # plot gan_goals
    if gan_goals is not None:
        x = [g[0] for g in gan_goals.detach().numpy()]
        y = [g[1] for g in gan_goals.detach().numpy()]
        ax.scatter(x,y, color='black')

    # plot rand_goals
    if rand_goals is not None:
        x = [g[0] for g in rand_goals.detach().numpy()]
        y = [g[1] for g in rand_goals.detach().numpy()]
        ax.scatter(x,y, color='gray')

    plt.show()


def initial_gan_train(goalGAN):
    ## initial training of GAN with random samples
    ## aim is to make G generate evenly distributed goals before starting the actual training
    ## if we do not run this, G generates goals concentrated around (0,0)

    for i in range(10):
        rand_goals  = torch.tensor(np.random.uniform(-1, 1, size=(num_samples_goalGAN_goals, 2)))        
        labels_rand = label_goals_complex(rand_goals, target_point)

        #z          = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
        #gan_goals  = goalGAN.Generator.forward(z).detach()
        #plot_goals(gan_goals, rand_goals, target_point)

        print("Init Iteration: {}".format(i))        
        goalGAN   = train_GAN(rand_goals, labels_rand, goalGAN)



def main():
    goalGAN = LSGAN(generator_input_size=G_Input_Size,
                        generator_hidden_size=G_Hidden_Size,
                        generator_output_size=2,
                        discriminator_input_size=2,
                        discriminator_hidden_size=D_Hidden_Size,
                        discriminator_output_size=1,
                        map_scale=map_length)

    initial_gan_train(goalGAN) 

    ### training
    for i in range(1000):
        z          = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
        gan_goals  = goalGAN.Generator.forward(z).detach()
        rand_goals = torch.tensor(np.random.uniform(-1, 1, size=(num_sample_random_goals,2)))
        goals      = torch.cat([gan_goals, rand_goals], axis=0)

        labels_gan  = label_goals_complex(gan_goals, target_point)
        labels_rand = label_goals_complex(rand_goals, target_point)
        labels      = labels_gan + labels_rand # concat lists 

        #plot_goals(gan_goals, rand_goals, target_point)

        print("Iteration: {},   Number of generated positive samples: {}/{}".format(i, np.sum(labels_gan), gan_goals.shape[0]))
        if np.sum(labels) < 2:
            print(".. reinitializing GAN")
            goalGAN.reset_GAN()
            continue

        if np.sum(labels_gan) > gan_goals.shape[0] * 0.95:
            print(".. training done")
            break

        goalGAN   = train_GAN(goals, labels, goalGAN)
        
    ### validation
    for i in range(5):
        z         = torch.randn(size=(2*num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
        gan_goals = goalGAN.Generator.forward(z).detach()
        labels    = label_goals_complex(gan_goals, target_point)

        print("Number of generated positive samples: {}/{}".format(np.sum(labels), gan_goals.shape[0]))
        plot_goals(gan_goals, None, target_point)

if __name__ == '__main__':
    main()