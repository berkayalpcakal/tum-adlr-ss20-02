from random import shuffle

import numpy as np
import torch
from scipy.stats import multivariate_normal
from sklearn.neighbors import KDTree

from multi_goal.GenerativeGoalLearning import train_GAN
from multi_goal.LSGAN import LSGAN
from multi_goal.utils import get_updateable_contour

####
start_point = np.array([-0.25, -0.50])
num_samples = 100
num_noise_samples = num_samples // 2
num_mixtures = 4
eps = 0.2

def create_target_trajectory():
    # for simple U-shaped maze
    delta = 0.01

    trajectory = []; angles = []
    trajectory.append(start_point); angles.append(np.array([0.567]))

    # 1. bottom-left to bottom-right
    while trajectory[-1][0] < 0.50:
        trajectory.append(trajectory[-1] + np.array([delta, 0]))
        angles.append(np.array([0.567]))

    # 2. bottom-right to upper-right
    while trajectory[-1][1] < 0.50:
        trajectory.append(trajectory[-1] + np.array([0, delta]))
        angles.append(np.array([0.567 - np.pi/2]))

    # 3. upper-right to upper-left
    while trajectory[-1][0] > -0.5:
        trajectory.append(trajectory[-1] + np.array([-delta, 0]))
        angles.append(np.array([0.567 - np.pi]))

    return trajectory, angles

def cirlce_coords(n_mixture, radius=0.4, center=start_point, rotation=0.567):
    thetas = np.linspace(0, 2 * np.pi//2, n_mixture, endpoint=False) + rotation
    xs, ys = center[0] + radius * np.sin(thetas), center[1] + radius * np.cos(thetas)
    return xs, ys


def multimodal_sample(batch_size, xs, ys, n_mixture=8, std=0.003):
    ms = [multivariate_normal(mean=[x, y], cov=std) for x, y in zip(xs, ys)]
    from_which_dist = np.random.randint(0, n_mixture, batch_size)
    return np.array([ms[idx].rvs() for idx in from_which_dist])


def do_label(target_pts, train_data, eps):
    labels    = np.zeros(shape=(train_data.shape[0]))
    tree      = KDTree(target_pts)
    dist, idx =tree.query(train_data)
    labels[np.argwhere(dist < eps)[:,0].reshape(-1)] = 1
    print("Ratio of positive samples; {}".format(np.sum(labels)/train_data.shape[0]))

    return labels


contour_fn = get_updateable_contour(xlim=(-1, 1), ylim=(-1, 1))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots()
    gan = LSGAN(generator_output_size=2, discriminator_input_size=2, gen_variance_coeff=0.001, generator_hidden_size=64, discriminator_hidden_size=128)
    G = gan.Generator
    trajectory, angles = create_target_trajectory()

    pts_scatter = ax.scatter([], [], c="orange")
    gan_scatter = ax.scatter([], [], c="blue")

    gan_pts = G(torch.randn(num_samples, 4)).detach().numpy()
    for i in range(len(trajectory)):
        circle_center = trajectory[i]; angle = angles[i]
        xs, ys = cirlce_coords(num_mixtures, center=circle_center, rotation=angle)
        target_pts = multimodal_sample(num_samples, xs, ys, num_mixtures)
        noise = np.random.uniform(-1, 1, size=(num_noise_samples, 2))
        train_data = np.concatenate((noise, gan_pts), axis=0)
        labels = do_label(target_pts, train_data, eps)
        gan = train_GAN(goals=torch.Tensor(train_data), labels=torch.Tensor(labels), goalGAN=gan)
        gan_pts = G(torch.randn(num_samples, 4)).detach().numpy()

        contour_fn(gan_pts, ax)
        pts_scatter.set_offsets(target_pts)
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
