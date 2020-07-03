from random import shuffle

import numpy as np
import torch
from scipy.stats import multivariate_normal

from multi_goal.GenerativeGoalLearning import train_GAN
from multi_goal.LSGAN import LSGAN
from multi_goal.utils import get_updateable_contour


def cirlce_coords(n_mixture, radius=0.7):
    some_rotation = 0.567
    thetas = np.linspace(0, 2 * np.pi, n_mixture) + some_rotation
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    return xs, ys


def multimodal_sample(batch_size, n_mixture=8, std=0.01):
    xs, ys = cirlce_coords(n_mixture)
    ms = [multivariate_normal(mean=[x, y], cov=std) for x, y in zip(xs, ys)]
    from_which_dist = np.random.randint(0, n_mixture, batch_size)
    return np.array([ms[idx].rvs() for idx in from_which_dist])


def do_label(pts, noise):
    all_data = np.vstack((pts, noise))
    labels   = np.vstack((np.ones((num_samples, 1)), np.zeros((num_samples, 1))))
    zipped = list(zip(all_data, labels))
    shuffle(zipped)
    all_data, labels = zip(*zipped)
    return all_data, labels


contour_fn = get_updateable_contour(xlim=(-1, 1), ylim=(-1, 1))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots()
    num_samples = 100
    num_mixtures = 8
    gan = LSGAN(generator_output_size=2, discriminator_input_size=2, gen_variance_coeff=0)
    G = gan.Generator
    ax.scatter(*cirlce_coords(num_mixtures), c="red")
    pts_scatter = ax.scatter([], [], c="orange")
    gan_scatter = ax.scatter([], [], c="blue")
    while True:
        pts = multimodal_sample(num_samples, num_mixtures)
        noise = np.random.uniform(-1, 1, size=(num_samples, 2))
        all_data, labels = do_label(pts, noise)
        gan = train_GAN(goals=torch.Tensor(all_data), labels=torch.Tensor(labels), goalGAN=gan)
        gan_pts = G(torch.randn(num_samples, 4)).detach().numpy()

        contour_fn(gan_pts, ax)
        pts_scatter.set_offsets(pts)
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()