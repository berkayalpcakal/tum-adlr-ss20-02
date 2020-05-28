import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib


class GoalGAN():
    def __init__(self, generator_input_size = 14, generator_hidden_size=256, generator_output_size=3,
                 discriminator_input_size = 3, discriminator_hidden_size=128, discriminator_output_size=1):
        self.Generator     = Generator(generator_input_size, generator_hidden_size, generator_output_size)
        self.Discriminator = Discriminator(discriminator_input_size, discriminator_hidden_size, discriminator_output_size)

        self.G_Optimizer   = torch.optim.RMSprop(self.Generator.parameters(),     lr=0.001, alpha=0.99)
        self.D_Optimizer   = torch.optim.RMSprop(self.Discriminator.parameters(), lr=0.001, alpha=0.99)

        self.generator_losses     = []
        self.discriminator_losses = []

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()

        self.loss = nn.MSELoss()    # TODO Change Loss

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh() * 10  # tanh returns values between [-1,1]
        )

        self.all_layers = nn.Sequential(
            self.layer1,
            self.layer2
        )

        weights_xavier_init(self)

    def forward(self, input):
        output = self.all_layers(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()

        self.loss = nn.MSELoss()    # TODO Change Loss

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
        )

        self.all_layers = nn.Sequential(
            self.layer1,
            self.layer2
        )

        weights_xavier_init(self)

    def forward(self, input):
        output = self.all_layers(input)
        return output


def weights_xavier_init(model):
    for m in model._modules:     
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
