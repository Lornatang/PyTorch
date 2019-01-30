from torch import nn

IMAGE_SIZE = 28 * 28 * 3
G_HIDDEN_SIZE = 1024
D_HIDDEN_SIZE = 256

NOISE = 100


# Generator
class Generator(nn.Module):
    def __init__(self, noise=NOISE):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(noise, G_HIDDEN_SIZE),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(G_HIDDEN_SIZE, G_HIDDEN_SIZE),
            nn.ReLU(True)
        )
        self.layer3 = nn.Linear(G_HIDDEN_SIZE, IMAGE_SIZE)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(IMAGE_SIZE, D_HIDDEN_SIZE),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(D_HIDDEN_SIZE, D_HIDDEN_SIZE),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Linear(D_HIDDEN_SIZE, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
