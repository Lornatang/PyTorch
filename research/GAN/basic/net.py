from torch import nn

IMAGE_SIZE = 28 * 28 * 1
HIDDEN_SIZE = 1024

NOISE = 64


# Generator
class Generator(nn.Module):
    def __init__(self, noise=NOISE):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(noise, HIDDEN_SIZE),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Sigmoid()
        )
        self.layer3 = nn.Linear(HIDDEN_SIZE, IMAGE_SIZE)

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
            nn.Linear(IMAGE_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
