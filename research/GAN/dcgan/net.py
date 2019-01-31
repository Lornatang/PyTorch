from torch import nn


NOISE = 100


# Generator
class Generator(nn.Module):
    def __init__(self, noise=NOISE):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(noise, 32 * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32 * 4, 32 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32 * 2, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32 * 2, 32 * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(32 * 4, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        out = x.view(-1, 1).squeeze(1)
        return out
