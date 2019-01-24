import torch
from torch import nn

NUM_CLASSES = 16


class inception(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels1_1,
                 out_channels2_1,
                 out_channels2_3,
                 out_channels3_1,
                 out_channels3_5,
                 out_channels4_1):
        super(inception, self).__init__()
        # First line
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1_1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels1_1),
            nn.ReLU(inplace=True)
        )

        # Second line
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels2_1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels2_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels2_1, out_channels2_3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels2_3),
            nn.ReLU(inplace=True)
        )

        # Third line
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3_1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels3_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels3_1, out_channels3_5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels3_5),
            nn.ReLU(inplace=True)
        )

        # Fourth line
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels4_1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels4_1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        out = torch.cat((f1, f2, f3, f4), dim=1)
        return out


class GoogLeNet(nn.Module):
    """use google network(inception v3).
    input img size is 96 * 96"""

    def __init__(self, num_classes=NUM_CLASSES):
        super(GoogLeNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(kernel_size=2)
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
