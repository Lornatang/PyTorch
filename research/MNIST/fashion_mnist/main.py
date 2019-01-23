"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: main.py
# time: 2018/8/21 15:45
# license: MIT
"""

import argparse
import os
import time

import torch
import torchvision
from torch import nn
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../../data/MNIST/FashionMNIST/',
                    help="""image path. Default='../../data/MNIST/FashionMNIST/'.""")
parser.add_argument('--epochs', type=int, default=10,
                    help="""num epochs. Default=10""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""0 ~ 9,. Default=10""")
parser.add_argument('--batch_size', type=int, default=128,
                    help="""batch size. Default=128""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learning_rate. Default=0.0001""")
parser.add_argument('--model_path', type=str, default='../../../models/pytorch/MNIST/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='fashion_mnist.pth',
                    help="""Model name""")
parser.add_argument('--display_epoch', type=int, default=1)
args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

# Define transforms.
train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(24),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.RandomCrop(24),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Fashion mnist dataset
train_dataset = torchvision.datasets.FashionMNIST(root=args.path,
                                                  train=True,
                                                  transform=train_transform,
                                                  download=True)

test_dataset = torchvision.datasets.FashionMNIST(root=args.path,
                                                 train=False,
                                                 transform=test_transform,
                                                 download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)


def main():
    # Load model
    model = torchvision.models.resnet18(pretrained=True).to(device)
    model.avgpool = nn.AvgPool2d(1, 1).to(device)
    model.fc = nn.Linear(512, 10).to(device)
    print(model)
    # cast
    cast = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    model.train()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.display_epoch == 0 or epoch == 1:
            end = time.time()
            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")

            # Test the model
            model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
            correct = 0.
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Test Accuracy: {100 * correct / total:.2f}.")

        # Save the model checkpoint
        torch.save(model, args.model_path + args.model_name)
    print(f"Model save to {args.model_path + args.model_name}.")


if __name__ == '__main__':
    main()
