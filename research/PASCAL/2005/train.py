"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: train.py
# time: 2018/8/24 17:52
# license: MIT
"""

import argparse
import os
import time

import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms, models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classify!""")
parser.add_argument('--path', type=str, default='../../data/PASCAL/2005/',
                    help="""image dir path default: '../../data/PASCAL/2005/'.""")
parser.add_argument('--epochs', type=int, default=10,
                    help="""Epoch default:10.""")
parser.add_argument('--batch_size', type=int, default=128,
                    help="""Batch_size default:128.""")
parser.add_argument('--lr', type=float, default=1e-4,
                    help="""learning_rate. Default=1e-4""")
parser.add_argument('--num_classes', type=int, default=6,
                    help="""num classes. Default: 6.""")
parser.add_argument('--model_path', type=str, default='../../../models/pytorch/PASCAL/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='2005.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch', type=int, default=1)

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(128),  # 将图像转化为800 * 800
    transforms.RandomHorizontalFlip(0.5),  # 有0.75的几率随机旋转
    transforms.RandomCrop(114),  # 从图像中裁剪一个24 * 24的
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])


# Load data
train_datasets = torchvision.datasets.ImageFolder(root=args.path + 'train/',
                                                  transform=transform)

train_loader = data.DataLoader(dataset=train_datasets,
                               batch_size=args.batch_size,
                               shuffle=True)

test_datasets = torchvision.datasets.ImageFolder(root=args.path + 'val/',
                                                 transform=transform)

test_loader = data.DataLoader(dataset=test_datasets,
                              batch_size=args.batch_size,
                              shuffle=True)


def main():
    print(f"Train numbers:{len(train_datasets)}")
    print(f"Test numbers:{len(test_datasets)}")

    # Load model
    # if torch.cuda.is_available():
    #     model = torch.load(args.model_path + args.model_name).to(device)
    # else:
    #     model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model = models.resnet18(pretrained=True).to(device)
    model.avgpool = nn.AvgPool2d(4, 1).to(device)
    model.fc = nn.Linear(512, args.num_classes).to(device)
    print(model)
    # cast
    cast = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # start time
        start = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.display_epoch == 0:
            end = time.time()
            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")

            model.eval()

            correct = 0.
            total = 0
            for images, labels in test_loader:
                # to GPU
                images = images.to(device)
                labels = labels.to(device)
                # print prediction
                outputs = model(images)
                # equal prediction and acc
                _, predicted = torch.max(outputs.data, 1)
                # val_loader total
                total += labels.size(0)
                # add correct
                correct += (predicted == labels).sum().item()

            print(f"Acc: {100 * correct / total:.4f}.")

        # Save the model checkpoint
        torch.save(model, args.model_path + args.model_name)
    print(f"Model save to {args.model_path + args.model_name}.")


if __name__ == '__main__':
    main()
