"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: prediction.py
# time: 2019/1/23 10:18
# license: MIT
"""

import os

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '../../data/CALTECH/102/'
BATCH_SIZE = 64
NUM_CLASSES = 102

MODEL_DIR = '../../../models/pytorch/CALTECH/'
MODEL_NAME = '102.pth'

# Create model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

transform = transforms.Compose([
    transforms.Resize(224),  # 将图像转化为800 * 800
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])


# Load data
val_datasets = torchvision.datasets.ImageFolder(root=WORK_DIR + 'val/',
                                                transform=transform)

val_loader = torch.utils.data.DataLoader(dataset=val_datasets,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)


def main():
    print(f"Val numbers:{len(val_datasets)}")

    # Load model
    if torch.cuda.is_available():
        model = torch.load(MODEL_DIR + MODEL_NAME).to(device)
    else:
        model = torch.load(MODEL_DIR + MODEL_NAME, map_location='cpu')

    model.eval()

    correct = 0.
    total = 0
    for images, labels in val_loader:
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

    print(f"Acc: {correct / total:.4f}.")


if __name__ == '__main__':
    main()
