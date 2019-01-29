import os

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

from research.MNIST.mnist.net import Net

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '../../../../../data/MNIST/mnist'
BATCH_SIZE = 128

MODEL_PATH = '../../../../models/pytorch/MNIST'
MODEL_NAME = 'mnist.pth'

# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize(28),  # 将图像转化为800 * 800
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])


# Load data
val_dataset = torchvision.datasets.ImageFolder(root=WORK_DIR + '/' + 'val',
                                               transform=transform)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)


def main():
    print(f"Val numbers:{len(val_dataset)}")

    # Load model
    if torch.cuda.is_available():
        model = torch.load(MODEL_PATH + '/' + MODEL_NAME).to(device)
    else:
        model = torch.load(MODEL_PATH + '/' + MODEL_NAME, map_location='cpu')

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
