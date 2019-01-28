import os

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '../../../../../data/CIFAR/cifar10'
BATCH_SIZE = 16

MODEL_PATH = '../../../../models/pytorch/CIFAR/'
MODEL_NAME = '10.pth'

# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
        model = torch.load(MODEL_PATH + MODEL_NAME).to(device)
    else:
        model = torch.load(MODEL_PATH + MODEL_NAME, map_location='cpu')

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
