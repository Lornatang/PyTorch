import os

import torch
import torchvision
from torch import nn, optim
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '../../data/CALTECH/4/'
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_CLASSES = 10

MODEL_PATH = '../../../models/pytorch/CALTECH/'
MODEL_NAME = '4.pth'

# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize(224),  # 将图像转化为224 * 224
    transforms.RandomHorizontalFlip(),  # 几率随机旋转
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])


# Load data
train_datasets = torchvision.datasets.ImageFolder(root=WORK_DIR + 'train/',
                                                  transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

val_datasets = torchvision.datasets.ImageFolder(root=WORK_DIR + 'val/',
                                                transform=transform)

val_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)


def main():
    print(f"Train numbers:{len(train_datasets)}")
    print(f"Val numbers:{len(val_datasets)}")

    model = Net()
    # cast
    cast = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-8)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        step = 1
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

            print(f"Step [{step * 64}/{10 * len(train_datasets)}], "
                  f"Loss: {loss.item():.8f}.")

        # Save the model checkpoint
        torch.save(model, MODEL_PATH + MODEL_NAME)
    print(f"Model save to {MODEL_PATH + MODEL_NAME}.")


if __name__ == '__main__':
    main()
