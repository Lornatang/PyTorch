import os

import torch
import torchvision
from research.CALTECH.C101.net import GoogleNet
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '../../../../../data/CALTECH/C101/'
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_CLASSES = 101

MODEL_DIR = '../../../../models/pytorch/CALTECH/'
MODEL_NAME = 'C101.pth'

# Create model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

transform = transforms.Compose([
    transforms.Resize(96),  # 96 * 96
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # lrn
])


# Load data
train_dataset = torchvision.datasets.ImageFolder(root=WORK_DIR + 'train/',
                                                 transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)


def main():
    print(f"Train numbers:{len(train_dataset)}")

    model = GoogleNet()
    model.train()
    # cast
    cast = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-8)
    step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
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

            print(f"Step [{step * 64}/{int(10 * len(train_dataset))}], "
                  f"Loss: {loss.item():.8f}.")
            step += 1

        # Save the model checkpoint
        torch.save(model, MODEL_DIR + MODEL_NAME)
    print(f"Model save to {MODEL_DIR + MODEL_NAME}.")


if __name__ == '__main__':
    main()
