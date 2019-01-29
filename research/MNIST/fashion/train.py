import os

import time
import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

# first train run this code
from research.MNIST.fashion.net import Net
# incremental training comments out that line of code.

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '../../../../../data/MNIST/fashion'
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 2e-5
NUM_CLASSES = 10

MODEL_PATH = '../../../../models/pytorch/MNIST'
MODEL_NAME = 'fashion.pth'

# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize(28),  # 将图像转化为28 * 28
    transforms.RandomHorizontalFlip(),  # 几率随机旋转
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])


# Load data
train_dataset = torchvision.datasets.ImageFolder(root=WORK_DIR + '/' + 'train',
                                                 transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)


def main():
    print(f"Train numbers:{len(train_dataset)}")

    # load model
    model = Net().to(device)
    # add train data
    # if torch.cuda.is_available():
    #     model = torch.load(MODEL_PATH + MODEL_NAME).to(device)
    # else:
    #     model = torch.load(MODEL_PATH + MODEL_NAME, map_location='cpu')
    # cast
    cast = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-8)

    step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        for images, labels in train_loader:
            start = time.time()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time()
            print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(train_dataset)}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * 1:.1f}sec!")
            step += 1

        # Save the model checkpoint
        torch.save(model, MODEL_PATH + '/' + MODEL_NAME)
    print(f"Model save to {MODEL_PATH + '/' + MODEL_NAME}.")


if __name__ == '__main__':
    main()
