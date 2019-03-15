import os
import time

import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '/tmp/cifar10'
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_CLASSES = 10

MODEL_PATH = './models'
MODEL_NAME = 'AlexNet.pth'

# Create model
if not os.path.exists(MODEL_PATH):
  os.makedirs(MODEL_PATH)

transform = transforms.Compose([
  transforms.RandomCrop(36, padding=4),
  transforms.RandomResizedCrop(32),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load data
train_dataset = torchvision.datasets.ImageFolder(root=WORK_DIR + '/' + 'train',
                                                 transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)


class AlexNet(nn.Module):
  
  def __init__(self, num_classes=NUM_CLASSES):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      # nets.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      # nets.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 3 * 3, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes),
    )
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x


def main():
  print(f"Train numbers:{len(train_dataset)}")
  
  # load model
  model = AlexNet().to(device)
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
    
    # cal one epoch time
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
      
      print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(train_dataset)}], "
            f"Loss: {loss.item():.8f}.")
      step += 1
    
    # cal train one epoch time
    end = time.time()
    print(f"Epoch [{epoch}/{NUM_EPOCHS}], "
          f"time: {end - start} sec!")
    
    # Save the model checkpoint
    torch.save(model, MODEL_PATH + '/' + MODEL_NAME)
  print(f"Model save to {MODEL_PATH + '/' + MODEL_NAME}.")


if __name__ == '__main__':
  main()
