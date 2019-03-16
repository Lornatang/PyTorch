import os
import time

import torch
import torchvision
from torch import nn, optim
from torch.nn import init
from torch.utils import data
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '/tmp/imagenet'
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_CLASSES = 10

MODEL_PATH = './models'
MODEL_NAME = 'SqueezenNet.pth'

# Create model
if not os.path.exists(MODEL_PATH):
  os.makedirs(MODEL_PATH)

transform = transforms.Compose([
  transforms.RandomCrop(256, padding=32),
  transforms.RandomResizedCrop(224),
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


class Fire(nn.Module):
  
  def __init__(self, inplanes, squeeze_planes,
               expand1x1_planes, expand3x3_planes):
    super(Fire, self).__init__()
    self.inplanes = inplanes
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.squeeze_activation = nn.ReLU(inplace=True)
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                               kernel_size=1)
    self.expand1x1_activation = nn.ReLU(inplace=True)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                               kernel_size=3, padding=1)
    self.expand3x3_activation = nn.ReLU(inplace=True)
  
  def forward(self, x):
    x = self.squeeze_activation(self.squeeze(x))
    return torch.cat([
      self.expand1x1_activation(self.expand1x1(x)),
      self.expand3x3_activation(self.expand3x3(x))
    ], 1)


class SqueezeNet(nn.Module):
  
  def __init__(self, num_classes=NUM_CLASSES):
    super(SqueezeNet, self).__init__()
    
    self.num_classes = num_classes
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
      Fire(64, 16, 64, 64),
      Fire(128, 16, 64, 64),
      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
      Fire(128, 32, 128, 128),
      Fire(256, 32, 128, 128),
      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
      Fire(256, 48, 192, 192),
      Fire(384, 48, 192, 192),
      Fire(384, 64, 256, 256),
      Fire(512, 64, 256, 256),
    )
    # Final convolution is initialized differently form the rest
    final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
    self.classifier = nn.Sequential(
      nn.Dropout(p=0.5),
      final_conv,
      nn.ReLU(inplace=True),
      nn.AvgPool2d(13, stride=1)
    )
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if m is final_conv:
          init.normal(m.weight.data, mean=0.0, std=0.01)
        else:
          init.kaiming_uniform(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()
  
  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x.view(x.size(0), self.num_classes)


def main():
  print(f"Train numbers:{len(train_dataset)}")
  
  # load model
  model = SqueezeNet().to(device)
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
