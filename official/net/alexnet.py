import os
import argparse
import random

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch AlexNet Training')
parser.add_argument(
  '--dataset',
  required=True,
  help='cifar10/100 | fmnist/mnist | folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument(
  '--workers',
  type=int,
  help='number of data loading workers',
  default=2)
parser.add_argument(
  '--batchSize',
  type=int,
  default=32,
  help='inputs batch size')
parser.add_argument(
  '--imageSize',
  type=int,
  default=32,
  help='the height / width of the inputs image to network')
parser.add_argument(
  '--niter',
  type=int,
  default=25,
  help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument(
  '--ngpu',
  type=int,
  default=1,
  help='number of GPUs to use')
parser.add_argument(
  '--outf',
  default='.',
  help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument(
  '--model',
  required=True,
  help='training models or testing models')

opt = parser.parse_args()
print(opt)

try:
  os.makedirs(opt.outf)
except OSError:
  pass

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
  print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([
  transforms.Resize(opt.imageSize),
  transforms.CenterCrop(opt.imageSize),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

if opt.model == 'train':
  if opt.dataset in 'folder':
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transform)
    nc = 3
  elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot,
                           download=True,
                           transform=transform)
    nc = 3
  elif opt.dataset == 'cifar100':
    dataset = dset.CIFAR100(root=opt.dataroot,
                            download=True,
                            transform=transform)
    nc = 3
  elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot,
                         download=True,
                         transform=transform)
    nc = 1
  elif opt.dataset == 'fmnist':
    dataset = dset.FashionMNIST(root=opt.dataroot,
                                download=True,
                                transform=transform)
    nc = 1
elif opt.model == 'test':
  if opt.dataset in 'folder':
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transform)
    nc = 3
  elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot,
                           download=True,
                           train=False,
                           transform=transform)
    nc = 3
  elif opt.dataset == 'cifar100':
    dataset = dset.CIFAR100(root=opt.dataroot,
                            download=True,
                            train=False,
                            transform=transform)
    nc = 3
  elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot,
                         download=True,
                         train=False,
                         transform=transform)
    nc = 1
  elif opt.dataset == 'fmnist':
    dataset = dset.FashionMNIST(root=opt.dataroot,
                                download=True,
                                train=False,
                                transform=transform)
    nc = 1

assert dataset
dataloader = torch.utils.data.DataLoader(
  dataset,
  batch_size=opt.batchSize,
  shuffle=True,
  num_workers=int(
    opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")

ngpu = int(opt.ngpu)


class AlexNet(nn.Module):

  def __init__(self, ngpus):
    super(AlexNet, self).__init__()
    self.ngpu = ngpus
    self.features = nn.Sequential(
      nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1),
      # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 1 * 1, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, 10),
    )

  def forward(self, inputs):
    inputs = self.features(inputs)
    inputs = inputs.view(-1, 256 * 1 * 1)
    inputs = self.classifier(inputs)
    return F.log_softmax(inputs, dim=1)


def train():
  print(f"Train numbers:{len(dataset)}")

  # load model
  model = AlexNet(ngpu).to(device)
  # Optimization
  optimizer = optim.Adam(
    model.parameters(),
    lr=opt.lr,
    betas=(opt.beta1, 0.999),
    weight_decay=1e-8)

  for epoch in range(opt.niter):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader, 0):
      data, target = data.to(device), target.to(device)
      # Forward pass
      optimizer.zero_grad()
      outputs = model(data)
      loss = F.nll_loss(outputs, target)

      # Backward and update paras
      loss.backward()
      optimizer.step()

      if batch_idx % 10 == 0:
        print(f"Train Epoch: {epoch} "
              f"[{batch_idx * len(data)}/{len(dataloader.dataset)} "
              f"({100. * batch_idx / len(dataloader):.0f}%)]\t"
              f"Loss: {loss.item():.6f}")

    # Save the model checkpoint
    torch.save(model, f"{opt.outf}/AlexNet_epoch_{epoch + 1}.pth")
  print(f"Model save to '{opt.outf}'.")


def test():
  model = torch.load(f'{opt.outf}/AlexNet_epoch_{opt.niter}.pth')
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in dataloader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      # sum up batch loss
      test_loss += F.nll_loss(output, target, reduction='sum').item()
      # get the index of the max log-probability
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(dataloader.dataset)

  print(
    f"\nTest set: Average loss: {test_loss:.4f}, "
    f"Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.0f}%)\n")


if opt.model == 'train':
  train()
elif opt.model == 'test':
  test()
