import os
import argparse
import random

import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset

import denset_util

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist | folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='inputs batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the inputs image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

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

if opt.dataset in 'folder':
  # folder dataset
  dataset = dset.ImageFolder(root=opt.dataroot,
                             transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.CenterCrop(opt.imageSize),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
  nc = 3
elif opt.dataset == 'cifar10':
  dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                           transforms.Resize(opt.imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
  nc = 3

elif opt.dataset == 'mnist':
  dataset = dset.MNIST(root=opt.dataroot, download=True,
                       transform=transforms.Compose([
                         transforms.Resize(opt.imageSize),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,)),
                       ]))
  nc = 1

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")


def train():
  print(f"Train numbers:{len(dataset)}")
  
  # load model
  model = denset_util.densenet121().to(device)
  # Optimization
  optimizer = optim.Adam(
    model.parameters(),
    lr=opt.lr,
    betas=(opt.beta1, 0.999),
    weight_decay=1e-8)
  
  for epoch in range(opt.niter):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader, 0):
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
    torch.save(model, f"{opt.outf}/DenseNet_epoch_{epoch + 1}.pth")
  print(f"Model save to '{opt.outf}'.")


def test():
  model = torch.load(f'{opt.outf}/DenseNet_epoch_{opt.niter}.pth')
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in dataloader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
  
  test_loss /= len(dataloader.dataset)
  
  print(f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(dataloader)} ({100. * correct / len(dataloader):.0f}%)\n")


train()
test()
