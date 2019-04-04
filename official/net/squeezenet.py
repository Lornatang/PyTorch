import os
import argparse
import random
import time

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset

parser = argparse.ArgumentParser(description='PyTorch AlexNet Training')
parser.add_argument('--dataset', required=True, help='cifar-10/100 | fmnist/mnist | folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--classes', type=int, required=True, help='classes of pictures')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=32, help='inputs batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the inputs image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--net', default='', help='model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--model', default='train', help='training models or testing models. (default: train)')

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
  elif opt.dataset == 'cifar-10':
    dataset = dset.CIFAR10(root=opt.dataroot,
                           download=True,
                           transform=transform)
    nc = 3
  elif opt.dataset == 'cifar-100':
    dataset = dset.CIFAR100(root=opt.dataroot,
                            download=True,
                            transform=transform)
    nc = 3
  elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot,
                         download=True,
                         transform=transforms.Compose([
                           transforms.Resize(opt.imageSize),
                           transforms.CenterCrop(opt.imageSize),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                         ]))
    nc = 1
  elif opt.dataset == 'fmnist':
    dataset = dset.FashionMNIST(root=opt.dataroot,
                                download=True,
                                transform=transforms.Compose([
                                  transforms.Resize(opt.imageSize),
                                  transforms.CenterCrop(opt.imageSize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                ]))
    nc = 1
elif opt.model == 'test':
  if opt.dataset in 'folder':
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transform)
    nc = 3
  elif opt.dataset == 'cifar-10':
    dataset = dset.CIFAR10(root=opt.dataroot,
                           download=True,
                           train=False,
                           transform=transform)
    nc = 3
  elif opt.dataset == 'cifar-100':
    dataset = dset.CIFAR100(root=opt.dataroot,
                            download=True,
                            train=False,
                            transform=transform)
    nc = 3
  elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot,
                         download=True,
                         train=False,
                         transform=transforms.Compose([
                           transforms.Resize(opt.imageSize),
                           transforms.CenterCrop(opt.imageSize),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                         ]))
    nc = 1
  elif opt.dataset == 'fmnist':
    dataset = dset.FashionMNIST(root=opt.dataroot,
                                download=True,
                                train=False,
                                transform=transforms.Compose([
                                  transforms.Resize(opt.imageSize),
                                  transforms.CenterCrop(opt.imageSize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                ]))
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

# define CrossEntropyLoss()
criterion = nn.CrossEntropyLoss().cuda()


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

  def __init__(self):
    super(SqueezeNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(nc, 64, kernel_size=3, stride=2),
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
    final_conv = nn.Conv2d(512, 101, kernel_size=1)
    self.classifier = nn.Sequential(
      nn.Dropout(p=0.5),
      final_conv,
      nn.ReLU(inplace=True),
      nn.AvgPool2d(1, stride=1)  # raw kernel_size=13, stride=1. For use img size 224 * 224.
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if m is final_conv:
          nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
        else:
          nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

  def forward(self, inputs):
    x = self.features(inputs)
    x = self.classifier(x)
    x = x.view(x.size(0), opt.classes)
    return x


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.count = 0
    self.sum = 0
    self.avg = 0
    self.val = 0
    self.name = name
    self.fmt = fmt

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
  def __init__(self, num_batches, *meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def print(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  @staticmethod
  def _get_batch_fmtstr(num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train():
  print(f"Train numbers:{len(dataset)}")

  # load model
  model = SqueezeNet().cuda()

  # define optimizer
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=opt.lr,
                               betas=(opt.beta1, 0.9),
                               weight_decay=opt.weight_decay)

  for epoch in range(opt.niter):
    end = time.time()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
      len(dataloader),
      batch_time,
      data_time,
      losses,
      top1,
      top5,
      prefix=f"Epoch: [{epoch + 1}]")

    model.train()

    for i, (data, target) in enumerate(dataloader, 0):
      # measure data loading time
      data_time.update(time.time() - end)

      data, target = data.cuda(), target.cuda()

      # compute output
      output = model(data)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), data.size(0))
      top1.update(acc1[0], data.size(0))
      top5.update(acc5[0], data.size(0))

      # compute gradient and do Adam step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % opt.print_freq == 0:
        progress.print(i)

    # Save the model checkpoint
    torch.save(model, f"{opt.outf}/SqueezeNet_epoch_{epoch + 1}.pth")
  print(f"Model save to '{opt.outf}'.")


def test():
  if torch.cuda.is_available():
    model = torch.load(f'{opt.outf}/Squeeze_Net_epoch_{opt.niter}.pth')
  else:
    model = torch.load(f'{opt.outf}/Squeeze_Net_epoch_{opt.niter}.pth', map_location="cpu")
  model.eval()

  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':6.3f')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
    len(dataloader),
    batch_time,
    losses,
    top1,
    top5,
    prefix='Test: ')

  with torch.no_grad():
    end = time.time()
    for i, (data, target) in enumerate(dataloader):
      data, target = data.cuda(), target.cuda()
      # compute output
      output = model(data)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), data.size(0))
      top1.update(acc1[0], data.size(0))
      top5.update(acc5[0], data.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % opt.print_freq == 0:
        progress.print(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))


if __name__ == '__main__':
  if opt.model == 'train':
    train()
  elif opt.model == 'test':
    test()
