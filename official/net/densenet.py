import os
import argparse
import random
import time

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
from collections import OrderedDict


parser = argparse.ArgumentParser(description='PyTorch AlexNet Training')
parser.add_argument(
  '--dataset',
  required=True,
  help='cifar-10/100 | fmnist/mnist | folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument(
  '--classes',
  type=int,
  required=True,
  help='classes of pictures')
parser.add_argument(
  '--workers',
  type=int,
  help='number of data loading workers',
  default=4)
parser.add_argument(
  '--batchSize',
  type=int,
  default=64,
  help='inputs batch size')
parser.add_argument(
  '--imageSize',
  type=int,
  default=224,
  help='the height / width of the inputs image to network')
parser.add_argument(
  '--niter',
  type=int,
  default=50,
  help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument(
  '--weight-decay',
  default=1e-4,
  type=float,
  metavar='W',
  help='weight decay (default: 1e-4)')
parser.add_argument(
  '-p',
  '--print-freq',
  default=10,
  type=int,
  metavar='N',
  help='print frequency (default: 10)')
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


def densenet121(**kwargs):
  r"""Densenet-121 model from
  `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
  """
  model = DenseNet(
    num_init_features=64,
    growth_rate=32,
    block_config=(
      6,
      12,
      24,
      16),
    **kwargs)
  return model


def densenet169(**kwargs):
  r"""Densenet-169 model from
  `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
  """
  model = DenseNet(
    num_init_features=64,
    growth_rate=32,
    block_config=(
      6,
      12,
      32,
      32),
    **kwargs)
  return model


def densenet201(**kwargs):
  r"""Densenet-201 model from
  `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
  """
  model = DenseNet(
    num_init_features=64,
    growth_rate=32,
    block_config=(
      6,
      12,
      48,
      32),
    **kwargs)
  return model


def densenet161(**kwargs):
  r"""Densenet-161 model from
  `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
  """
  model = DenseNet(
    num_init_features=96,
    growth_rate=48,
    block_config=(
      6,
      12,
      36,
      24),
    **kwargs)
  return model


class _DenseLayer(nn.Sequential):
  def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
    super(_DenseLayer, self).__init__()
    self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
    self.add_module('relu1', nn.ReLU(inplace=True)),
    self.add_module(
      'conv1',
      nn.Conv2d(
        num_input_features,
        bn_size *
        growth_rate,
        kernel_size=1,
        stride=1,
        bias=False)),
    self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
    self.add_module('relu2', nn.ReLU(inplace=True)),
    self.add_module(
      'conv2',
      nn.Conv2d(
        bn_size *
        growth_rate,
        growth_rate,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False)),
    self.drop_rate = drop_rate

  def forward(self, x):
    new_features = super(_DenseLayer, self).forward(x)
    if self.drop_rate > 0:
      new_features = nn.dropout(
        new_features,
        p=self.drop_rate,
        training=self.training)
    return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
  def __init__(
          self,
          num_layers,
          num_input_features,
          bn_size,
          growth_rate,
          drop_rate):
    super(_DenseBlock, self).__init__()
    for i in range(num_layers):
      layer = _DenseLayer(
        num_input_features +
        i *
        growth_rate,
        growth_rate,
        bn_size,
        drop_rate)
      self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
  def __init__(self, num_input_features, num_output_features):
    super(_Transition, self).__init__()
    self.add_module('norm', nn.BatchNorm2d(num_input_features))
    self.add_module('relu', nn.ReLU(inplace=True))
    self.add_module(
      'conv',
      nn.Conv2d(
        num_input_features,
        num_output_features,
        kernel_size=1,
        stride=1,
        bias=False))
    self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
  r"""Densenet-BC model class, based on
  `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

  Args:
      growth_rate (int) - how many filters to add each layer (`k` in paper)
      block_config (list of 4 ints) - how many layers in each pooling block
      num_init_features (int) - the number of filters to learn in the first convolution layer
      bn_size (int) - multiplicative factor for number of bottle neck layers
        (i.e. bn_size * k features in the bottleneck layer)
      drop_rate (float) - dropout rate after each dense layer
  """

  def __init__(
          self,
          growth_rate=32,
          block_config=(
                  6,
                  12,
                  24,
                  16),
          num_init_features=64,
          bn_size=4,
          drop_rate=0):

    super(DenseNet, self).__init__()

    # First convolution
    self.features = nn.Sequential(OrderedDict([
      ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
      ('norm0', nn.BatchNorm2d(num_init_features)),
      ('relu0', nn.ReLU(inplace=True)),
      ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    # Each denseblock
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
      block = _DenseBlock(
        num_layers=num_layers,
        num_input_features=num_features,
        bn_size=bn_size,
        growth_rate=growth_rate,
        drop_rate=drop_rate)
      self.features.add_module('denseblock%d' % (i + 1), block)
      num_features = num_features + num_layers * growth_rate
      if i != len(block_config) - 1:
        trans = _Transition(
          num_input_features=num_features,
          num_output_features=num_features // 2)
        self.features.add_module('transition%d' % (i + 1), trans)
        num_features = num_features // 2

    # Final batch norm
    self.features.add_module('norm5', nn.BatchNorm2d(num_features))

    # Linear layer
    self.classifier = nn.Linear(num_features, opt.classes)

    # Official init from torch repo.
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

  def forward(self, inputs):
    x = self.features(inputs)
    nn.ReLU(inplace=True)
    nn.AvgPool2d(kernel_size=7, stride=1)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
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
  model = densenet121().cuda()

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
    torch.save(model, f"{opt.outf}/DenseNet_epoch_{epoch + 1}.pth")
  print(f"Model save to '{opt.outf}'.")


def test():
  if torch.cuda.is_available():
    model = torch.load(f'{opt.outf}/DenseNet_epoch_{opt.niter}.pth')
  else:
    model = torch.load(f'{opt.outf}/DenseNet_epoch_{opt.niter}.pth', map_location="cpu")
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
