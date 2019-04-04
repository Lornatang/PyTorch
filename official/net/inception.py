import os
import argparse
import random
import time

import torch.nn as nn
import torch.nn.functional as F
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


def inception_v3(**kwargs):
  r"""Inception v3 model architecture from
  `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
  """
  return Inception3(**kwargs)


class Inception3(nn.Module):

  def __init__(self, aux_logits=True, transform_input=False):
    # raw aux_logits=True, transform_input=False. For use img 299 * 299.
    super(Inception3, self).__init__()
    self.aux_logits = aux_logits
    self.transform_input = transform_input
    self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
    self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
    self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
    self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
    self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
    self.Mixed_5b = InceptionA(192, pool_features=32)
    self.Mixed_5c = InceptionA(256, pool_features=64)
    self.Mixed_5d = InceptionA(288, pool_features=64)
    self.Mixed_6a = InceptionB(288)
    self.Mixed_6b = InceptionC(768, channels_7x7=128)
    self.Mixed_6c = InceptionC(768, channels_7x7=160)
    self.Mixed_6d = InceptionC(768, channels_7x7=160)
    self.Mixed_6e = InceptionC(768, channels_7x7=192)
    if aux_logits:
      self.AuxLogits = InceptionAux(768, opt.classes)
    self.Mixed_7a = InceptionD(768)
    self.Mixed_7b = InceptionE(1280)
    self.Mixed_7c = InceptionE(2048)
    self.fc = nn.Linear(2048, 101)

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        import scipy.stats as stats
        stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        X = stats.truncnorm(-2, 2, scale=stddev)
        values = torch.Tensor(X.rvs(m.weight.data.numel()))
        values = values.view(m.weight.data.size())
        m.weight.data.copy_(values)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x):
    global aux
    if self.transform_input:
      x = x.clone()
      x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
      x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
      x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    # 299 x 299 x 3
    x = self.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.Mixed_6e(x)
    # 17 x 17 x 768
    if self.training and self.aux_logits:
      aux = self.AuxLogits(x)
    # 17 x 17 x 768
    x = self.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.Mixed_7c(x)
    # 8 x 8 x 2048
    # raw kernel_size=8.For use img size 299 * 299.
    x = F.avg_pool2d(x, kernel_size=8)
    # 1 x 1 x 2048
    x = F.dropout(x, training=self.training)
    # 1 x 1 x 2048
    x = x.view(x.size(0), -1)
    # 2048
    x = self.fc(x)
    # 1000 (num_classes)
    if self.training and self.aux_logits:
      return x, aux
    return x


class InceptionA(nn.Module):

  def __init__(self, in_channels, pool_features):
    super(InceptionA, self).__init__()
    self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

    self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
    self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

    self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
    self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

    self.branch_pool = BasicConv2d(
      in_channels, pool_features, kernel_size=1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch5x5 = self.branch5x5_1(x)
    branch5x5 = self.branch5x5_2(branch5x5)

    branch3x3dbl = self.branch3x3dbl_1(x)
    branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
    branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

    branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = self.branch_pool(branch_pool)

    outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
    return torch.cat(outputs, 1)


class InceptionB(nn.Module):

  def __init__(self, in_channels):
    super(InceptionB, self).__init__()
    self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

    self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
    self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

  def forward(self, x):
    branch3x3 = self.branch3x3(x)

    branch3x3dbl = self.branch3x3dbl_1(x)
    branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
    branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

    branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

    outputs = [branch3x3, branch3x3dbl, branch_pool]
    return torch.cat(outputs, 1)


class InceptionC(nn.Module):

  def __init__(self, in_channels, channels_7x7):
    super(InceptionC, self).__init__()
    self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

    c7 = channels_7x7
    self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
    self.branch7x7_2 = BasicConv2d(
      c7, c7, kernel_size=(
        1, 7), padding=(
        0, 3))
    self.branch7x7_3 = BasicConv2d(
      c7, 192, kernel_size=(
        7, 1), padding=(
        3, 0))

    self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
    self.branch7x7dbl_2 = BasicConv2d(
      c7, c7, kernel_size=(
        7, 1), padding=(
        3, 0))
    self.branch7x7dbl_3 = BasicConv2d(
      c7, c7, kernel_size=(
        1, 7), padding=(
        0, 3))
    self.branch7x7dbl_4 = BasicConv2d(
      c7, c7, kernel_size=(
        7, 1), padding=(
        3, 0))
    self.branch7x7dbl_5 = BasicConv2d(
      c7, 192, kernel_size=(
        1, 7), padding=(
        0, 3))

    self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch7x7 = self.branch7x7_1(x)
    branch7x7 = self.branch7x7_2(branch7x7)
    branch7x7 = self.branch7x7_3(branch7x7)

    branch7x7dbl = self.branch7x7dbl_1(x)
    branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
    branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
    branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
    branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

    branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = self.branch_pool(branch_pool)

    outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
    return torch.cat(outputs, 1)


class InceptionD(nn.Module):

  def __init__(self, in_channels):
    super(InceptionD, self).__init__()
    self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
    self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

    self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
    self.branch7x7x3_2 = BasicConv2d(
      192, 192, kernel_size=(
        1, 7), padding=(
        0, 3))
    self.branch7x7x3_3 = BasicConv2d(
      192, 192, kernel_size=(
        7, 1), padding=(
        3, 0))
    self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

  def forward(self, x):
    branch3x3 = self.branch3x3_1(x)
    branch3x3 = self.branch3x3_2(branch3x3)

    branch7x7x3 = self.branch7x7x3_1(x)
    branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
    branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
    branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

    branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
    outputs = [branch3x3, branch7x7x3, branch_pool]
    return torch.cat(outputs, 1)


class InceptionE(nn.Module):

  def __init__(self, in_channels):
    super(InceptionE, self).__init__()
    self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

    self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
    self.branch3x3_2a = BasicConv2d(
      384, 384, kernel_size=(
        1, 3), padding=(
        0, 1))
    self.branch3x3_2b = BasicConv2d(
      384, 384, kernel_size=(
        3, 1), padding=(
        1, 0))

    self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
    self.branch3x3dbl_3a = BasicConv2d(
      384, 384, kernel_size=(
        1, 3), padding=(
        0, 1))
    self.branch3x3dbl_3b = BasicConv2d(
      384, 384, kernel_size=(
        3, 1), padding=(
        1, 0))

    self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch3x3 = self.branch3x3_1(x)
    branch3x3 = [
      self.branch3x3_2a(branch3x3),
      self.branch3x3_2b(branch3x3),
    ]
    branch3x3 = torch.cat(branch3x3, 1)

    branch3x3dbl = self.branch3x3dbl_1(x)
    branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
    branch3x3dbl = [
      self.branch3x3dbl_3a(branch3x3dbl),
      self.branch3x3dbl_3b(branch3x3dbl),
    ]
    branch3x3dbl = torch.cat(branch3x3dbl, 1)

    branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = self.branch_pool(branch_pool)

    outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
    return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

  def __init__(self, in_channels):
    super(InceptionAux, self).__init__()
    self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
    self.conv1 = BasicConv2d(128, 768, kernel_size=5)
    # raw kernel_size=5. For use img 299 * 299.
    self.conv1.stddev = 0.01
    self.fc = nn.Linear(768, opt.classes)
    self.fc.stddev = 0.001

  def forward(self, x):
    # 17 x 17 x 768
    x = F.avg_pool2d(x, kernel_size=5, stride=3)
    # raw kernel_size=5, stride=3. For use img 299 * 299.
    # 5 x 5 x 768
    x = self.conv0(x)
    # 5 x 5 x 128
    x = self.conv1(x)
    # 1 x 1 x 768
    x = x.view(x.size(0), -1)
    # 768
    x = self.fc(x)
    # 1000
    return x


class BasicConv2d(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(BasicConv2d, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return F.relu(x, inplace=True)


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
  model = inception_v3().cuda()

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
    torch.save(model, f"{opt.outf}/GoogLeNet_epoch_{epoch + 1}.pth")
  print(f"Model save to '{opt.outf}'.")


def test():
  if torch.cuda.is_available():
    model = torch.load(f'{opt.outf}/GoogLeNet_epoch_{opt.niter}.pth')
  else:
    model = torch.load(f'{opt.outf}/GoogLeNet_epoch_{opt.niter}.pth', map_location="cpu")
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
