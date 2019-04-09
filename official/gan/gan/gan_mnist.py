import argparse
import os
import random

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='inputs batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
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

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
  print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = dset.MNIST(root=opt.dataroot, download=True,
                     transform=transforms.Compose([
                       transforms.Resize(opt.imageSize),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5]),
                     ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
nc = 1


class Generator(nn.Module):
  def __init__(self, gpus):
    super(Generator, self).__init__()
    self.ngpu = gpus
    self.main = nn.Sequential(
      nn.Linear(nz, 128),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(128, 256),
      nn.BatchNorm1d(256),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(256, 512),
      nn.BatchNorm1d(512),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(512, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(1024, nc * opt.imageSize * opt.imageSize),
      nn.Tanh()
    )

  def forward(self, inputs):
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)

    return outputs.view(outputs.size(0), nc, opt.imageSize, opt.imageSize)


class Discriminator(nn.Module):
  def __init__(self, gpus):
    super(Discriminator, self).__init__()
    self.ngpu = gpus
    self.main = nn.Sequential(
      nn.Linear(nc * opt.imageSize * opt.imageSize, 512),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, inputs):
    inputs = inputs.view(inputs.size(0), -1)
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)

    return outputs


criterion = nn.BCELoss()

netD = Discriminator(ngpu)
netG = Generator(ngpu)

if opt.cuda:
  criterion.to(device)
  netD.to(device)
  netG.to(device)

if opt.netD and opt.netG != '':
  if torch.cuda.is_available():
    netD = torch.load(opt.netD)
    netG = torch.load(opt.netG)
  else:
    netD = torch.load(opt.netD, map_location='cpu')
    netG = torch.load(opt.netG, map_location='cpu')

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor


def main():
  for epoch in range(opt.niter):
    for i, (real_imgs, _) in enumerate(dataloader):
      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################

      # Adversarial ground truths
      real_label = torch.full((real_imgs.size(0), 1), 1, device=device)
      fake_label = torch.full((real_imgs.size(0), 1), 0, device=device)

      real_imgs = real_imgs.to(device)

      # -----------------
      #  Train Generator
      # -----------------

      netG.zero_grad()

      # Sample noise as generator input
      noise = torch.randn(real_imgs.size(0), nz)

      # Generate a batch of images
      fake_imgs = netG(noise)

      # Loss measures generator's ability to fool the discriminator
      errG = criterion(netD(fake_imgs), real_label)

      errG.backward()
      optimizerG.step()

      # ---------------------
      #  Train Discriminator
      # ---------------------

      netD.zero_grad()

      # Measure discriminator's ability to classify real from generated samples
      real_loss = criterion(netD(real_imgs), real_label)
      fake_loss = criterion(netD(fake_imgs.detach()), fake_label)
      errD = (real_loss + fake_loss) / 2

      errD.backward()
      optimizerD.step()

      print(f'[{epoch + 1}/{opt.niter}][{i}/{len(dataloader)}] '
            f'Loss_D: {errD.item():.4f} '
            f'Loss_G: {errG.item():.4f}.')

      if i % 100 == 0:
        vutils.save_image(real_imgs,
                          f'{opt.outf}/real_samples.png',
                          normalize=True)
        vutils.save_image(netG(noise).detach(),
                          f'{opt.outf}/fake_samples_epoch_{epoch + 1}.png',
                          normalize=True)

    # do checkpointing
    torch.save(netG, f'{opt.outf}/netG_epoch_{epoch + 1}.pth')
    torch.save(netD, f'{opt.outf}/netD_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
  main()
