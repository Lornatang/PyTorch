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
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='inputs batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
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

if opt.dataset in ['imagenet', 'folder', 'lfw']:
  # folder dataset
  dataset = dset.ImageFolder(root=opt.dataroot,
                             transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.CenterCrop(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
  nc = 3
elif opt.dataset == 'lsun':
  dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                      transform=transforms.Compose([
                        transforms.Resize(opt.imageSize),
                        transforms.CenterCrop(opt.imageSize),
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
elif opt.dataset == 'cifar-100':
  dataset = dset.CIFAR100(root=opt.dataroot, download=True,
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

elif opt.dataset == 'fake':
  dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                          transform=transforms.ToTensor())
  nc = 3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


class Generator(nn.Module):
  def __init__(self, ngpus):
    super(Generator, self).__init__()
    self.ngpu = ngpus

    self.main = nn.Sequential(
      nn.Linear(nz, 128),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(128, 256),
      nn.BatchNorm1d(256, 0.8),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(256, 512),
      nn.BatchNorm1d(512, 0.8),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(512, 1024),
      nn.BatchNorm1d(1024, 0.8),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(1024, opt.imageSize * opt.imageSize),
      nn.Tanh()
    )

  def forward(self, inputs):
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs


class Discriminator(nn.Module):
  def __init__(self, ngpus):
    super(Discriminator, self).__init__()
    self.ngpu = ngpus

    self.main = nn.Sequential(
      nn.Linear(opt.imageSize * opt.imageSize, 512),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(256, nc),
      nn.Sigmoid(),
    )

  def forward(self, inputs):
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)

netG = Generator(ngpu).to(device)
netG.apply(weights_init)


if opt.netD and opt.netG != '':
  if torch.cuda.is_available():
    netD = torch.load(opt.netD)
    netG = torch.load(opt.netG)
  else:
    netD = torch.load(opt.netD, map_location='cpu')
    netG = torch.load(opt.netG, map_location='cpu')

print(netD)
print(netG)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def main():
  for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      # train with real
      netD.zero_grad()
      real_cpu = data[0].to(device)
      batch_size = real_cpu.size(0)
      label = torch.full((batch_size,), real_label, device=device)

      output = netD(real_cpu)
      errD_real = criterion(output, label)
      errD_real.backward()
      D_x = output.mean().item()

      # train with fake
      noise = torch.randn(batch_size, nz, 1, 1, device=device)
      fake = netG(noise)
      label.fill_(fake_label)
      output = netD(fake.detach())
      errD_fake = criterion(output, label)
      errD_fake.backward()
      D_G_z1 = output.mean().item()
      errD = errD_real + errD_fake
      optimizerD.step()

      ############################
      # (2) Update G network: maximize log(D(G(z)))
      ###########################
      netG.zero_grad()
      label.fill_(real_label)  # fake labels are real for generator cost
      output = netD(fake)
      errG = criterion(output, label)
      errG.backward()
      D_G_z2 = output.mean().item()
      optimizerG.step()

      # Clip weights of discriminator
      for p in discriminator.parameters():
        p.data.clamp_(-opt.clip_value, opt.clip_value)

      # Train the generator every n_critic iterations
      if i % opt.n_critic == 0:
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z)
        # Adversarial loss
        loss_G = -torch.mean(discriminator(gen_imgs))

        loss_G.backward()
        optimizer_G.step()

      print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (epoch + 1, opt.niter, i, len(dataloader),
               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
      if i % 100 == 0:
        vutils.save_image(real_cpu,
                          '%s/real_samples.png' % opt.outf,
                          normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(),
                          '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch + 1),
                          normalize=True)

    # do checkpointing
    torch.save(netG, '%s/netG_epoch_%d.pth' % (opt.outf, epoch + 1))
    torch.save(netD, '%s/netD_epoch_%d.pth' % (opt.outf, epoch + 1))


if __name__ == '__main__':
  main()
