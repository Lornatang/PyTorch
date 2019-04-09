import argparse
import os
import random

from torch import autograd
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
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
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
                         transforms.Normalize((0.5,), (0.5,)),
                         ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
nc = 1

# Loss weight for gradient penalty
lambda_gp = 10


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
    def __init__(self, ngpus):
        super(Discriminator, self).__init__()
        self.ngpu = ngpus

        self.main = nn.Sequential(
            nn.Linear(nc * opt.imageSize * opt.imageSize, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        if inputs.is_cuda and self.ngpu > 1:
            outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            outputs = self.main(inputs)
        return outputs


netD = Discriminator(ngpu)
netG = Generator(ngpu)

if opt.cuda:
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
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor


def compute_gradient_penalty(net, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.randn(real_samples.size(0), 1, 1, 1)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = net(interpolates)
    fake = torch.full((real_samples.size(0), 1), 1, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penaltys = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penaltys


def main():
    for epoch in range(opt.niter):
        for i, (real_imgs, _) in enumerate(dataloader):

            # configure input
            real_imgs = real_imgs.to(device)

            # -----------------
            #  Train Discriminator
            # -----------------

            netD.zero_grad()

            # Sample noise as generator input
            noise = torch.randn(real_imgs.size(0), nz)

            # Generate a batch of images
            fake_imgs = netG(noise)

            # Real images
            real_validity = netD(real_imgs)
            # Fake images
            fake_validity = netD(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data)

            # Loss measures generator's ability to fool the discriminator
            errD = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            errD.backward()
            optimizerD.step()

            optimizerG.zero_grad()

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:

                # ---------------------
                #  Train Generator
                # ---------------------

                # Generate a batch of images
                fake_imgs = netG(noise)
                # Adversarial loss
                errG = -torch.mean(netD(fake_imgs))

                errG.backward()
                optimizerG.step()

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
