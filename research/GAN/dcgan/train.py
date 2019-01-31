import os

import torch
import torchvision
from torch import utils, optim, nn
from torchvision import transforms
from torchvision.utils import save_image

# first train run this code
# from research.GAN.dcgan.net import Discriminator, Generator
# incremental training comments out that line of code.

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '../../../../../data/GAN/basic'
NUM_EPOCHS = 300
BATCH_SIZE = 100
LEARNING_RATE = 2e-4
OPTIM_BETAS = (0.5, 0.999)

NOISE = 100

MODEL_PATH = '../../../../models/pytorch/GAN/dcgan/'
MODEL_D = 'D.pth'
MODEL_G = 'G.pth'

# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(WORK_DIR + '/' + 'gen'):
    os.makedirs(WORK_DIR + '/' + 'gen')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

to_pil_image = transforms.ToPILImage()

# mnist train_dataset
train_dataset = torchvision.datasets.ImageFolder(root=WORK_DIR + '/' + 'train',
                                                 transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

# first train run this line
# D = Discriminator().to(device)
# G = Generator().to(device)
# load model
if torch.cuda.is_available():
    D = torch.load(MODEL_PATH + 'D.pth').to(device)
    G = torch.load(MODEL_PATH + 'G.pth').to(device)
else:
    D = torch.load(MODEL_PATH + 'D.pth', map_location='cpu')
    G = torch.load(MODEL_PATH + 'G.pth', map_location='cpu')

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss().to(device)
d_optimizer = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)
g_optimizer = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)


# Start training
def main():
    step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        for images, _ in train_loader:
            D.zero_grad()

            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(BATCH_SIZE,).to(device)
            fake_labels = torch.zeros(BATCH_SIZE,).to(device)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()
            real_score = outputs.mean().item()

            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            noise = torch.randn(BATCH_SIZE, NOISE, 1, 1).to(device)
            fake = G(noise)
            outputs = D(fake.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            fake_score_z1 = outputs.mean().item()

            # Backprop and optimize
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            G.zero_grad()
            outputs = D(fake)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            fake_score_z2 = outputs.mean().item()
            g_optimizer.step()

            step += 1

            # func (item): Tensor turns into an int
            print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(train_dataset)}], "
                  f"d_loss: {d_loss.item():.4f}, "
                  f"g_loss: {g_loss.item():.4f}, "
                  f"D(x): {real_score:.4f}, "
                  f"D(G(z)): {fake_score_z1:.4f} / {fake_score_z2:.4f}.")

            images = images.reshape(images.size(0), 1, 32, 32)
            save_image(images, WORK_DIR + '/' + 'gen' + '/' + 'real' + '.jpg')
            fake_images = fake.reshape(fake.size(0), 1, 32, 32)
            save_image(fake_images, WORK_DIR + '/' + 'gen' + '/' + str(epoch) + '.jpg')

        # Save the model checkpoint
        torch.save(D, MODEL_PATH + MODEL_D)
        torch.save(G, MODEL_PATH + MODEL_G)
    print(f"Model save to '{MODEL_PATH}'!")


if __name__ == '__main__':
    main()
