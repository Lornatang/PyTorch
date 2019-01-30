import os

import torch
import torchvision
from torch import utils, optim, nn
from torchvision import transforms
from torchvision.utils import save_image

# first train run this code
from research.GAN.dcgan.net import Discriminator, Generator
# incremental training comments out that line of code.

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = '../../../../../data/GAN/basic'
NUM_EPOCHS = 20
BATCH_SIZE = 50
LEARNING_RATE = 1e-4
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

# CIFAR10 train_dataset
train_dataset = torchvision.datasets.ImageFolder(root=WORK_DIR + '/' + 'train',
                                                 transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

# first train run this line
D = Discriminator().to(device)
G = Generator().to(device)
# load model
# if torch.cuda.is_available():
#     D = torch.load(MODEL_PATH + 'D.pth').to(device)
#     G = torch.load(MODEL_PATH + 'G.pth').to(device)
# else:
#     D = torch.load(MODEL_PATH + 'D.pth', map_location='cpu')
#     G = torch.load(MODEL_PATH + 'G.pth', map_location='cpu')

# Binary cross entropy loss and optimizer
criterion = nn.BCEWithLogitsLoss().to(device)
d_optimizer = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)
g_optimizer = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Start training
def main():
    step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        for images, _ in train_loader:
            # D.zero_grad()
            # images = images.reshape(BATCH_SIZE, -1).to(device)

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
            real_score = outputs

            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            z = torch.randn(BATCH_SIZE, NOISE, 1, 1).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # Backprop and optimize
            d_loss = d_loss_real + d_loss_fake
            reset_grad()
            d_loss.backward()
            d_optimizer.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = torch.randn(BATCH_SIZE, NOISE, 1, 1).to(device)
            fake_images = G(z)
            outputs = D(fake_images)

            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)

            # Backprop and optimize
            reset_grad()
            g_loss.backward()
            g_optimizer.step()

            step += 1

            if step % 10 == 0:
                print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(train_dataset)}], "
                      f"d_loss: {d_loss:.8f}, "
                      f"g_loss: {g_loss:.8f}, "
                      f"D(x): {real_score.mean():.4f}, "
                      f"D(G(z)): {fake_score.mean():.4f}.")

            if epoch % 1 == 0:
                images = images.reshape(images.size(0), 1, 32, 32)
                save_image(images, WORK_DIR + '/' + 'gen' + '/' + 'real' + '.jpg')
                fake_images = fake_images.reshape(fake_images.size(0), 1, 32, 32)
                save_image(fake_images, WORK_DIR + '/' + 'gen' + '/' + str(epoch) + '.jpg')

        # Save the model checkpoint
        torch.save(D, MODEL_PATH + MODEL_D)
        torch.save(G, MODEL_PATH + MODEL_G)
    print(f"Model save to '{MODEL_PATH}'!")


if __name__ == '__main__':
    main()
