import os

import torch
import torchvision
from torchvision import transforms

# from research.MNIST.mnist.net import Net

WORK_DIR = '../../../../../data/MNIST/mnist'
BATCH_SIZE = 1

MODEL_PATH = '../../../../models/pytorch/MNIST/'
MODEL_NAME = 'mnist.pth'


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tensor image transforms to PIL image
to_pil_image = transforms.ToPILImage()

label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# check file name is exist
for dir_index in range(0, 10):
    if not os.path.exists(WORK_DIR + '/' + 'test' + '/' + label[dir_index]):
        os.makedirs(WORK_DIR + '/' + 'test' + '/' + label[dir_index])

transform = transforms.ToTensor()

# Load data
test_dataset = torchvision.datasets.ImageFolder(root=WORK_DIR + '/' + 'test',
                                                transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)


def main():
    print(f"Image numbers:{len(test_dataset)}")

    # Load model
    if torch.cuda.is_available():
        model = torch.load(MODEL_PATH + MODEL_NAME).to(device)
    else:
        model = torch.load(MODEL_PATH + MODEL_NAME, map_location='cpu')

    index = 0
    for images, _ in test_loader:
        # to GPU
        images = images.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)

        img = to_pil_image(images[0])
        img.save(str(WORK_DIR + '/' + 'test' + '/' + label[predicted]) + '/' + str(index) + '.jpg')
        index += 1


if __name__ == '__main__':
    main()
