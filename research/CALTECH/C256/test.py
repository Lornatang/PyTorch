import torch
import torchvision
from torchvision import transforms

BATCH_SIZE = 1

MODEL_PATH = '../../../models/pytorch/CALTECH/'
MODEL_NAME = 'C256.pth'


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# Load data
test_datasets = torchvision.datasets.ImageFolder(root='test',
                                                 transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)


def main():
    print(f"Image numbers:{len(test_datasets)}")

    # Load model
    if torch.cuda.is_available():
        model = torch.load(MODEL_PATH + MODEL_NAME).to(device)
    else:
        model = torch.load(MODEL_PATH + MODEL_NAME, map_location='cpu')

    model.eval()

    for images, labels in test_loader:
        # to GPU
        images = images.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)


if __name__ == '__main__':
    main()
