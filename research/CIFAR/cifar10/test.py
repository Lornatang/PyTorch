import torch
import torchvision
from torchvision import transforms

from research.CIFAR.cifar10.net import GoogLeNet

BATCH_SIZE = 1

MODEL_PATH = '../../../../models/pytorch/CIFAR/'
MODEL_NAME = '10.pth'


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize(96),  # 将图像转化为224 * 224
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])


# Load data
test_dataset = torchvision.datasets.ImageFolder(root='test',
                                                transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)


def label_name(index):
    label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return label[index]


def main():
    print(f"Image numbers:{len(test_dataset)}")

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

        print(f"Classifier is {label_name(predicted)}.")
        print(predicted)


if __name__ == '__main__':
    main()
