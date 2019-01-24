import torch
import torchvision
from torchvision import transforms

BATCH_SIZE = 1

MODEL_DIR = '../../../models/pytorch/CALTECH/'
MODEL_NAME = '102.pth'


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize(224),  # 将图像转化为224 * 224
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])


# Load data
test_datasets = torchvision.datasets.ImageFolder(root='/Users/mac/program/lorna/pytorch-project/CALTECH/4/test',
                                                 transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)


def main():
    print(f"Image numbers:{len(test_datasets)}")

    # Load model
    if torch.cuda.is_available():
        model = torch.load(MODEL_DIR + MODEL_NAME).to(device)
    else:
        model = torch.load(MODEL_DIR + MODEL_NAME, map_location='cpu')

    model.eval()

    for images, labels in test_loader:
        # to GPU
        images = images.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)

        if predicted == 0:
            print(f"Classifier is airplane.")
        if predicted == 1:
            print(f"Classifier is car.")
        if predicted == 2:
            print(f"Classifier is face.")
        if predicted == 3:
            print(f"Classifier is motorbike.")


if __name__ == '__main__':
    main()
