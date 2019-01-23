"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: prediction.py
# time: 2018/8/24 22:18
# license: MIT
"""

import argparse
import os

import torch
import torchvision
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--path', type=str, default='../../data/PASCAL_VOC/2005/',
                    help="""image dir path default: '../../data/PASCAL_VOC/2005/'.""")
parser.add_argument('--batch_size', type=int, default=128,
                    help="""Batch_size default:128.""")
parser.add_argument('--num_classes', type=int, default=6,
                    help="""num classes. Default: 6.""")
parser.add_argument('--model_path', type=str, default='../../../models/pytorch/PASCAL/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='2005.pth',
                    help="""Model name.""")

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(128),  # 将图像转化为800 * 800
    transforms.RandomCrop(114),  # 从图像中裁剪一个24 * 24的
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])


# Load data
test_datasets = torchvision.datasets.ImageFolder(root=args.path + 'val/',
                                                 transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=args.batch_size,
                                          shuffle=True)


def main():
    print(f"Test numbers:{len(test_datasets)}")

    # Load model
    if torch.cuda.is_available():
        model = torch.load(args.model_path + args.model_name).to(device)
    else:
        model = torch.load(args.model_path + args.model_name, map_location='cpu')

    model.eval()

    correct = 0.
    total = 0
    for images, labels in test_loader:
        # to GPU
        images = images.to(device)
        labels = labels.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)
        # val_loader total
        total += labels.size(0)
        # add correct
        correct += (predicted == labels).sum().item()

    print(f"Acc: {100 * correct / total:.4f}.")


if __name__ == '__main__':
    main()
