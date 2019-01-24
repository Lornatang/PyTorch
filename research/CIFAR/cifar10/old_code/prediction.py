"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: prediction.py
# time: 2018/8/14 09:35
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
parser.add_argument('--path', type=str, default='../../data/CIFAR/cifar10/',
                    help="""image dir path default: '../../data/CIFAR/cifar10/'.""")
parser.add_argument('--batch_size', type=int, default=1,
                    help="""Batch_size default:1.""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""num classes. Default: 10.""")
parser.add_argument('--model_path', type=str, default='../../../models/pytorch/CIFAR/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='cifar10.pth',
                    help="""Model name.""")

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(32),  # 将图像转化为128 * 128
    transforms.RandomCrop(24),  # 从图像中裁剪一个114 * 114的
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化
])

# Load data
test_datasets = torchvision.datasets.CIFAR10(root=args.path,
                                             download=True,
                                             transform=transform,
                                             train=False)


test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=args.batch_size,
                                          shuffle=True)


def main():
    print(f"test numbers: {len(test_datasets)}.")
    # Load model
    if torch.cuda.is_available():
        model = torch.load(args.model_path + args.model_name).to(device)
    else:
        model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model.eval()

    correct_prediction = 0.
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
        print(f"pred {predicted}")
        correct_prediction += (predicted == labels).sum().item()

    print(f"Acc: {(correct_prediction / total):4f}")


if __name__ == '__main__':
    main()
