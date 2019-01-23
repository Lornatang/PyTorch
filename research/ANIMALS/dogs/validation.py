"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: validation.py
# time: 2018/8/14 09:43
# license: MIT
"""

import argparse
import os

import time
import torch
import torchvision
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--path', type=str, default='../../data/ANIMALS/dogs/',
                    help="""image dir path default: '../../data/ANIMALS/dogs/'.""")
parser.add_argument('--batch_size', type=int, default=1,
                    help="""Batch_size default:1.""")
parser.add_argument('--num_classes', type=int, default=120,
                    help="""num classes""")
parser.add_argument('--model_path', type=str, default='../../../models/pytorch/ANIMALS/dogs/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='dogs.pth',
                    help="""Model name.""")
args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(128),  # 将图像转化为128 * 128
    transforms.RandomCrop(114),  # 从图像中裁剪一个114 * 114的
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])
# Load data
val_datasets = torchvision.datasets.ImageFolder(root=args.path + 'val/',
                                                transform=transform)


val_loader = torch.utils.data.DataLoader(dataset=val_datasets,
                                         batch_size=args.batch_size,
                                         shuffle=True)
# train_datasets zip
item = val_datasets.class_to_idx


def val():
    # Load model
    if torch.cuda.is_available():
        model = torch.load(args.model_path + args.model_name).to(device)
    else:
        model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model.eval()
    start = time.time()
    for i, (images, _) in enumerate(val_loader):
        # to GPU
        images = images.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)
        di = {v: k for k, v in item.items()}

        pred = di[int(predicted[0])]

        file = str(val_datasets.imgs[i])[2:-5]

        print(f"{i+1}.({file}) is {pred}!")

    print(f"\nTime: {time.time() - start:.1f} sec!")


if __name__ == '__main__':
    val()
