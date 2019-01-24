"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: resize.py
# time: 2018/8/18 22:16
# license: MIT
"""

import os.path
import glob
import cv2


def convert_size(file, out_dir, width, height):
    raw_img = cv2.imread(file)
    try:
        new_img = cv2.resize(raw_img, (width, height))
        cv2.imwrite(os.path.join(out_dir, os.path.basename(file)), new_img)
    except Exception as e:
        print(e)


for root in os.listdir(
        '../../data/CVPR/09/train/'):
    dirs = os.path.join(
        '../../data/CVPR/09/train/', root)
    print(f"{dirs} has done!")
    for img in glob.glob(dirs + '/*'):
        convert_size(img, dirs, 224, 224)


for root in os.listdir(
        '../../data/CVPR/09/val/'):
    dirs = os.path.join(
        '../../data/CVPR/09/val/', root)
    print(f"{dirs} has done!")
    for img in glob.glob(dirs + '/*'):
        convert_size(img, dirs, 224, 224)
