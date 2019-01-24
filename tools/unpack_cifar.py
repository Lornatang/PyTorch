import os
import pickle

import cv2
import numpy as np

# source directory
CIFAR10_DIR = '/Users/mac/data/CIFAR/cifar10'
CIFAR100_DIR = '/Users/mac/data/CIFAR/cifar100'

# extract cifar img in here.
CIFAR10_TRAIN_DIR = CIFAR10_DIR + '/' + 'train'
CIFAR10_VAL_DIR = CIFAR10_DIR + '/' + 'val'
CIFAR100_TRAIN_DIR = CIFAR100_DIR + '/' + 'trains'
CIFAR100_VAL_DIR = CIFAR100_DIR + '/' + 'val'

dir_list = [CIFAR10_TRAIN_DIR, CIFAR100_TRAIN_DIR, CIFAR10_VAL_DIR, CIFAR100_VAL_DIR]

# check file name is exist
for dir_index in range(0, 4):
    if not os.path.exists(dir_list[dir_index]):
        os.makedirs(dir_list[dir_index])


# extract the binaries, encoding must is 'bytes'!
def unpickle(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    return data


class CIFAR10(object):
    def __init__(self):
        # generate training data sets.
        for j in range(1, 6):
            # Read five files in turn.
            data_dir = CIFAR10_DIR + '/' + "data_batch_" + str(j)
            train_data = unpickle(data_dir)
            print(data_dir + " is loading...")

            for i in range(0, 10000):
                # binary files are converted to images.
                img = np.reshape(train_data[b'data'][i], (3, 32, 32))
                img = img.transpose(1, 2, 0)
                img_path = CIFAR10_TRAIN_DIR + '/' + str(train_data[b'labels'][i]) + '_' + str(
                    i + (j - 1) * 10000) + '.jpg'
                cv2.imwrite(img_path, img)
            print(data_dir + " loaded.")

        print("test_batch is loading...")

        # generate the validation data set.
        val_data = CIFAR10_DIR + '/' + 'test_batch'
        val_data = unpickle(val_data)
        for i in range(0, 10000):
            # binary files are converted to images
            img = np.reshape(val_data[b'data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            img_path = CIFAR10_VAL_DIR + '/' + str(val_data[b'labels'][i]) + '_' + str(i) + '.jpg'
            cv2.imwrite(img_path, img)
        print("test_batch loaded.")
        return


class CIFAR100(object):
    def __init__(self):
        # generate training data sets.

        data_dir = CIFAR100_DIR + '/' + 'train'
        train_data = unpickle(data_dir)
        print(data_dir + " is loading...")

        for i in range(0, 50000):
            # binary files are converted to images.
            img = np.reshape(train_data[b'data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            img_path = CIFAR100_TRAIN_DIR + '/' + str(train_data[b'fine_labels'][i]) + '_' + str(i) + '.jpg'
            cv2.imwrite(img_path, img)
        print(data_dir + " loaded.")

        print("test_batch is loading...")

        # generate the validation data set.
        val_data = CIFAR100_DIR + '/' + 'test'
        val_data = unpickle(val_data)
        for i in range(0, 10000):
            # binary files are converted to images
            img = np.reshape(val_data[b'data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            img_path = CIFAR100_VAL_DIR + '/' + str(val_data[b'fine_labels'][i]) + '_' + str(i) + '.jpg'
            cv2.imwrite(img_path, img)
        print("test_batch loaded.")
        return
