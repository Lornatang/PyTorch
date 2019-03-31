# 1. Introduction

1. **[AlexNet](https://www.jianshu.com/p/c5510449e8a6)**

2. **DenseNet**

3. **[Inception(GoogleNet_v3)](https://www.jianshu.com/p/79ef7ed956ac)**

4. **[LeNet](https://www.jianshu.com/p/05e562a8ed19)**

5. **ResNet**

6. **SqueezeNet**

7. **[VGG](https://www.jianshu.com/p/a52991ab86e0)**


# 2. Dataset

## 2.1 Caltech_101/Caltech_256

### 2.1.1 Description
Pictures of objects belonging to 101 categories. About 40 to 800 images per category. 
Most categories have about 50 images. Collected in September 2003 by Fei-Fei Li, Marco Andreetto, and Marc 'Aurelio Ranzato.  The size of each image is roughly 300 x 200 pixels.
We have carefully clicked outlines of each object in these pictures, these are included under the 'Annotations.tar'. 
There is also a matlab script to view the annotaitons, 'show_annotations.m'.

### 2.1.2 How to use the dataset
If you are using the Caltech 101 dataset for testing your recognition algorithm you should try and make your results comparable to the results of others. We suggest training and testing on fixed number of pictures and repeating the experiment with different random selections of pictures in order to obtain error bars. Popular number of training images: 1, 3, 5, 10, 15, 20, 30. Popular numbers of testing images: 20, 30. See also the discussion below.
When you report your results please keep track of which images you used and which were misclassified. We will soon publish a more detailed experimental protocol that allows you to report those details. See the Discussion section for more details.

### 2.1.3 Download
Collection of pictures: [101_ObjectCategories.tar.gz (131Mbytes)](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz) or [256_ObjectCategories.tar](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar).

Outlines of the objects in the pictures: [1] [Annotations.tar](http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar) [2] [show_annotation.m](http://www.vision.caltech.edu/Image_Datasets/Caltech101/show_annotation.m)

## 2.2 NotMNIST

### 2.2.1 Context
This dataset was created by Yaroslav Bulatov by taking some publicly available fonts and extracting glyphs from them to make a dataset similar to MNIST. There are 10 classes, with letters A-J.

### 2.2.2 Content
A set of training and test images of letters from A to J on various typefaces. The images size is 28x28 pixels.

### 2.2.3 Acknowledgements
The dataset can be found on Tensorflow github page as well as on the blog from Yaroslav, here.

### 2.2.4 Download
Collection of pictures: [notMNIST_small.tar.gz](https://www.kaggle.com/lubaroli/notmnist/downloads/notMNIST_small.tar.gz/1)

### 2.3 MNIST
To learn more :[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)!

### 2.3.1 Download
Collection of pictures: 

- [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set images (9912422 bytes) 

- [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz):  training set labels (28881 bytes) 

- [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz):   test set images (1648877 bytes) 

- [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz):   test set labels (4542 bytes)

## 2.4 CIFAR-10/CIFAR-100

### 2.4.1 Introduction
CIFAR-10:

- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

- The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 
![CIFAR-10](https://github.com/Lornatang/pytorch/blob/master/img/CIFAR-10.png)

CIFAR-100:

- This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

### 2.4.2 Download
Collection of pictures: 

- [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

- [cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

*To learn more: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)*









# Result

|    Model       |  DataSet   |    Accuracy    |Epoch |
|:--------------:|:----------:|:--------------:|:----:|
|LeNet-5         |MNIST       |98%$\pm$0.5%    |25
|LeNet-5         |CIFAR-10    |56%$\pm$1.4%    |25
|LeNet-5         |NOT-MNIST   |89%$\pm$2%      |25
|LeNet-5[+]      |Caltech-101 |35%$\pm$5%      |25
|LeNet-5[+]      |Caltech-256 |_               |25
|AlexNet[+]      |MNIST       |_               |25
|AlexNet[+]      |CIFAR-10    |_               |25
|AlexNet         |Caltech-101 |71%$\pm$3%      |25
|AlexNet         |Caltech-256 |_               |25
|VGG-11          |Caltech-101 |93%$\pm$1%      |25
|VGG-11_bn       |Caltech-101 |96%$\pm$2%      |25
|SqueezeNet      |Caltech-101 |25%$\pm$3%      |25
|DenseNet-121    |Caltech-101 |97%$\pm$1%      |25
|ResNet-18       |Caltech-101 |99%$\pm$0.5%    |25

*Tips:
 
- **" _"** indicates that it has not been tested yet and will be tested later.*
- **"[+]"** Represents fine tuning of the model. 
- Caltech_101 was trained with 100% training and 100% testing! But that doesn't matter!
