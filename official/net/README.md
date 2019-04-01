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

## 2.3 MNIST/Fashion-MNIST
To learn more for MNIST: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)!

To learn more for Fashion-MNIST: [https://github.com/rois-codh/kmnist](https://github.com/rois-codh/kmnist)!

### 2.3.1 Download
Collection of pictures: 

**MNIST**

- [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set images (9912422 bytes) 

- [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz):  training set labels (28881 bytes) 

- [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz):   test set images (1648877 bytes) 

- [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz):   test set labels (4542 bytes)

**Fashion-MNIST**

- [train-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz):  training set images (9912422 bytes) 
  
- [train-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz):  training set labels (28881 bytes) 

- [t10k-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz):   test set images (1648877 bytes) 

- [t10k-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz):   test set labels (4542 bytes)
    

## 2.4 CIFAR-10/CIFAR-100

### 2.4.1 Introduction
CIFAR-10:

- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

- The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

CIFAR-100:

- This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

### 2.4.2 Download
Collection of pictures: 

- [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

- [cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

*To learn more: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)*

## 2.5 Fruits 360 dataset: A dataset of images containing fruits

### 2.5.1 Content
The following fruits are included: Apples (different varieties: Golden, Red Yellow, Granny Smith, Red, Red Delicious), Apricot, Avocado, Avocado ripe, Banana (Yellow, Red, Lady Finger), Cactus fruit, Cantaloupe (2 varieties), Carambula, Cherry (different varieties, Rainier), Cherry Wax (Yellow, Red, Black), Chestnut, Clementine, Cocos, Dates, Granadilla, Grape (Blue, Pink, White (different varieties)), Grapefruit (Pink, White), Guava, Hazelnut, Huckleberry, Kiwi, Kaki, Kumsquats, Lemon (normal, Meyer), Lime, Lychee, Mandarine, Mango, Mangostan, Maracuja, Melon Piel de Sapo, Mulberry, Nectarine, Orange, Papaya, Passion fruit, Peach (different varieties), Pepino, Pear (different varieties, Abate, Kaiser, Monster, Williams), Physalis (normal, with Husk), Pineapple (normal, Mini), Pitahaya Red, Plums (different varieties), Pomegranate, Pomelo Sweetie, Quince, Rambutan, Raspberry, Redcurrant, Salak, Strawberry (normal, Wedge), Tamarillo, Tangelo, Tomato (different varieties, Maroon, Cherry Red), Walnut.

### 2.5.2 Dataset properties
Total number of images: 65429.

Training set size: 48905 images (one fruit per image).

Test set size: 16421 images (one fruit per image).

Multi-fruits set size: 103 images (more than one fruit (or fruit class) per image)

Number of classes: 95 (fruits).

Image size: 100x100 pixels.

Filename format: image_index_100.jpg (e.g. 32_100.jpg) or r_image_index_100.jpg (e.g. r_32_100.jpg) or r2_image_index_100.jpg or r3_image_index_100.jpg. "r" stands for rotated fruit. "r2" means that the fruit was rotated around the 3rd axis. "100" comes from image size (100x100 pixels).

Different varieties of the same fruit (apple for instance) are stored as belonging to different classes.

### 2.5.3 How we made it:

- Fruits were planted in the shaft of a low speed motor (3 rpm) and a short movie of 20 seconds was recorded.

- A Logitech C920 camera was used for filming the fruits. This is one of the best webcams available.

- Behind the fruits we placed a white sheet of paper as background.

- However due to the variations in the lighting conditions, the background was not uniform and we wrote a dedicated algorithm which extract the fruit from the background. This algorithm is of flood fill type: we start from each edge of the image and we mark all pixels there, then we mark all pixels found in the neighborhood of the already marked pixels for which the distance between colors is less than a prescribed value. We repeat the previous step until no more pixels can be marked.

- All marked pixels are considered as being background (which is then filled with white) and the rest of pixels are considered as belonging to the object.

- The maximum value for the distance between 2 neighbor pixels is a parameter of the algorithm and is set (by trial and error) for each movie.

- Pictures from the test-multiple_fruits folder were taken with a Nexus 5X phone.

### 2.5.4 How to cite:

- Horea Muresan, Mihai Oltean, Fruit recognition from images using deep learning, Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.

- The paper introduces the dataset and an implementation of a Neural Network trained to recognized the fruits in the dataset.

### 2.5.5 Download
Collection of pictures: [fruits.zip](https://www.kaggle.com/moltean/fruits/downloads/fruits.zip/44)

# Result

|Model|DataSet|Top-1|Top-5|Epoch|
|:----|:-----:|:---:|:---:|:----|
|LeNet-5|MNIST|97.21|99.97|25
|LeNet-5|Fashion-MNIST|85.77|99.77|25
|LeNet-5|CIFAR-10|55.01|94.98|25
|LeNet-5|CIFAR-100|19.00|46.54|25
|LeNet-5|Caltech-101|27.198|43.794|25
|LeNet-5|Caltech-256| | |25
|LeNet-5|Fruits|87.065|99.245|25


*Tips:
 
- **" _"** indicates that it has not been tested yet and will be tested later.*
- **"[+]"** Represents fine tuning of the model. 
- Caltech_101/Caltech_256 was trained with 100% training and 100% testing! But that doesn't matter!
