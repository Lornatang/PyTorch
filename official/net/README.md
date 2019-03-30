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

# Result

|    Model       |  DataSet   |     Accuracy   |Epoch |
|:--------------:|:----------:|:--------------:|:----:|
|LeNet-5         |Caltech-101 |35%$\pm$5%      |25
|LeNet-5         |Caltech-256 |_               |25
|AlexNet[+]      |Caltech-101 |71%$\pm$3%      |25
|AlexNet[+]      |Caltech-256 |_               |25
|VGG-11[+]       |Caltech-101 |_               |25
|VGG-11_bn[+]    |Caltech-101 |96%$\pm$%2      |25
|SqueezeNet[+]   |Caltech-101 |25%$\pm$%3      |25
|DenseNet-121[+] |Caltech-101 |97%$\pm$1%      |25
|ResNet-18[+]    |Caltech-101 |99%$\pm$1%      |25

*Tips:
 
- **" _"** indicates that it has not been tested yet and will be tested later.*
- **"[+]"** Represents fine tuning of the model. 
