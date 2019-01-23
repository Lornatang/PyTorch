# 一个有趣的实例：给猫和狗的图像分类

<br>

有很多的图像数据集是专门用来给深度学习模型进行基准测试的，我在这篇文章中用到的数据集来自 Cat vs Dogs Kaggle competition，这份数据集包含了大量狗和猫的带有标签的图片。

<br>

## 和大多数 Kaggle 比赛一样，这份数据集也包含两个文件夹：

- 训练文件夹：它包含了 25000 张猫和狗的图片，每张图片都含有标签，这个标签是作为文件名的一部分。我们将用这个文件夹来训练和评估我们的模型。

- 测试文件夹：它包含了 12500 张图片，每张图片都以数字来命名。对于这份数据集中的每幅图片来说，我们的模型都要预测这张图片上是狗还是猫（1= 狗，0= 猫）。事实上，这些数据也被 Kaggle 用来对模型进行打分，然后在排行榜上排名。

文件目录结构如下:

	  └── train
        └── dog
            └── dog.1.jpg
            ...
            └── dog.n.jpg
        └── cat
            └── cat.1.jpg
            ...
            └── cat.n.jpg
     └── test
        └── dog
            └── dog.1.jpg
            ...
            └── dog.n.jpg
        └── cat
            └── cat.1.jpg
            ...
            └── cat.n.jpg
    └── val
        └── unknown
            └── dog.1.jpg
            ...
            └── cat.n.jpg
            
[数据下载点这里](https://www.kaggle.com/c/dogs-vs-cats)

### 使用的是Pytorch框架

#### 使用说明

- train: 

`python train.py`

- test: 

`python test.py`

- val: 

`python validation.py`

Acc: 0.925.

## LINCENSE: MIT
