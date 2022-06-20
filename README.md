# LeViT-TensorRT
The goal of this project is to accelerate the deployment of __LeViT__ networks using __TensorRT__, including test results for FP16 and Int8.
## Overview
As the current visual Transformer with the fastest inference speed, LeViT is significantly better than existing CNNs and visual Transformers in terms of speed/accuracy trade-offs, such as ViT, DeiT, etc., and the top-1 accuracy rate reaches 80%. On CPU, Lower LeViT 3.3x faster than EfficientNet, 2x faster on gpu and nearly 10x faster on arm. It has high application value in scenarios with limited computing power.
## Setup 
LeViT original repo：[_LeViT_](https://github.com/facebookresearch/LeViT) 
1.Data preparation
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```
2.environments
```
nvidia-docker pull registry.cn-hangzhou.aliyuncs.com/trt2022/dev
```
Use this image as a baseline for your TensorRT environment.
```
bash
$ git clone https://github.com/JQZhai/LeViT-TensorRT.git
```
Model download address used in this project：[_LeViT-128S_](https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth) 

