# LeViT-TensorRT
The goal of this project is to accelerate the deployment of __LeViT__ networks using __TensorRT__, including test results for FP16 and Int8.
## Overview
As the current visual Transformer with the fastest inference speed, LeViT is significantly better than existing CNNs and visual Transformers in terms of speed/accuracy trade-offs, such as ViT, DeiT, etc., and the top-1 accuracy rate reaches 80%. On CPU, Lower LeViT 3.3x faster than EfficientNet, 2x faster on gpu and nearly 10x faster on arm. It has high application value in scenarios with limited computing power.
## Setup 
LeViT original repo：[_LeViT_](https://github.com/facebookresearch/LeViT) \
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
2.environments\
As for X86 platform use this image as a baseline for your TensorRT environment.
```
nvidia-docker pull registry.cn-hangzhou.aliyuncs.com/trt2022/dev
```
```bash
$ git clone https://github.com/JQZhai/LeViT-TensorRT.git
```
```bash
$ cd LeViT-TensorRT
```
```bash
$ pip install -r requirment.txt
```
3.Model Zoo\
[_LeViT-128S_](https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth)
[_LeViT-128_](https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth)
[_LeViT-192_](https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth)
[_LeViT-256_](https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth)
[_LeViT-384_](https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth) 

## Export to ONNX and Build TensorRT Engine
1.Download the __Levit__ pretrained model from Model Zoo.Evaluate the accuracy of the Pytorch pretrained model.
```bash
$ python main.py --eval --model LeViT_256 --data-path /path/to/imagenet
```
2.`export.py` exports a pytorch model to onnx format.
```bash
$ python export.py --model <model name> --finetune path/to/pth.file
```
3. Build the TensorRT engine using `trtexec`.  
```bash
$ trtexec --onnx=path/to/onnx.file --buildOnly  --saveEngine=path/to/engine.file --workspace=4096
```  
For fp16 mode, fp16 cannot store very large and very small numbers like fp32. So we let some nodes fall back to fp32 mode to ensure the correctness of the final output.Keep the same input as the onnx format model, and use the output in onnx fp32 mode as the standard to calculate the error.
```bash
$ polygraphy debug precision ../LeViT-128S.onnx \
-v --fp16 --workspace 28G --no-remove-intermediate --log-file ./log_file.json \
--trt-min-shapes 'input_0:[1,3,224,224]' \
--trt-opt-shapes 'input_0:[1,3,224,224]' \
--trt-max-shapes 'input_0:[1,3,224,224]' \
-p fp32 --mode bisect --dir forward --show-output \
--artifacts ./polygraphy_debug.engine --art-dir ./art-dir \
--check \
polygraphy run polygraphy_debug.engine \
--trt --load-outputs onnx_res.json --load-inputs onnx_input.json \
--abs 1e-2 -v --rel 1e-2
```  
We can use the trtexec to test the throughput of the TensorRT engine.
```bash
$ trtexec --loadEngine=path/to/engine.file
``` 
