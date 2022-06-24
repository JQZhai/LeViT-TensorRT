#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from copy import deepcopy
from collections import OrderedDict

onnxFile = "../QAT/quant_LeVit-QAT.onnx"

# 用 onnx-graphsurgeon 打印 .onnx 文件逐层信息 ----------------------------------
graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(onnxFile)))

graph.inputs[0].shape = ['A',3,224,224]   # 调整输入维度，字符串表示 dynamic shape
# graph.inputs[1].shape = ['B']
graph.outputs[0].shape = [1,1000]       # 调整输出维度

print("# Traverse the node: -----------------------------------------------------")  # 遍历节点，打印：节点信息，输入张量，输出张量，父节点名，子节点名
for index, node in enumerate(graph.nodes):
    # if node.op == 'Where' and \
    #     node.o().op == 'Softmax':
    if 'DequantizeLinear_1048' in node.name:
    # if 'MatMul' in node.name:
        print("Node%4d: op=%s, name=%s, attrs=%s"%(index, node.op,node.name, "".join(["{"] + [str(key)+" : "+str(value)+", " for key, value in node.attrs.items()] + ["}"])))
        for jndex, inputTensor in enumerate(node.inputs):
            print("\tInTensor  %d: %s"%(jndex, inputTensor))
        for jndex, outputTensor in enumerate(node.outputs):
            print("\tOutTensor %d: %s"%(jndex, outputTensor))

        fatherNodeList = []
        for newNode in graph.nodes:
            for newOutputTensor in newNode.outputs:
                if newOutputTensor in node.inputs:
                    fatherNodeList.append(newNode)
        for jndex, newNode in enumerate(fatherNodeList):
            print("\tFatherNode%d: %s"%(jndex,newNode.name))

        sonNodeList = []
        for newNode in graph.nodes:
            for newInputTensor in newNode.inputs:
                if newInputTensor in node.outputs:
                    sonNodeList.append(newNode)
        for jndex, newNode in enumerate(sonNodeList):
            print("\tSonNode   %d: %s"%(jndex,newNode.name))

print("# Traverse the tensor: ---------------------------------------------------") # 遍历张量，打印：张量信息，以本张量作为输入张量的节点名，以本张量作为输出张量的节点名，父张量信息，子张量信息
# for index,(name,tensor) in enumerate(graph.tensors().items()):
#     print("Tensor%4d: name=%s, desc=%s"%(index, name, tensor))
#     for jndex, inputNode in enumerate(tensor.inputs):
#         print("\tInNode      %d: %s"%(jndex, inputNode.name))
#     for jndex, outputNode in enumerate(tensor.outputs):
#         print("\tOutNode     %d: %s"%(jndex, outputNode.name))

#     fatherTensorList = []
#     for newTensor in list(graph.tensors().values()):
#         for newOutputNode in newTensor.outputs:
#             if newOutputNode in tensor.inputs:
#                 fatherTensorList.append(newTensor)
#     for jndex, newTensor in enumerate(fatherTensorList):
#         print("\tFatherTensor%d: %s"%(jndex,newTensor))

#     sonTensorList = []
#     for newTensor in list(graph.tensors().values()):
#         for newInputNode in newTensor.inputs:
#             if newInputNode in tensor.outputs:
#                 sonTensorList.append(newTensor)
#     for jndex, newTensor in enumerate(sonTensorList):
#         print("\tSonTensor   %d: %s"%(jndex,newTensor))