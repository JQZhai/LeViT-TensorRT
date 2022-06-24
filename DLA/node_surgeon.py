from collections import OrderedDict
from copy import deepcopy
import numpy as np
import onnx
import onnx_graphsurgeon as gs

sourceOnnx = "../LeViT-128S.onnx"
destinationOnnx = "./LeViT-128S-DLA.onnx"

Identity = True
HardSigmod = False

nIdentity = 0
nHardsigmoid = 0


graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

dict_ = {'Identity_11':[1,196,4,-1], 'Identity_10':[1,196,128], 'Identity_9':[1,49,6,-1],
        'Identity_8':[1,49,192], 'Identity_7':[1,49,6,-1], 'Identity_6':[1,49,192],
        'Identity_5':[1,16,8,-1], 'Identity_4':[1,16,256], 'Identity_3':[1,16,8,-1],
        'Identity_2':[1,16,256], 'Identity_1':[1,16,8,-1], 'Identity_0':[1,16,256]}

if Identity:
    for index, node in enumerate(graph.nodes):
        if node.name in dict_ and \
            node.o().op == 'Reshape' and node.o().inputs[1] == node.outputs[0]:
            t1 = gs.Constant(str(index)+'_Identity', np.ascontiguousarray(np.array(dict_[node.name], dtype=np.int64)))
            node.o().inputs[1] = t1
            node.outputs.clear()
            nIdentity +=1
            graph.cleanup().toposort()

if HardSigmod:
    for index, node in enumerate(graph.nodes):

        if 'HardSigmoid' in node.name:
            v0 = gs.Constant(str(index)+'_v0', np.ascontiguousarray(np.array([0], dtype=np.float32)))
            v1 = gs.Constant(str(index)+'_v1', np.ascontiguousarray(np.array([1], dtype=np.float32)))

            mul_out = gs.Variable(str(index)+"_mul_out", dtype=np.float32)
            mul_node = gs.Node(op='Mul',name=str(index)+'_Mul', inputs=node.i().outputs, outputs=[mul_out])
            graph.nodes.append(mul_node)

            min_out = gs.Variable(str(index)+"_min_out", dtype=np.float32)
            min_node = gs.Node(op='Min',name=str(index)+'_Min', inputs=[v1,mul_out], outputs=[min_out])
            graph.nodes.append(min_node)

            max_out = gs.Variable(str(index)+"_max_out", dtype=np.float32)
            max_node = gs.Node(op='Max',name=str(index)+'_Max', inputs=[v0,min_out], outputs=[max_out])
            graph.nodes.append(max_node)

            node.o().inputs = [node.i().outputs[0], max_node.outputs[0]]
            node.outputs.clear()
            nHardsigmoid +=1

            graph.cleanup().toposort()



print("finish onnx-graphsurgeon!")
onnx.save(gs.export_onnx(graph), "ModifyModel_2.onnx")
print("%4d Identity" %nIdentity)
print("%4d HardSigmod" %nHardsigmoid)

