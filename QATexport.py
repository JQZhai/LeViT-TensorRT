import argparse
import time
import torch
from pathlib import Path
from timm.models import create_model
import levit
import levit_c

def AutoExport(check_add):
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    quant_modules.initialize()
    model = create_model(
    'LeViT_128S',
    num_classes=1000,
    distillation=False,
    pretrained=False,
    fuse=False,
    )
    checkpoint = torch.load(check_add, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.cuda()
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    input_names = ["input_0"]
    output_names = ["output_0"]
    torch.onnx.export(model, dummy_input, 'QAT/quant_LeVit-QAT.onnx',
                    verbose=False, opset_version=13,
                    input_names=input_names,
                    output_names=output_names,
                    do_constant_folding=False,
                    )

if __name__ == '__main__':
    AutoExport('QAT/quant_LeVit-calibrated45.pth')