import argparse
import time
import torch
from pathlib import Path
from timm.models import create_model
import levit
import levit_c

def get_args_parser():
    parser = argparse.ArgumentParser(
        'LeViT export script', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='LeViT_128S', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')
    parser.add_argument('--finetune', default='LeViT-128S.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    return parser

def export(model):
    model.eval()
    input_names = ["input_0"]
    output_names = ["output_0"]
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, 'LeViT-128S.onnx',
                    verbose=False, opset_version=12,
                    input_names=input_names,
                    output_names=output_names,
                    do_constant_folding=True,
                    )

def main(args):
    print(args)
    device = torch.device(args.device)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=1000,
        distillation=False,
        pretrained=False,
        fuse=False,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias',
                  'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    export(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'LeViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    