import torch
import time
import numpy as np
import onnxruntime
import sys
import argparse
sys.path.append('../')
from datasets import build_dataset

def parse_option():
    parser = argparse.ArgumentParser('Evaluation script of Swin Transformer TensorRT engine', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch-size', required=True, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='../imagenet_1k', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', default='./weights/swin_tiny_patch4_window7_224.engine', help='TensorRT engine')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],type=str, help='Image Net dataset path')
    parser.add_argument('--num_workers', default=10, type=int)
    args, unparsed = parser.parse_known_args()

    return args

def get_input_shape(binding_dims):
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s' %(str(binding_dims)))

class Processor():
    def __init__(self, model):
        self.ort_session =  onnxruntime.InferenceSession(model,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = []
        for i in range(len(self.ort_session.get_outputs())):
            output_name = self.ort_session.get_outputs()[i].name
            print("output name {}:".format(i), output_name)
            output_shape = self.ort_session.get_outputs()[i].shape
            print("output shape {}:".format(i), output_shape)
            self.output_names.append(output_name)
        
        self.input_shape = get_input_shape(self.ort_session.get_inputs()[0].shape)
        print('---self.input_shape: ', self.input_shape)

    def inference(self, img):
        res = self.ort_session.run(self.output_names, {self.input_name: img})
        return res

def create_dataset_eval(args):
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False
    )
    return data_loader_val

def validate(data_loader_val, model_path, args):
    total_cnt = 0
    accurate_cnt = 0

    processor = Processor(model_path)

    start = time.time()
    for image, target in data_loader_val:
        if len(image) == args.batch_size:
            print('total_cnt: ', total_cnt)
            total_cnt += len(image)
            cur_image = image.numpy()
            batch_images = cur_image
            batch_labels = target.numpy()

            outputs_onnxrt = processor.inference(batch_images)
            accurate_cnt += image_class_accurate(outputs_onnxrt, batch_labels)
    duration = time.time() - start

    print("Evaluation of TRT QAT model on {} images: {}, fps: {}".format(total_cnt,
                                                                         float(accurate_cnt) / float(total_cnt),
                                                                         float(total_cnt) / float(duration)))
    print('Duration: ', duration)

def image_class_accurate(pred, target):
    '''
    Return the number of accurate prediction.
    - pred: engine's output (batch_size, 1, class_num)
    - target: labels of sampl (batch_size, )
    '''
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    pred_label = np.squeeze(np.argmax(pred, axis=-1))
    correct_cnt = np.sum(pred_label == target)
    print('pred_label: ', pred_label, ' target: ', target)

    return correct_cnt

if __name__ == '__main__':
    _ = parse_option()
    data_loader_val = create_dataset_eval(_)
    validate(data_loader_val, _.resume, _)