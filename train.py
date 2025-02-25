# Import required libraries
import argparse  # For parsing command line arguments
import logging  # For handling logs
import os  # For file and directory operations
import random  # For generating random numbers
import numpy as np  # Numerical computation library
import torch  # PyTorch deep learning framework
import torch.backends.cudnn as cudnn  # For controlling CUDNN acceleration options

from importlib import import_module  # For dynamically importing modules

# Import custom modules
# from lora_sam import LoRA_Sam  # Implementation of LoRA (Low-Rank Adaptation)
from segment_anything import sam_model_registry  # SAM model registry tool
from segment_anything.modeling import PromptEncoder  # SAM model registry tool
from trainer import trainer_synapse  # Trainer for the Synapse dataset
from icecream import ic  # Debugging tool

# Define command line argument parsing
parser = argparse.ArgumentParser()
# Data path, output path and dataset name
parser.add_argument('--root_path', type=str, default='./trainset/train_npz_new_224', help='Data root directory')
parser.add_argument('--output', type=str, default='./output/sam/results', help='Output results directory')
parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')

# Data and training related parameters
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='Data list directory')
parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
parser.add_argument('--max_iterations', type=int, default=30000, help='Maximum number of iterations')
parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of training epochs')
parser.add_argument('--stop_epoch', type=int, default=20, help='Epoch to stop training')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
parser.add_argument('--n_gpu', type=int, default=2, help='Number of GPUs')
parser.add_argument('--split', type=str, default='train', help='List directory split')

# Determinism and learning rate control parameters
parser.add_argument('--deterministic', type=int, default=1, help='Whether to use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0025, help='Learning rate')

# Image size, random seed and model configuration
parser.add_argument('--img_size', type=int, default=224, help='Input image size')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')

# --------------
# parser.add_argument('--vit_name', type=str, default='vit_b', help='Selected ViT model')
# parser.add_argument('--ckpt', type=str, default='model_weights/sam_vit_b_01ec64.pth', help='SAM pre-trained model path')
# --------------
parser.add_argument('--vit_name', type=str, default='vit_h', help='Selected ViT model')
parser.add_argument('--ckpt', type=str, default='model_weights/sam_vit_h_4b8939.pth', help='SAM pre-trained model path')
# --------------

# LoRA related parameters
parser.add_argument('--lora_ckpt', type=str, default=None, help='Path to the fine-tuned LoRA model')
parser.add_argument('--rank', type=int, default=4, help='Rank parameter for LoRA adaptation')

# Other configurations
parser.add_argument('--warmup', action='store_true', help='Whether to use learning rate warmup')
parser.add_argument('--warmup_period', type=int, default=250, help='Number of iterations for warmup')
parser.add_argument('--AdamW', action='store_true', help='Whether to use the AdamW optimizer')

# Network selection
parser.add_argument('--module', type=str, default='net_injector', help='Name of the dynamically loaded module')
parser.add_argument('--dice_param', type=float, default=0.8, help='Dice loss parameter')

parser.add_argument(
        '--use_amp',
        action='store_true',
        help='Use Automatic Mixed Precision for training'
    )

parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='Number of iterations to wait before logging training status (default: 100)'
    )

parser.add_argument(
    '--save_interval',
    type=int,
    default=50,
    help='Number of epochs to wait before saving the model (default: 50)'
)
args = parser.parse_args()

if __name__ == "__main__":
    # Set deterministic training to control randomness
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Dataset configuration
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        }
    }

    # Set experiment output directory
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name + '_force_decoder'
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    print('Use net New Vit H decoder')
    sam, img_embedding_size = sam_model_registry["build_sam_vit_h_new"](checkpoint=args.ckpt,
                                                                        image_size=224,
                                                                        num_classes=args.num_classes + 1,
                                                                        pixel_mean=[0, 0, 0],
                                                                        pixel_std=[1, 1, 1]
                                                                        )

    # Dynamically load the LoRA module and initialize
    pkg = import_module(args.module)

    classnames = ["spleen", "right kidney", "left kidney", "gallbladder", "liver", "stomach", "aorta", "pancreas"]

    class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach',
                     7: 'aorta', 8: 'pancreas'}

    net = pkg.MultiModalSegmentor(sam, classnames, lora_rank=4).cuda()

    # Load LoRA weights (if specified)
    # print('------Load model weight------')
    # weight = './output/sam/results/Synapse_224_pretrain_vit_h_new_2_decoder_epo300_bs12_lr0.0026/epoch_300.pth'
    # net.load_all_weights(weight)

    # Set multimask_output based on classification task configuration
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4  # Low resolution adjustment

    # Save configuration file
    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    # Start training
    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
