# 导入所需的库
import argparse  # 用于解析命令行参数
import logging  # 处理日志
import os  # 文件和目录操作
import random  # 随机数生成
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.backends.cudnn as cudnn  # 控制CUDNN加速选项

from importlib import import_module  # 动态导入模块

# 导入自定义模块
# from lora_sam import LoRA_Sam  # LoRA（低秩适配）实现
from segment_anything import sam_model_registry  # SAM模型注册工具
from segment_anything.modeling import PromptEncoder  # SAM模型注册工具
from trainer import trainer_synapse  # Synapse数据集的训练器
from icecream import ic  # 调试工具

# 定义命令行参数解析
parser = argparse.ArgumentParser()
# 数据路径、输出路径和数据集名称
parser.add_argument('--root_path', type=str, default='./trainset/train_npz_new_224', help='数据根目录')
parser.add_argument('--output', type=str, default='./output/sam/results', help='输出结果目录')
parser.add_argument('--dataset', type=str, default='Synapse', help='实验名称')

# 数据和训练相关参数
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='数据列表路径')
parser.add_argument('--num_classes', type=int, default=8, help='分类数')
parser.add_argument('--max_iterations', type=int, default=30000, help='最大迭代次数')
parser.add_argument('--max_epochs', type=int, default=200, help='最大训练轮数')
parser.add_argument('--stop_epoch', type=int, default=20, help='停止训练的轮数')
parser.add_argument('--batch_size', type=int, default=12, help='每GPU的批量大小')
parser.add_argument('--n_gpu', type=int, default=2, help='GPU数量')
parser.add_argument('--split', type=str,default='train', help='list dir')

# 决定性和学习率控制参数
parser.add_argument('--deterministic', type=int, default=1, help='是否使用确定性训练')
parser.add_argument('--base_lr', type=float, default=0.0025, help='学习率')

# 图像尺寸、随机种子和模型配置
parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
parser.add_argument('--seed', type=int, default=1234, help='随机种子')

# --------------
# parser.add_argument('--vit_name', type=str, default='vit_b', help='选择的ViT模型')
# parser.add_argument('--ckpt', type=str, default='model_weights/sam_vit_b_01ec64.pth', help='sam预训练模型路径')
# --------------
parser.add_argument('--vit_name', type=str, default='vit_h', help='选择的ViT模型')
parser.add_argument('--ckpt', type=str, default='model_weights/sam_vit_h_4b8939.pth', help='sam预训练模型路径')
# --------------

# LoRA相关参数
parser.add_argument('--lora_ckpt', type=str, default=None, help='微调的LoRA模型路径')
parser.add_argument('--rank', type=int, default=4, help='LoRA适配的秩参数')

# 其他配置
parser.add_argument('--warmup', action='store_true', help='是否启用学习率预热')
parser.add_argument('--warmup_period', type=int, default=250, help='预热持续的迭代数')
parser.add_argument('--AdamW', action='store_true', help='是否使用AdamW优化器')

# 网络选择
parser.add_argument('--module', type=str, default='net_injector', help='动态加载的模块名称')
parser.add_argument('--dice_param', type=float, default=0.8, help='Dice损失的参数')

parser.add_argument(
        '--use_amp',
        action='store_true',
        help='Use Automatic Mixed Precision for training'
    )

parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='How many iterations to wait before logging training status (default: 100)'
    )

parser.add_argument(
    '--save_interval',
    type=int,
    default=50,
    help='How many epochs to wait before saving the model (default: 50)'
)
args = parser.parse_args()

if __name__ == "__main__":
    # 设置确定性训练以控制随机性
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 数据集配置
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        }
    }

    # 设置实验输出目录
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


    # 动态加载LoRA模块并初始化
    pkg = import_module(args.module)

    classnames = ["spleen", "right kidney", "left kidney", "gallbladder", "liver", "stomach", "aorta", "pancreas"]

    class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach',
                     7: 'aorta', 8: 'pancreas'}

    net = pkg.MultiModalSegmentor(sam, classnames, lora_rank=4).cuda()

    # 加载LoRA权重（如果指定）
    # print('------Load model weight------')
    # weight = './output/sam/results/Synapse_224_pretrain_vit_h_new_2_decoder_epo300_bs12_lr0.0026/epoch_300.pth'
    # net.load_all_weights(weight)

    # 根据分类任务配置输出
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4  # 低分辨率调整

    # 保存配置文件
    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    # 开始训练
    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
