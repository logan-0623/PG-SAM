import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from torchvision import transforms

from icecream import ic

class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach',
                 7: 'aorta', 8: 'pancreas'}


def inference(args, multimask_output, db_config, model, test_save_path=None):
    print("Testing ...")

    text_dir = " ./testset/test_vol_h5/output_image_text_pairs/texts " # put your test data here

    db_test = Synapse_dataset(args.volume_path, args.list_dir, text_dir, split="train", data="test", transform = None)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
    # for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
        text_batch = sampled_batch['text']
        metric_i = test_single_volume(image, label, text_batch, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'])

        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes + 1):
        try:
            logging.info('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1]))
        except:
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info("Testing Finished!")
    return 1



def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str,
                        default='./testset/test_vol_h5/',
                        help='Path to test volumes')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/',
                        help='Directory containing dataset lists')
    parser.add_argument('--output_dir', type=str,
                        default='./output/sam/test_results',
                        help='Directory for saving outputs')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for SAM model')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='Use deterministic training')

    parser.add_argument('--lora_ckpt', type=str,
                        default='', # put the saved model weight file
                        help='Path to LoRA checkpoint')
    # --------------
    parser.add_argument('--vit_name', type=str, default='vit_h', help='the Vit model')
    parser.add_argument('--ckpt', type=str, default='model_weights/sam_vit_h_4b8939.pth', help='sam check point')
    # --------------

    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='net_injector', help='Dynamic module import')
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()

    if args.config is not None:
        config_dict = config_to_dict(args.config)
        for key, value in config_dict.items():
            setattr(args, key, value)

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    print('Use net New Vit H decoder')
    sam, img_embedding_size = sam_model_registry["build_sam_vit_h_new"](checkpoint="model_weights/sam_vit_h_4b8939.pth",
                                                                        image_size=224,
                                                                        num_classes=args.num_classes + 1,
                                                                        pixel_mean=[0, 0, 0],
                                                                        pixel_std=[1, 1, 1]
                                                                        )

    pkg = import_module(args.module)
    classnames = ["spleen", "kidney(R)", "kidney(L)", "gallbladder", "liver", "stomach", "aorta", "pancreas"]
    net = pkg.MultiModalSegmentor(sam, classnames, lora_rank=4).cuda()

    assert args.lora_ckpt is not None
    net.load_all_weights(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # Initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder+ '/' + 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    test_save_path = "" # put your path to save

    print('---------------------------------------------------------------------------------')
    print("dataset_config[dataset_name]['Dataset']", dataset_config[dataset_name]['Dataset'])

    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path)
