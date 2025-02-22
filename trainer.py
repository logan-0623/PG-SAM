# trainer.py

import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from PIL import Image

from utils import DiceLoss, Focal_loss, train_single_volume
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Mapping from class indices to names
class_to_name = {
    1: 'spleen', 2: 'right kidney', 3: 'left kidney',
    4: 'gallbladder', 5: 'liver', 6: 'stomach',
    7: 'aorta', 8: 'pancreas'
}

def calc_loss(outputs, low_res_label_batch, label_batch, ce_loss, dice_loss, dice_weight:float=0.8):

    low_res_logits = outputs['low_res_logits']
    mask_224 = outputs['masks']

    # 低纬度 的 loss （ dim=56 ）
    loss_ce1 = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice1 = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss1 = ((1 - dice_weight) * loss_ce1 + dice_weight * loss_dice1)

    # 224维度的 mask loss
    loss_ce2 = ce_loss(mask_224, label_batch[:].long())
    loss_dice2 = dice_loss(mask_224, label_batch, softmax=True)
    loss2 = ((1 - dice_weight) * loss_ce2 + dice_weight * loss_dice2)

    loss = loss1 + loss2
    loss_ce = loss_ce1 + loss_ce2
    loss_dice = loss_dice1 + loss_dice2

    return loss, loss_ce, loss_dice


class AddGaussianNoise(object):
    """
    Adds Gaussian noise to an image.
    """
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=1):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img).astype(np.float32)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            img += N
            img = np.clip(img, 0, 255).astype('uint8')
            return Image.fromarray(img).convert('RGB')
        return img

class AddPepperNoise(object):
    """
    Adds salt and pepper noise to an image.
    """
    def __init__(self, snr: float, p: float = 0.9):
        assert isinstance(snr, float) and isinstance(p, float), "snr and p must be floats"
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).astype(np.uint8)
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = 1 - self.snr
            mask = np.random.choice(
                (0, 1, 2),
                size=(h, w, 1),
                p=[signal_pct, noise_pct / 2., noise_pct / 2.]
            )
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # Salt noise
            img_[mask == 2] = 0    # Pepper noise
            return Image.fromarray(img_).convert('RGB')
        return img

def setup_logging(snapshot_path):
    """
    Set up logging to file and stdout.
    """
    os.makedirs(snapshot_path, exist_ok=True)
    log_file = os.path.join(snapshot_path, "log.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def initialize_dataloader(args):
    """
    Initialize the DataLoader for the Synapse dataset.
    """
    base_dir = "/mnt/sda/feilongtang/John/Miccai_sam/code/dual-sam/trainset/train_npz_new_224"
    list_dir = "./lists/lists_Synapse"
    data_type = "Big"  # This can be parameterized if needed

    logging.info(f'Using dataset type: {data_type}')

    if data_type == "Big":
        text_dir = "/mnt/sda/feilongtang/John/Miccai_sam/code/dual-sam/trainset/output_image_text_pairs_all_1/texts"
    else:
        text_dir = "/mnt/sda/feilongtang/John/Miccai_sam/code/dual-sam/trainset/output_image_text_pairs/texts"
        logging.info('Using small dataset')

    transform = transforms.Compose([
        RandomGenerator([224, 224], [56, 56])
    ])

    print('Use Picture Enhancement')
    dataset = Synapse_dataset(
        base_dir=base_dir,
        list_dir=list_dir,
        text_dir=text_dir,
        split="train",
        data=data_type,
        transform=transform
    )

    logging.info(f'Training dataset size: {len(dataset)}')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader = DataLoader(
        dataset,
        batch_size=8, # 小一点的 batch size
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    return dataloader

def save_model(model, save_path, is_parallel):
    """
    Save the model state_dict.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        if is_parallel:
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    """
    Main training loop for the Synapse dataset.
    """
    setup_logging(snapshot_path)
    logging.info(f"Training started with arguments: {args}")

    dataloader = initialize_dataloader(args)

    model = model.cuda()
    model.train()

    for param in model.GuideMatrixGenerator.sam_model.mask_decoder.parameters():
        param.requires_grad = True

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logging.info(f"Trainable: {name}")
    #     else:
    #         logging.warning(f"Frozen: {name}")



    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total parameters: {total_params}, Trainable parameters: {trainable_params}')

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes + 1)

    # Initialize optimizer
    base_lr = args.base_lr

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    if args.AdamW:
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],  # Explicit list
            lr=b_lr,
            betas=(0.9, 0.999),
            weight_decay=0.1
        )

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    iter_num = 0
    max_iterations = args.max_epochs * len(dataloader)
    logging.info(f"{len(dataloader)} iterations per epoch. {max_iterations} max iterations.")

    # Initialize progress bar
    epoch_iterator = tqdm(range(args.max_epochs), desc="Epoch", ncols=70)

    for epoch_num in epoch_iterator:
        for i_batch, sampled_batch in enumerate(dataloader):
            image_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            text_batch = sampled_batch['text']
            low_res_label_batch = sampled_batch['low_res_label'].cuda()

            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            outputs1 = model(image_batch, text_batch, multimask_output, args.img_size, gt=low_res_label_batch)

            assert outputs1['low_res_logits'].requires_grad, "outputs1 has no gradients!"



            loss, loss_ce1, loss_dice1 = calc_loss(outputs1, low_res_label_batch, label_batch, ce_loss, dice_loss,
                                                    dice_weight=args.dice_param)

            assert loss.requires_grad, "Loss does not require gradients!"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (
                            1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_


            iter_num += 1

            # Logging
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
            writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)

            logging.info(f'Iteration {iter_num}: Loss={loss.item():.4f}, CE={loss_ce1.item():.4f}, Dice={loss_dice1.item():.4f}')

        # Visualizations every log_interval iterations
        if iter_num % args.log_interval == 0:
            img = image_batch[0, 0:1, :, :]
            img = (img - img.min()) / (img.max() - img.min())
            writer.add_image('train/Image', img, iter_num)

            # output_masks = outputs['masks']

            # output_masks = (outputs1['masks'] + outputs2['masks']) / 2
            # output_masks = torch.argmax(F.softmax(output_masks, dim=1), dim=1, keepdim=True)

            # writer.add_image('train/Prediction', output_masks[0] * 50, iter_num)
            #
            # ground_truth = label_batch[0].unsqueeze(0) * 50
            # writer.add_image('train/GroundTruth', ground_truth, iter_num)

        # Save model checkpoints at save_interval epochs
        if (epoch_num + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(snapshot_path, f'epoch_{epoch_num + 1}.pth')
            save_model(model, checkpoint_path, False)

        # Early stopping
        if (epoch_num + 1) >= args.stop_epoch:
            final_checkpoint = os.path.join(snapshot_path, f'epoch_{epoch_num + 1}.pth')

            logging.info("Early stopping triggered.")
            epoch_iterator.close()
            break

    # Final model save
    final_save_path = os.path.join(snapshot_path, f'final_epoch_{args.max_epochs}.pth')
    save_model(model, final_save_path, False)
    writer.close()
    logging.info("Training Finished!")
    return "Training Finished!"
