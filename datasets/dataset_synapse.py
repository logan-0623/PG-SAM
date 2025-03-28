import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_crop(image, label):
    min_ratio = 0.2
    max_ratio = 0.8

    w, h = image.shape

    ratio = random.random()

    scale = min_ratio + ratio * (max_ratio - min_ratio)

    new_h = int(h * scale)
    new_w = int(w * scale)

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    image = image[x:x + new_w, y:y + new_h]
    label = label[x:x + new_w, y:y + new_h]

    return image, label


def random_scale(image, label, scale_factor=0.6):
    min_ratio = 0.2
    max_ratio = 0.8

    w, h = image.shape

    ratio = random.random()

    scale = min_ratio + ratio * (max_ratio - min_ratio)

    new_h = int(h * scale)
    new_w = int(w * scale)

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    image = image[x:x + new_w, y:y + new_h]
    label = label[x:x + new_w, y:y + new_h]

    return image, label


def random_elastic(image, label, alpha, sigma,
                   alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])

    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)

    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    labelB = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    labelC = map_coordinates(labelB, indices, order=1, mode='constant').reshape(shape)

    return imageC, labelC


def random_gaussian(image, var=0.1):
    noise = np.random.normal(0, var, image.shape)
    image = image + noise
    return image


def random_gaussian_filter(im, K_size=3, sigma=1.3):
    im = im * 255
    img = np.asarray(np.uint8(im))
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    out = out.astype(np.float32) / 255
    out = np.squeeze(out)
    return out


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        text = sample['text']  # Preserve text during transformation

        # if random.random() > 0.55:
        #     image, label = random_rotate(image, label)
        # elif random.random() > 0.25:
        #     image, label = random_scale(image, label)
        # elif random.random() > 0.15:
        #     image, label = random_elastic(image, label, image.shape[1] * 2,
        #                                   image.shape[1] * 0.08,
        #                                   image.shape[1] * 0.08)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))

        sample = {
            'image': image,
            'label': label.long(),
            'low_res_label': low_res_label.long(),
            'text': text  # Include text in transformed sample
        }

        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, text_dir, split, transform=None, data = None):
        self.transform = transform
        self.split = split
        self.data = data

        if self.data == "Big":
            print("----------------------------------Use full data to train----------------------------------")
            self.sample_list = open(os.path.join(list_dir, "train_full.txt")).readlines()
        elif self.data == "test":
            print("----------------------------------Use test data----------------------------------")
            self.sample_list = open(os.path.join(list_dir, "test_vol.txt")).readlines()
        else:
            self.sample_list = open(os.path.join(list_dir, f"{split}.txt")).readlines()

        self.data_dir = base_dir

        self.text_dir = text_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        if self.data == "test":
            slice_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(slice_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, f"{slice_name}.npz")
            data = np.load(data_path)
            image, label = data['image'], data['label']

        # print('label',  label)

        # Load text description using index-based naming (text_0.txt, text_1.txt, etc.)
        if self.text_dir is not None:
            # print("----------------------idx------------------------------",idx)
            text_path = os.path.join(self.text_dir, f"text_{idx}.txt")
            try:
                with open(text_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
            except (FileNotFoundError, IOError):
                text = "No description available."
        else:
            text = None

        sample = {
            'image': image,
            'label': label,
            'text': text,
            'case_name': slice_name
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class Synapse_dataset_test(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


# Test code
if __name__ == "__main__":
    base_dir = "./trainset/train_npz_new_224"
    list_dir = "./lists/lists_Synapse"
    text_dir = "./trainset/output_image_text_pairs/texts"

    # Create dataset instance
    dataset = Synapse_dataset(
        base_dir=base_dir,
        list_dir=list_dir,
        text_dir=text_dir,
        split="train",
        transform=RandomGenerator([224, 224], [56, 56])
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    # Test batch loading
    for batch in dataloader:
        print("Batch contents:")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Low-res label shape: {batch['low_res_label'].shape}")
        print(f"Text sample: {batch['text'][0]}")
        print(f"Case name: {batch['case_name'][0]}")
        break