import logging
import scipy
import os
import pickle
import itertools
import random
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import transforms
from torchvision.transforms import functional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_train_dataloader():
    raw_dataset = FFHQ(BASE_DIR='/home/ubuntu/')
    dataset = DataLoader(
        raw_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        drop_last=False,
    )

    return dataset


def get_files(BASE_DIR):
    files = []
    images_dir = os.path.join(BASE_DIR, "raw", "train")
    masks_dir = os.path.join(BASE_DIR, "masks", "train")

    images = os.listdir(images_dir)

    for image in images:
        files.append((os.path.join(images_dir, image), os.path.join(masks_dir, image)))

    return files


def arcface_process(path):
    # ArcFace / ResNet50 trained on VGGFace2
    img = Image.open(path).convert("RGB")
    img = transforms.Resize((224, 224))(img)
    img = transforms.ToTensor()(img)  # [0, 1]
    img = img * 255  # [0, 255]
    img = transforms.Normalize(mean=[131.0912, 103.8827, 91.4953], std=[1, 1, 1])(img)
    img = img.float()
    return img


def dilate_erosion_mask(mask_path, size):
    # Mask
    mask = Image.open(mask_path).convert("RGB")
    mask = transforms.Resize((size, size))(mask)
    mask = transforms.ToTensor()(mask)  # [0, 1]

    # Hair mask + Hair image
    hair_mask = mask[0, ...]
    # hair_mask = torch.unsqueeze(hair_mask, axis=0)
    hair_mask = hair_mask.numpy()
    hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=5)
    hair_mask_erode = scipy.ndimage.binary_erosion(
        hair_mask, iterations=5
    )  # , structure=np.ones((3, 3)).astype(hair_mask.dtype))

    hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
    hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

    return torch.from_numpy(hair_mask_dilate), torch.from_numpy(hair_mask_erode)


def process_image(img_path, mask_path, size=None, normalize=None):
    # Full image
    img = Image.open(img_path).convert("RGB")
    if size is None:
        img = transforms.Resize((1024, 1024))(img)
    else:
        img = transforms.Resize((size, size))(img)
    img = transforms.ToTensor()(img)

    if normalize is not None:
        img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

    # Mask
    mask = Image.open(mask_path).convert("RGB")
    if size is None:
        mask = transforms.Resize((1024, 1024))(mask)
    else:
        mask = transforms.Resize((size, size))(mask)
    mask = transforms.ToTensor()(mask)

    # Hair mask + Hair image
    hair_mask = mask[0, ...]
    hair_mask = torch.unsqueeze(hair_mask, axis=0)
    hair_image = img * hair_mask

    # Face mask + Face image
    face_mask = mask[-1, ...]
    face_mask = torch.unsqueeze(face_mask, axis=0)
    face_image = img * face_mask

    # Keep face and hair. Remove background image
    mask = torch.mean(mask, 0, keepdim=False)
    mask = (mask > 0.001).float()
    image_fg = img * mask

    return [img, mask, hair_mask, hair_image, face_mask, face_image, image_fg]


class FFHQ(Dataset):
    def __init__(self, BASE_DIR):

        logging.info(f"Reading data from:{BASE_DIR}")
        self.files = get_files(BASE_DIR)
        self.length = len(self.files)
        logging.info(f"Samples:{self.length}")

        self.indexes = [idx for idx in range(self.length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        paths = self.files[idx]

        # :paths is a tuple of 2 images; image and mask

        # Prepare arcface image
        arcface_image = arcface_process(paths[0])

        (
            image,
            mask,
            hair_mask,
            hair_image,
            face_mask,
            face_image,
            image_fg,
        ) = process_image(paths[0], paths[1])

        return (
            idx,
            arcface_image,
            image,
            mask,
            hair_mask,
            hair_image,
            face_mask,
            face_image,
            image_fg,
        )
