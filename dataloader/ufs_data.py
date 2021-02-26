# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from typing import Tuple
import pytorch_lightning as pl
import glob
import os
import math
import torch
import PIL
from PIL import Image, ImageDraw
from typing import Mapping
from augmentations.augmentations import (
    BinaryClassMap,
)
import matplotlib.pyplot as plt


class UFSegmentationDataset(Dataset):
    def __init__(
        self,
        cfg: Mapping = None,
        data_root: str = None,
        bbox_dir: str = None,
        transform=None,
    ):  # initial logic happens like transform
        super(UFSegmentationDataset, self).__init__()
        self.data_root = data_root
        self.bbox_dir = bbox_dir
        self.height = cfg["data"]["height"]
        self.width = cfg["data"]["width"]
        self.classes = cfg["data"]["classes"].lower()
        self.train = cfg["train"]["training"]
        self.img_files = glob.glob(os.path.join(self.data_root, "images", "*.jpg"))
        # if self.bbox_dir is not None:
        #     # List all the bounding box file paths
        #     bbox_list = glob.glob(os.path.join(bbox_dir, "*.txt"))
        self.transforms = transform
        if self.classes == "binary" and self.transforms is not None:
            self.transforms.append(
                BinaryClassMap()
            )  # Assigns binary class labels for the mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_files[index]
        mask_path = os.path.join(
            self.data_root, "masks", os.path.basename(img_path).split(".")[0] + ".png"
        )
        # Pre=process images with bbox
        if self.bbox_dir is not None:
            img, mask = self.pre_process_image_with_bb(img_path, mask_path)
            # if img.size != (self. width, self.height):
            #     print(f"Image path is: {img_path} and image size is: {img.size}")
        else:
            img = Image.open(img_path)
            mask = (
                np.array(Image.open(mask_path).convert("L")) // 255
            )  # Mapping class labels to 0's and 1's
            mask = Image.fromarray(mask)
        if self.transforms is not None:
            return self.transforms(img, mask)
        else:
            return img, mask

    def __len__(self):  # return count of sample we have
        return len(self.img_files)

    def __repr__(self):
        return "Under Water Fish Segmentation Dataset"

    # Code for cropping the images based on the bounding box to the model input size
    def pre_process_image_with_bb(self, img_path: str, mask_path: str = None):
        # (x1, y1) -- (left, upper), (x2, y2) -- (right, lower) coordinates
        w, h = Image.open(img_path).size
        (x1, y1), (x2, y2) = self.get_crop_coordinates(img_path)
        if (x2 - x1) < self.width:
            # if (x1 - (round(720-(x2 - x1))/2) < 0):
            half_x = (self.width - (x2 - x1)) / 2
            # Make sure the crop region does't go beyond the width of the image
            x2 = x2 + math.floor(half_x) if (x2 + math.floor(half_x)) < w else w
            # else:
            x1 = x1 - math.ceil(half_x) if (x1 - math.ceil(half_x)) > 0 else 0
        if (y2 - y1) < self.height:
            # if (y1 - (y2 - y1) < 0):
            half_y = (self.height - (y2 - y1)) / 2
            # Make sure the crop region does't go beyond the height of the image
            y2 = y2 + math.floor(half_y) if (y2 + math.floor(half_y)) < h else h
            # else:
            y1 = y1 - math.ceil(half_y) if (y1 - math.ceil(half_y)) > 0 else 0
        img = Image.open(img_path).crop((x1, y1, x2, y2))
        if mask_path is None:
            return (
                img,
                (x1, y1),
                (x2, y2),
            )  # Return pre-processed image and new bbox co-ordinates
        else:
            mask = Image.open(mask_path).convert("L").crop((x1, y1, x2, y2))
            return img, mask

    def get_crop_coordinates(self, path):
        bbox_file = os.path.join(
            self.bbox_dir, os.path.basename(path).split(".")[0] + ".txt"
        )
        if not os.path.exists(bbox_file):
            print(f"Label file does not exist {bbox_file}")
        with open(Path(bbox_file)) as f:
            objects = []
            for line in f:
                objects.append(line.strip().split())
                for obj in objects:
                    # If there are multiple fishes in the image crop the image including all of them
                    # Checking if the variables are already created
                    if "x1" and "y1" and "x2" and "y2" in locals():
                        # Top left coordinates of a bounding box
                        (x1, y1) = (min(x1, int(obj[1])), min(y1, int(obj[2])))
                        # Bottom right coordinates of a bounding box
                        (x2, y2) = (max(x2, int(obj[3])), max(y2, int(obj[4])))
                    # Single object/ fish in the image
                    else:
                        # Top left coordinates of a bounding box
                        (x1, y1) = (int(obj[1]), int(obj[2]))
                        # Bottom right coordinates of a bounding box
                        (x2, y2) = (int(obj[3]), int(obj[4]))
        return (x1, y1), (x2, y2)


class KittiDataset(Dataset):
    def __init__(
        self, data_root, mode="train", transform=None
    ):  # initial logic happens like transform
        super(KittiDataset, self).__init__()
        if mode == "train":
            self.img_files = glob.glob(
                os.path.join(data_root, "training", "image_2", "*.png")
            )
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(
                    os.path.join(
                        data_root, "training", "semantic", os.path.basename(img_path)
                    )
                )
            self.transforms = transform
            # self.transforms = transforms.Compose(
            #     [transforms.Resize((256, 256)),
            #     transforms.ToTensor()])
        else:
            self.img_files = glob.glob(
                os.path.join(data_root, "testing", "image_2", "*.png")
            )
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(
                    os.path.join(
                        data_root, "testing", "semantic", os.path.basename(img_path)
                    )
                )

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        img = Image.open(img_path).convert("RGB")
        label = Image.open(mask_path)
        return self.transforms(img), self.transforms(label)

    def __len__(self):  # return count of sample we have
        return len(self.img_files)

    def __repr__(self):
        return "KITTI Dataset"


class UFSDataModule(pl.LightningDataModule):
    def __init__(
        self, data_root, transform=None
    ):  # initial logic happens like transform
        super(UFSDataModule, self).__init__()
        self.img_files = glob.glob(os.path.join(data_root, "images", "*.jpg"))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(
                os.path.join(
                    data_root, "masks", os.path.basename(img_path).split(".")[0]
                )
                + ".png"
            )
        self.transforms = (
            transform  # transforms.Normalize((198, 198, 198), (64, 64, 64))
        )

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")
        # return torch.from_numpy(img).float(), torch.from_numpy(mask).float()
        return self.transforms(img), self.transforms(mask)

    def __len__(self):  # return count of sample we have
        return len(self.img_files)

    def __repr__(self):
        return "Under Water Fish Segmentation Dataset"
