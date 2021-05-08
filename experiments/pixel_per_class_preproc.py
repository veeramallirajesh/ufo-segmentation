# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
import os
import numpy as np
from dataloader.ufs_data import UFSegmentationDataset
from omegaconf import DictConfig, OmegaConf
from dataloader.preprocessing import Compose, ToTensor
from augmentations.augmentations import FreeScale
from PIL import Image
import matplotlib.pyplot as plt

transform = Compose(
    [
        FreeScale(size=(512, 512)),
        ToTensor()
    ]
)
cfg = OmegaConf.load("../config.yaml")
mask_path = f"/home/rveeramalli/ufo-segmentation/data_v1/masks"
train_indices = f"/home/rveeramalli/ufo-segmentation/data_v1/indices_ufs/train.npy"
bbox_dir = f"/home/rveeramalli/ufo-segmentation/data/simple_cod_subset/ground-truth"
data_root = f"/home/rveeramalli/ufo-segmentation/data_v1"

indices = np.load(train_indices)
dataset = UFSegmentationDataset(cfg=cfg, data_root=data_root, bbox_dir=bbox_dir, transform=transform)
save_path = f"/home/rveeramalli/ufo-segmentation/data_v1/processed"
for idx in indices:
    print(f"processing sample {idx}")
    img, mask = dataset[idx]
    name = os.path.splitext(os.path.basename(dataset.img_files[idx]))[0]
    img = Image.fromarray(img.numpy().squeeze()).convert("L")
    mask = Image.fromarray(mask.numpy().astype(np.uint8).squeeze()).convert("L")
    img.save(os.path.join(save_path, "images", name + ".jpg"))
    mask.save(os.path.join(save_path, "masks", name + ".png"))








