# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
import os
import time
from hydra.experimental import (
    initialize,
    initialize_config_module,
    initialize_config_dir,
    compose,
)
from omegaconf import OmegaConf, open_dict

import pytorch_lightning as pl

from models.pspnet import PSPNetLitModel
from utils.data import get_data_module

import numpy as np
import matplotlib.pyplot as plt

HOME_DIR = "/home/rajesh/pspnet"
os.chdir(HOME_DIR)
logs_path = os.path.join("outputs", "2020-11-18", "12-42-38")  # 17-06-29
version = 0
epoch = 19
with initialize(config_path=os.path.join(logs_path, ".hydra")):
    cfg = compose(
        config_name="config",
        overrides=[
            "data.dir=/media/ssd2/datasets/public-datasets",
            "train.gpus=0",
            "train.batch_size=8",
            "data.dataset=kitti",
            "data.num_workers=8",
        ],
    )

print(OmegaConf.to_yaml(cfg))

dm = get_data_module(cfg)
# dm.setup()
dm.setup(stage="fit")  # for the Kitti dataset

with open_dict(cfg):
    cfg.model.classes = dm.classes
    print("Number of classes are {}".format(dm.classes))

model = PSPNetLitModel.load_from_checkpoint(
    os.path.join(
        logs_path,
        "lightning_logs",
        "version_{}".format(version),
        "checkpoints",
        "epoch={}.ckpt".format(epoch),
    ),
    cfg=cfg,
)
model.eval()
model.freeze()

acc_start_time = 0
start = time.time()
batch_size = cfg.train.batch_size
for idx, (images, segmentations) in enumerate(dm.test_dataloader()):

    predictions = model(images).numpy()
    # print(predictions.shape)
    images, segmentations = images.numpy(), segmentations.numpy()
    predictions = np.argmax(predictions, axis=1)
    acc_start_time = time.time() - start
    print(
        "Processed {} images, with batch size {}, took {} seconds, averaging {} seconds per image.".format(
            (idx + 1) * batch_size,
            batch_size,
            acc_start_time,
            acc_start_time / ((idx + 1) * batch_size),
        )
    )
    # break

print(
    "Two example images, images in first row, targets in second, predictions in third, colorbar indicates class ids"
)

fig, axes = plt.subplots(figsize=(16, 8), nrows=1, ncols=2)
plt.tight_layout()
im = axes[0].imshow(images[0].mean(axis=0), cmap="gray")
im = axes[1].imshow(images[1].mean(axis=0), cmap="gray")
for item in axes:
    item.axis("off")
fig.subplots_adjust(right=0.8)
plt.show()

fig, axes = plt.subplots(figsize=(16, 10), nrows=2, ncols=2)
plt.tight_layout()
im = axes[0, 0].imshow(segmentations[0], vmin=0, vmax=dm.classes)
im = axes[1, 0].imshow(predictions[0], vmin=0, vmax=dm.classes)
im = axes[0, 1].imshow(segmentations[1], vmin=0, vmax=dm.classes)
im = axes[1, 1].imshow(predictions[1], vmin=0, vmax=dm.classes)

for row in axes:
    for item in row:
        item.axis("off")

# fig.subplots_adjust(bottom=0.9)
cbar_ax = fig.add_axes([0, -0.05, 1, 0.05])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_ticks(range(dm.classes))
cbar.ax.set_xticklabels(
    [
        "{} - ".format(idx) + "Label" if idx in np.unique(predictions[:2]) else ""
        # "{} - ".format(idx) + dm.classes[idx].name if idx in np.unique(predictions[:2]) else ""
        for idx in range(dm.classes)
    ],
    rotation=90,
)
plt.show()

print(
    "Showing difference between targets and predictions, yellow indicates successful predictions, unsuccessful otherwise"
)

fig, axes = plt.subplots(figsize=(16, 8), nrows=1, ncols=2)
im = axes[0].imshow((segmentations[0] - predictions[0]) == 0)
im = axes[1].imshow((segmentations[1] - predictions[1]) == 0)
for item in axes:
    item.axis("off")
plt.show()
