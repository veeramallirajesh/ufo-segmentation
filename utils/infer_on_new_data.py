# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

import os
import torch
from torch.nn import Threshold
import hydra
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataloader.ufs_data import UFSegmentationDataset
from omegaconf import DictConfig
from torchvision import transforms
from runtime.utils import get_new_bbox_coordinates, post_process_output_with_bbox


def thresh(pred):
    pred = pred.squeeze()
    pred = torch.where(pred > 0.5, 1, 0)
    return pred


# Loading config file is taken care by hydra
@hydra.main(config_name="../config")
def test_model(cfg: DictConfig):
    """
    Function to run the model inference on the test data.
    """
    image_path = cfg["test"]["dir"]
    bbox_dir = cfg["test"]["bbox_dir"]
    result_save_path = cfg["test"]["result_path"]
    images = os.listdir(image_path)
    transform = transforms.Compose(
        [
            transforms.Resize(
                size=(cfg["data"]["width"], cfg["data"]["height"]),
                interpolation=Image.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    dataset = UFSegmentationDataset(cfg=cfg, data_root=image_path, bbox_dir=bbox_dir)
    models_dir = cfg["train"]["saved_model_path"]
    model_path = os.path.join(models_dir, "03-02-2021 16:14:18", "model.pt")
    model = torch.load(model_path, map_location="cpu").eval()
    for i, im in enumerate(images):
        with torch.no_grad():
            print(f"processing img: {i + 1}")
            img_path = os.path.join(image_path, im)
            img = dataset.pre_process_image_with_bb(img_path)
            img = transform(img)
            img = img.unsqueeze(dim=1)  # Add batch dimension
            pred = model(img)
            # Post-processing predictions with bbox.
            top_left, bottom_right = dataset.get_crop_coordinates(img_path)
            new_top_left, new_bottom_right = get_new_bbox_coordinates(
                top_left, bottom_right, dataset.width, dataset.height
            )
            pred = post_process_output_with_bbox(pred, new_top_left, new_bottom_right)
            pred = thresh(pred)
            frame = Image.fromarray((pred.numpy()* 255).astype(np.uint8))
            if not os.path.exists(os.path.join(result_save_path, "pred")):
                os.mkdir(os.path.join(result_save_path, "pred"))
            frame.save(os.path.join(result_save_path, "pred", im[:-4] + ".png"))

if __name__ == "__main__":
    test_model()
