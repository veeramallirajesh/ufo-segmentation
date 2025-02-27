# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

import os
import torch
from torch.nn import Threshold
import hydra
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from dataloader.ufs_data import UFSegmentationDataset
from omegaconf import DictConfig
from torchvision import transforms
from typing import Tuple, Optional
from runtime.utils import get_new_bbox_coordinates, PostProcessOutputWithBbox
from utils.export_to_torchscript import load_torchscript_model


def thresh(pred: torch.Tensor):
    """
    Apply thresholding on the predictions from the model

    Args:
        pred (torch.Tensor): Predicted tensor from the model (sigmoid output).

    Returns: Tensor after thresholding.

    """
    pred = pred.squeeze()
    pred = torch.where(pred > 0.5, 1, 0)
    return pred


# Check if the co-ordinates are the result crop enlargement without and resizing
def check_coordinates(cfg: DictConfig, top: Tuple, bottom: Tuple) -> bool:
    """
    Module to check if the image is resized or processed with bounding box.

    Args:
        cfg (DictConfig): Configuration dictionary
        top (Tuple): top-left coordinates of the bounding box
        bottom (Tuple):  bottom-right coordinates of the bounding box

    Returns: Boolean value

    """
    height, width = (
        cfg["test"]["height"],
        cfg["test"]["width"],
    )  # height and width of all the test images.
    if (top[0] and top[1]) != 0 and (bottom[0] < width) and (bottom[1] < height):
        if ((bottom[0] - top[0]) == 512) and ((bottom[1] - top[1]) == 512):
            return True
    return False


# Loading config file is taken care by hydra
@hydra.main(config_name="../config")
def test_model(cfg: DictConfig):
    """
    Function to run the model inference on the unknown test data.

    Args:
        cfg (DictConfig): Configuration dictionary from hydras
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
    model_path = os.path.join(
        models_dir, "03-02-2021 16:14:18", "model.pt"
    )  # PspNet model
    model = torch.load(model_path, map_location="cpu").eval()
    resized = 0
    pp = PostProcessOutputWithBbox()
    for i, im in enumerate(tqdm(images, colour='green')):
        # if im == "17_07_2016_07h11m04s892ms.jpg":
        #     print("Bug case")
        with torch.no_grad():
            print(f"processing img: {i + 1}")
            img_path = os.path.join(image_path, im)
            img, (x1, y1), (x2, y2) = dataset.pre_process_image_with_bb(img_path=img_path)
            points = np.array((x1, y1, x2, y2))  # Numpy Array of coordinate points
            simple_case = check_coordinates(cfg, (x1, y1), (x2, y2))
            img = transform(img)
            img = img.unsqueeze(dim=1)  # Add batch dimension
            pred = model(img)
            pred = thresh(pred).numpy().astype(np.uint8)
            if simple_case:
                # Post-processing predictions with bbox.
                top_left, bottom_right = dataset.get_crop_coordinates(img_path)
                new_top_left, new_bottom_right = get_new_bbox_coordinates(
                    top_left, bottom_right, dataset.width, dataset.height
                )
                pred = pp.post_process_output_with_bbox(
                    pred, new_top_left, new_bottom_right
                )
                frame = Image.fromarray((pred * 255).astype(np.uint8))
                os.makedirs(os.path.join(result_save_path, 'simple', "pred"), exist_ok=True)
                frame.save(os.path.join(result_save_path, 'simple', "pred", im[:-4] + ".png"))
                # Saving the bbox coordinates to a txt file.
                os.makedirs(os.path.join(result_save_path, 'simple', "bbox"), exist_ok=True)
                save_path = os.path.join(result_save_path, 'simple', "bbox", im[:-4] + ".txt")
                points.tofile(save_path, sep=" ")  # numpy save to file

            else: # Resized scenario
                print("Resized case")
                resized += 1
                frame = Image.fromarray((pred * 255).astype(np.uint8)).resize(
                    size=((x2 - x1), (y2 - y1)), resample=Image.BILINEAR
                )
                os.makedirs(
                    os.path.join(result_save_path, "resized", "pred"), exist_ok=True
                )
                frame.save(
                    os.path.join(result_save_path, "resized", "pred", im[:-4] + ".png")
                )
                # Saving the bbox coordinates to a txt file.
                os.makedirs(
                    os.path.join(result_save_path, "resized", "bbox"), exist_ok=True
                )
                save_path = os.path.join(
                    result_save_path, "resized", "bbox", im[:-4] + ".txt"
                )
                points.tofile(save_path, sep=" ")  # numpy save to file
    print(f"Number of resized images are {resized}")


if __name__ == "__main__":
    test_model()
