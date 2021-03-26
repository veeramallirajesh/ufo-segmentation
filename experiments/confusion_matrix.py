# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import plot_confusion_matrix


def confusion_matrix(mask_path: str, pred_path: str, h: int, w: int):
    """
    Module to compute TP, TN, FN, FP values.
    Args:
        mask_path (str): ground-truth mask path
        pred_path (str): predicted mask path
        h (int): height
        w (int): width

    Returns: TP, TN, FP, FN

    """
    gt = (
        np.array(
            Image.open(mask_path)
            .convert("L")
            .resize(size=(w, h), resample=Image.NEAREST)
        )
        // 255
    )
    pred = np.array(Image.open(pred_path).convert("L")) // 255
    assert gt.shape == pred.shape, "Sizes are not equal"
    TP = np.sum(pred * gt)  # True Positives
    FP = np.sum(pred * (1 - gt))  # False Positives
    FN = np.sum((1 - pred) * gt)  # False Negatives
    TN = np.sum((1 - pred) * (1 - gt))  # True Negatives

    return TP, TN, FP, FN


if __name__ == "__main__":
    gt_mask_path = f"/Users/kavya/Documents/Master-Thesis/Underwater-Segmentation/results/test_gt/18.png"
    pred_path = f"/Users/kavya/Documents/Master-Thesis/Underwater-Segmentation/results/deeplabv3/pred/18.png"
    height, width = 512, 512
    TP, TN, FP, FN = confusion_matrix(gt_mask_path, pred_path, height, width)
    print(
        f"True-Positive: {TP}, True-Negative: {TN}, False-Positive: {FP}, False-Negative: {FN}"
    )
