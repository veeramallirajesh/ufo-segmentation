# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

import os
import sys
import math

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
import collections
from typing import List, Any, Mapping, Tuple

smooth_const = 1e-6


def move_dim(t1, source, target):
    assert source != target
    offset = 0

    if 0 < source < target:
        offset = 1

    return (
        t1.unsqueeze(target).transpose(target, offset + source).squeeze(offset + source)
    )


def iou_pytorch(outputs, labels, smooth=smooth_const):
    intersection = (
        (outputs & labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def iou_loss(outputs, labels, smooth=smooth_const):
    intersection = (outputs * labels).sum((1, 2))
    union = ((outputs + labels) - (outputs * labels)).sum((1, 2))

    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()


def multiclass_iou_score(outputs, labels, n_classes=2, n_dims=2, smooth=smooth_const):
    axis = tuple(range(2, 2 + n_dims))
    labels = move_dim(one_hot(labels, num_classes=n_classes), -1, 1)
    intersection = (outputs * labels).sum(axis)
    union = ((outputs + labels) - (outputs * labels)).sum(axis)

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def multiclass_iou_loss(outputs, labels, n_classes=2, n_dims=2, smooth=smooth_const):
    return 1 - multiclass_iou_score(outputs, labels, n_classes, n_dims, smooth)


def multiclass_dice_score(outputs, labels, n_classes=2, n_dims=2, smooth=smooth_const):
    axis = tuple(range(2, 2 + n_dims))
    labels_oh = move_dim(one_hot(labels, num_classes=n_classes), -1, 1).float()
    numerator = 2 * (outputs * labels_oh).sum(axis)
    denominator = (torch.square(outputs) + torch.square(labels_oh)).sum(
        axis
    )  # no square?

    frac = (numerator + smooth) / (denominator + smooth)
    return frac.mean()


def weighted_multiclass_dice_loss(outputs, labels, n_classes=2):
    dice = 0.0
    labels_oh = move_dim(one_hot(labels, num_classes=n_classes), -1, 1).float()

    for index in range(n_classes):
        dice += (1 / torch.square(labels_oh[:, index].sum() + 1)) * dice_score(
            outputs[:, index], labels_oh[:, index]
        )
    return 1 - dice


def multiclass_dice_loss(outputs, labels, n_classes=2, n_dims=2, smooth=smooth_const):
    return 1 - multiclass_dice_score(outputs, labels, n_classes, n_dims, smooth)


def dice_score(outputs, labels, smooth=smooth_const):
    numerator = 2 * (outputs * labels).sum(dim=(1, 2))
    denominator = (torch.square(outputs) + torch.square(labels)).sum(
        dim=(1, 2)
    )  # NO square?

    frac = (numerator + smooth) / (denominator + smooth)
    return frac.mean()


def dice_loss(
    outputs: torch.Tensor, labels: torch.Tensor, smooth=smooth_const
) -> torch.Tensor:
    return 1 - dice_score(outputs, labels, smooth)


# ALPHA < 0.5 penalises FP more, > 0.5 penalises FN more
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, ce_ratio=0.5):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio

    def forward(self, inputs, targets, reduce_samples=True, smooth=1):
        e = 1e-07

        # flatten label and prediction tensors
        inputs = inputs.flatten(start_dim=1, end_dim=-1)
        targets = targets.flatten(start_dim=1, end_dim=-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, e, 1.0 - e)
        out = -(
            self.alpha
            * (
                (targets * torch.log(inputs))
                + ((1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs))
            )
        )
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)

        if reduce_samples:
            return combo.mean()
        else:
            return combo


def parse_args():
    train = True
    debug = False
    train_on_gpu = torch.cuda.is_available()

    if len(sys.argv) > 2:
        if sys.argv[2] == "train":
            pass
        elif sys.argv[2] == "eval":
            train = False
        else:
            raise RuntimeError(
                f"{sys.argv[2]} cannot be understood as either train/eval"
            )

    if len(sys.argv) > 3:
        if sys.argv[3] == "run":
            pass
        elif sys.argv[3] == "debug":
            debug = True
        else:
            raise RuntimeError(
                f"{sys.argv[3]} cannot be understood as either run/debug"
            )

    return train, debug, train_on_gpu


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title("Original mask", fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title("Transformed image", fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title("Transformed mask", fontsize=fontsize)


def get_value(dictionary: Mapping, keys: List[str], default: Any) -> Any:
    """
    Recursive get function for dictionaries.

    For example if a dict is structured as follows, dictionary["a"] = {"b": 8},
    then to get the value of dictionary["a"]["b"] safely without raising KeyError,
    ```get_from_dict(dictionary, ["a", "b"], default)``` could be used where default
    is the value expected if any of the keys are not actually present.

    Parameters
    ----------
    dictionary : dict
        Dictionary to get the values from.
    keys : List[str]
        List of strings, where each element is a key in the dictionary.
    default : Any
        Default value expected in case any of the keys not present.

    Returns
    -------
    Any
        If all the keys exist, then the value corresponding to the last key is returned.
        Else, the default value is returned.

    """
    if not keys:
        return default
    for key in keys[:-1]:
        dictionary = dictionary.get(key, {})
        if not dictionary or not isinstance(dictionary, collections.Mapping):
            return default
    return dictionary.get(keys[-1], default)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert (
            predict.shape[0] == target.shape[0]
        ), "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


def get_new_bbox_coordinates(top_left: Tuple, bottom_right: Tuple, w, h) -> Tuple:
    half_x = (512 - (bottom_right[0] - top_left[0])) / 2
    if half_x < 0:
        half_x = 0
    half_y = (512 - (bottom_right[1] - top_left[1])) / 2
    if half_y < 0:
        half_y = 0
    # x2 = 512 - math.ceil(half_x)
    # x1 = 0 + math.floor(half_x)
    # y2 = 512 - math.ceil(half_y)
    # y1 = 0 + math.floor(half_y)
    # Pixels that were modified during pre-processing
    # subtracted 20 pixels to make sure areas within bbox are not deactivated
    new_top_left = ((0 + math.ceil(half_x) - 20), (0 + math.ceil(half_y) - 20))
    # Added 20 pixels to make sure areas within bbox are not deactivated
    new_bottom_right = (
        (512 - math.floor(half_x) + 20),
        (512 - math.floor(half_y) + 20),
    )

    return new_top_left, new_bottom_right


def post_process_output_with_bbox(top_left: Tuple, bottom_right: Tuple):
    pass
