# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

import os
import sys
import math
import numpy as np

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
    # print(f"IOU pytorch shapes ypred:{outputs.shape}, yhat:{labels.shape}")
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


def visualize(
    image: np.ndarray, mask: np.ndarray, pred: np.ndarray, save_path: str, idx: int
):
    """
    Function to plot and visualize the predictions of the model
    """
    result_path = os.path.join(save_path, "result_plots")
    # pred = np.where(pred == 255, 1, 0) # Mapping 255 to 1
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    fontsize = 18

    f, ax = plt.subplots(2, 2, figsize=(8, 8))

    # Making sure the arrays are squeezed
    ax[0, 0].imshow(image.squeeze())
    ax[0, 0].set_title("Original image", fontsize=fontsize)

    ax[1, 0].imshow(mask.squeeze())
    ax[1, 0].set_title("Original mask", fontsize=fontsize)

    ax[0, 1].imshow(pred.squeeze())
    ax[0, 1].set_title("Prediction ", fontsize=fontsize)

    ax[1, 1].imshow((mask - pred) == 0)
    ax[1, 1].set_title("Difference", fontsize=fontsize)

    plt.savefig(os.path.join(result_path, str(idx) + ".png"), format="png")
    plt.close()  # Close figure


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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def discard_features(model):
    """
    Function to discard weights of unused layers of the encoder.
    We are using encoder depth of only 3.
    """
    del model.encoder.features[7:]
    return model


def get_new_bbox_coordinates(top_left: Tuple, bottom_right: Tuple, w, h) -> Tuple:
    half_x = (w - (bottom_right[0] - top_left[0])) / 2
    if half_x < 0:
        half_x = 0
    half_y = (h - (bottom_right[1] - top_left[1])) / 2
    if half_y < 0:
        half_y = 0
    # x2 = 512 - math.ceil(half_x)
    # x1 = 0 + math.floor(half_x)
    # y2 = 512 - math.ceil(half_y)
    # y1 = 0 + math.floor(half_y)
    # Pixels that were modified during pre-processing
    # subtracted 20 pixels to make sure areas within bbox are not deactivated
    new_top_left = ((0 + math.ceil(half_x)), (0 + math.ceil(half_y)))
    # Added 20 pixels to make sure areas within bbox are not deactivated
    new_bottom_right = (
        (w - math.floor(half_x)),
        (h - math.floor(half_y)),
    )

    return new_top_left, new_bottom_right


def post_process_output_with_bbox(
    frame: np.ndarray, top_left: Tuple, bottom_right: Tuple
):
    """
    Function to make all the pixels outside the bounding box region of interest to black--0
    """
    dummy_frame = np.zeros_like(frame)
    dummy_frame[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = frame[
        top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
    ]
    # Second method of doing the same
    # frame[bottom_right[1]:, :] = 0
    # frame[:, :top_left[0]] = 0
    # frame[:top_left[1], :] = 0
    # frame[:, bottom_right[0]:] = 0
    return frame


def display_mask_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Function to overlay predicted mask on the original image with alpha blending.
    img: original image array -- here grayscale images (512x512)
    mask: predicted mask from the model (512x512).
    """
    if mask.max() == 1:
        mask *= 255  # if mask is of 0's and 1's
    if img.max() < 1:
        img = img.astype(np.float) * 150
    else:
        img = img.astype(np.float) * 0.7
    img[:, :] += 0.5 * mask  # If there is no channel dimension
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)
