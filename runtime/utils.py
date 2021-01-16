# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

import os
import sys

import torch
from torch.nn.functional import one_hot
smooth_const = 1e-6

def move_dim(t1, source, target):
    assert source != target
    offset = 0

    if 0 < source < target:
        offset = 1

    return t1.unsqueeze(target).transpose(target, offset + source).squeeze(offset + source)

def iou_pytorch(outputs, labels, smooth=smooth_const):
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
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
    denominator = (torch.square(outputs) + torch.square(labels_oh)).sum(axis)  # no square?

    frac = (numerator + smooth) / (denominator + smooth)
    return frac.mean()


def weighted_multiclass_dice_loss(outputs, labels, n_classes=2):
    dice = 0.0
    labels_oh = move_dim(one_hot(labels, num_classes=n_classes), -1, 1).float()

    for index in range(n_classes):
        dice += (1/torch.square(labels_oh[:, index].sum() + 1)) * dice_score(outputs[:, index], labels_oh[:, index])
    return 1 - dice


def multiclass_dice_loss(outputs, labels, n_classes=2, n_dims=2, smooth=smooth_const):
    return 1 - multiclass_dice_score(outputs, labels, n_classes, n_dims, smooth)


def dice_score(outputs, labels, smooth=smooth_const):
    numerator = 2 * (outputs * labels).sum((1, 2))
    denominator = (torch.square(outputs) + torch.square(labels)).sum((1, 2))  # NO square?

    frac = (numerator + smooth) / (denominator + smooth)
    return frac.mean()


def dice_loss(outputs, labels, smooth=smooth_const):
    return 1 - dice_score(outputs, labels, smooth)


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
            raise RuntimeError(f"{sys.argv[2]} cannot be understood as either train/eval")

    if len(sys.argv) > 3:
        if sys.argv[3] == "run":
            pass
        elif sys.argv[3] == "debug":
            debug = True
        else:
            raise RuntimeError(f"{sys.argv[3]} cannot be understood as either run/debug")

    return train, debug, train_on_gpu