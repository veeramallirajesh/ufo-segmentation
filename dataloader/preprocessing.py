# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

import cv2
import numpy as np
import torch
from albumentations import Resize
from PIL import Image
from torchvision.transforms import functional as F


def strip_dict(dict1, keys_to_remove):
    additional_data = dict(dict1)
    for key in keys_to_remove:
        if key in additional_data:
            del additional_data[key]

    return additional_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ApplyPreprocessing:
    def __init__(self, preprocess_f):
        self.preprocess_f = preprocess_f

    def __call__(self, sample):
        image = np.asarray(sample.convert("RGB"))
        img = self.preprocess_f(image)
        img = Image.fromarray(image).convert("L")
        return img


class ToTensor(object):
    """
    ToTensor transformation for the joint transform for both the image and target

    Based on the implementation in the vision repository of pytorch.
    https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
    """

    def __call__(self, image: Image, target: Image) -> (torch.Tensor, torch.Tensor):
        """
        Returns a tuple of tensors corresponding to the image and target

        Parameters
        ----------
        image : Image
            PIL image to transform to a tensor
        target : Image
            PIL image to transform to a tensor

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Tuple of tensors corresponding to (image, target)
        """
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target
