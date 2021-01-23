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
from typing import Any


class Compose(object):
    """
    Compose different transformations, for joint transforms of both image and target.

    Based on the implementation in the vision repository of pytorch.
    https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py

    Attributes
    ----------
    transforms : :obj: `list` of :obj: 'Transform'
        List of transformation objects which take in image and target and transform
        them as needed.
    """

    def __init__(self, transforms):
        """
        Instantiates the Compose transform object.

        Parameters
        ----------
        transforms : :obj: `list` of :obj: 'Transform'
            List of transformation objects which take in image and target and transform
            them as needed.
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """
        Returns the transformed image and target through each of the transforms in
        self.transforms.

        Parameters
        ----------
        image : PIL Image
            PIL image to be transformed
        target : PIL Image
            PIL target to be transformed

        Returns
        -------
        :obj: `(Image, Image)` or :obj: `(torch.Tensor, torch.Tensor)`
            2-tuple of image and target, either as PIL images or torch tensors based
            on the transformations.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def append(self, transform: Any):
        """
        Appends a transform to the the object, by appending to the transforms attribute.

        Parameters
        ----------
        transform : 'Transform' like object
            A transform like object to append to transforms.
        """
        self.transforms.append(transform)


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
