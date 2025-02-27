# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

from segmentation_models_pytorch import PSPNet
from typing import Mapping
from runtime.utils import get_value

# MobilenetV2 encoder and PSPNET decoder for segmentation
class PspNetModel:
    """
    PSPNet is a fully convolution neural network for image semantic segmentation. Consist of encoder and Spatial Pyramid (decoder).
    Spatial Pyramid build on top of encoder and does not use “fine-features” (features of high spatial resolution).
    PSPNet can be used for multiclass segmentation of high resolution images,
    however it is not good for detecting small objects and producing accurate, pixel-level mask.
    """

    def __init__(self, cfg: Mapping = None):
        self.model = PSPNet(
            encoder_name=get_value(cfg, ["model", "encoder_name"], "mobilenet_v2"),
            in_channels=get_value(cfg, ["model", "in_channels"], 1),
            encoder_weights=get_value(cfg, ["model", "encoder_weights"], "imagenet"),
            activation=get_value(cfg, ["model", "activation"], "identity"),
        )  # could be "softmax2d" for mutliclass or 'sigmoid' for binary

    def __repr__(self):
        return "PSPNET MODEL"
