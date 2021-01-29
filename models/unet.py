# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
from segmentation_models_pytorch import Unet
from typing import Mapping
from runtime.utils import get_value
import collections

# model = Unet(encoder_name="mobilenet_v2",
#              in_channels=1,
#              encoder_weights="imagenet",
#              activation="identity")

# MobilenetV2 encoder and UNET decoder for segmentation
class UnetModel:
    """
    Unet is a fully convolution neural network for image semantic segmentation.
    Consist of encoder and decoder parts connected with skip connections.
    Encoder extract features of different spatial resolution (skip connections) which are used by decoder \
    to define accurate segmentation mask. Use concatenation for fusing decoder blocks with skip connections.
    """

    def __init__(self, cfg: Mapping = None):
        self.model = Unet(
            encoder_name=get_value(cfg, ["model", "encoder_name"], "mobilenet_v2"),
            in_channels=get_value(cfg, ["model", "in_channels"], 1),
            encoder_weights=get_value(cfg, ["model", "encoder_weights"], "imagenet"),
            activation=get_value(cfg, ["model", "activation"], "identity"),
        )

    def __repr__(self):
        return "UNET MODEL"
