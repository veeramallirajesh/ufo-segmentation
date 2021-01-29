# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

from segmentation_models_pytorch import DeepLabV3
from typing import Mapping
from runtime.utils import get_value

# MobilenetV2 encoder and DeepLabV3 decoder for segmentation
class PspNetModel:
    def __init__(self, cfg: Mapping = None):
        self.model = DeepLabV3(
            encoder_name=get_value(cfg, ["model", "encoder_name"], "mobilenet_v2"),
            in_channels=get_value(cfg, ["model", "in_channels"], 1),
            encoder_weights=get_value(cfg, ["model", "encoder_weights"], "imagenet"),
            activation=get_value(cfg, ["model", "activation"], "identity"),
        )  # could be "softmax2d" for mutliclass or 'sigmoid' for binary

    def __repr__(self):
        return "DeepLab3 MODEL"
