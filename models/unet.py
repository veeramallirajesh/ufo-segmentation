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


class UnetModel:
    def __init__(self, cfg: Mapping = None):
        self.model = Unet(
            encoder_name=get_value(cfg, ["model", "encoder_name"], "mobilenet_v2"),
            in_channels=get_value(cfg, ["model", "in_channels"], 1),
            encoder_weights=get_value(cfg, ["model", "encoder_weights"], "imagenet"),
            activation=get_value(cfg, ["model", "activation"], "identity"),
        )

    def __repr__(self):
        return "UNET MODEL"
