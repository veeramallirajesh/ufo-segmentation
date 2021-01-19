# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
from segmentation_models_pytorch import Unet
from typing import Mapping

# model = Unet(encoder_name="mobilenet_v2",
#              in_channels=1,
#              encoder_weights="imagenet",
#              activation="identity")

class UnetModel:
    def __init__(self, cfg: Mapping = None):
        self.model = Unet(encoder_name=cfg['model']['encoder_name'],
             in_channels=cfg['model']['in_channels'],
             encoder_weights=cfg['model']['encoder_weights'],
             activation=cfg['model']['activation'])

    def __repr__(self):
        return "UNET MODEL"
