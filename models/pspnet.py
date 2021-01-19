# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

from segmentation_models_pytorch import PSPNet
from typing import Mapping

# model = PSPNet(encoder_name="mobilenet_v2",
#                in_channels=1,
#                encoder_weights="imagenet",
#                activation="identity")

class PspNetModel:
    def __init__(self, cfg: Mapping = None):
        self.model = PSPNet(encoder_name=cfg['model']['encoder_name'],
             in_channels=cfg['model']['in_channels'],
             encoder_weights=cfg['model']['encoder_weights'],
             activation=cfg['model']['activation'])# could be "softmax2d" for mutliclass or 'sigmoid' for binary

    def __repr__(self):
        return "PSPNET MODEL"

