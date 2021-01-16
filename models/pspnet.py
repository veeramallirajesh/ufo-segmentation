# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

from segmentation_models_pytorch import PSPNet

model = PSPNet(encoder_name="mobilenet_v2",
               in_channels=3,
               encoder_weights="imagenet",
               activation="sigmoid")

