# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

model_path = "/home/rveeramalli/ufo-segmentation/src/saved_models/10-01-2021 22:08:15/model_epoch_90.pt"#23-12-2020 16:58:55/model_epoch_25.pt
img_path = "/home/rveeramalli/ufo-segmentation/data_v1/images/19_07_2016_04h55m52s713ms.jpg"

img = Image.open(img_path).convert('L')
img = img.resize(size=(720, 360), resample=Image.NEAREST)
transforms = transforms.Compose([transforms.ToTensor()])
img = transforms(img)
img = img.unsqueeze(dim=0)
with torch.no_grad():
    model = torch.load(model_path, map_location="cpu")
    pred = model(img)
print("done")
