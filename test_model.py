# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

model_path = "/home/rveeramalli/ufo-segmentation/src/saved_models/23-01-2021 12:32:00/model.pt"  # 23-12-2020 16:58:55/model_epoch_25.pt
img_path = (
    "/home/rveeramalli/ufo-segmentation/data_v1/images/19_07_2016_05h10m52s404ms.jpg" # 19_07_2016_04h55m52s713ms.jpg
)

img = Image.open(img_path).convert("L")
img = img.resize(size=(512, 512), resample=Image.NEAREST)
transforms = transforms.Compose([transforms.ToTensor()])
img = transforms(img)
img = img.unsqueeze(dim=0)
with torch.no_grad():
    model = torch.load(model_path, map_location="cpu").eval()
    pred = model(img).numpy().squeeze()
    pred = np.where(pred > 0.5, 1, 0) # Thresholding the predictions
    # plt.imsave("/home/rveeramalli/ufo-segmentation/data_v1/test/6_pred.png", pred.numpy().squeeze(), cmap='gray')
print("done")

# if __name__ == "__main__":
#     plot_inference()
