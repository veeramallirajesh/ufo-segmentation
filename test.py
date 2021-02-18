# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""


import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from segmentation_models_pytorch import PSPNet
from dataloader.ufs_data import KittiDataset
from torch.utils.data import DataLoader

# from runtime.training import ModelTrainer
import tqdm
from torch.optim import Adam
from runtime.utils import iou_loss, dice_loss

im = Image.open(
    "/home/rveeramalli/ufo-segmentation/data_v1/images/19_07_2016_04h30m48s541ms.jpg"
)

# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Lambda(lambda x: torch.cat([x, x, x], 0))])
transform = transforms.Compose([transforms.ToTensor()])
im = transform(im)
im = torch.unsqueeze(im, 0)

model = PSPNet(
    encoder_name="mobilenet_v2",
    in_channels=1,
    encoder_weights="imagenet",
    activation="sigmoid",
)
model = model.cuda() if torch.cuda.is_available() else model
pred = model(im)
pred = pred.detach().numpy().squeeze()
print(pred.shape)
# print(im.shape)
# print(torch.__version__)
# print(type(im))
# plt.imshow(pred)
# plt.show()

dataset = KittiDataset(
    data_root="/Users/kavya/Documents/Master-Thesis/Underwater-Segmentation/datasets/data_semantics/",
    mode="train",
)
print(dataset)

train_loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=False)

# for img, mask in train_loader:
#     print(img.shape, mask.shape)
device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"device used is {device}")
# n_epochs = 10
# current_batch = 0
# optimizer = Adam(model.parameters())
# for epoch in range(n_epochs):
#     for img, mask in train_loader:
#         current_batch += 1
#         model.train()
#         pred = model(img)
#         loss = iou_loss(pred, mask)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         # print(loss.item())
#         print(f"{epoch + 1}-{current_batch}: {loss.item()}")
#     current_batch = 0

# trainer = ModelTrainer()
