# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

from functools import reduce
import numpy as np
from torch.utils.data import Sampler
import matplotlib.path as mp
import os


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.indices[index]


class DetectedObject:
    def __init__(self, points, bbox=None, label=None):
        self.bbox = bbox
        self.points = points
        self.label = label



def clip(a, lower, upper):
    return max(lower, min(a, upper))

# check if all the masks exists for the corresponding images
def image_target_map_check(img_dir, mask_dir):
    images = [img for img in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, img))]
    for img in images:
        # Get the name of the image without file extension
        base_name = img.split('.')[0]
        if os.path.exists(os.path.join(mask_dir, base_name + ".png")):
            continue
        else:
            # Remove image if corresponding mask is not found
            os.remove(os.path.join(img_dir, img))
    print("Number of images: {}, Number of masks: {}".format(len(os.listdir(img_dir)), len(os.listdir(mask_dir))))
if __name__=="__main__":
    img_dir = "/Users/kavya/Documents/Master-Thesis/Underwater-Segmentation/data/segmentation/simple_cod_subset/v1/images"
    mask_dir = "/Users/kavya/Documents/Master-Thesis/Underwater-Segmentation/data/segmentation/simple_cod_subset/v1/masks"
    image_target_map_check(img_dir, mask_dir)
