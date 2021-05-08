# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
import numpy as np
import os
from PIL import Image
from typing import List
from dataloader.ufs_data import UFSegmentationDataset
from dataloader.train_val_test_split import TrainValTestSplit
import matplotlib.pyplot as plt


def get_class_ratio(mask_path: str, indices: np.ndarray):
    """
    Function to measure the percentage of background and foreground pixels in the training images.
    To determine if there is a class-imbalance problem.
    """
    masks = sorted(os.listdir(mask_path))
    ratios = []
    for idx in range(len(masks)): # for idx in indices:
        img = Image.open(os.path.join(mask_path, masks[idx])).convert("L")
        img_array = np.array(img)
        # Fish pixels distributions per image
        class_fish_ratio = np.count_nonzero(img_array) / (
            img_array.shape[0] * img_array.shape[1]
        )
        ratios.append(class_fish_ratio)
    print("All masks are processed")
    print(
        f"The Mean ratio of background-foreground classes is {1-np.mean(ratios)}:{np.mean(ratios)}"
    )
    plot_pie_chart(ratios)


def plot_pie_chart(ratios: List):
    """

    Args:
        ratios (List):  List containing ratios of foreground/fish pixels to background for all the images in the set

    Returns: None

    """
    class_ratio = [1 - np.mean(ratios), np.mean(ratios)]
    plt.title("Distribution of background-foreground pixels")
    plt.pie(class_ratio, labels=["Background", "Foreground/Fish"], autopct='%.2f')
    # plt.legend()
    plt.show()

def main():
    indices_path = "/home/rveeramalli/ufo-segmentation/data_v1/indices_ufs/train.npy"
    mask_path = "/home/rveeramalli/ufo-segmentation/data_v1/processed/masks"
    indices = np.load(indices_path)
    get_class_ratio(mask_path, indices)

if __name__ == "__main__":
    main()

# 17_07_2016_07h11m04s892ms
# 17_07_2016_07h11m05s202ms
