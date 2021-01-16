# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import cv2
import numpy as np
import PIL
import matplotlib.pyplot as plt
from PIL import Image

# hydra loads config.yaml file automatically when the application is run.
@hydra.main(config_name="../config")
def draw_bounding_box(cfg : DictConfig) -> None:
    # print configurations
    print(OmegaConf.to_yaml(cfg))
    # Data directory path
    data_dir = cfg['data']['dir']
    # Images path
    images_path = str(Path(data_dir, cfg['data']['images']))
    # Bounding box labels path
    labels_path = str(Path(data_dir, cfg['data']['labels']))
    # Directory to save overlaid images
    output_dir = str(Path(data_dir, cfg['data']['output']))
    images_list = [(images_path + f) for f in sorted(os.listdir(images_path)) if os.path.isfile(os.path.join(images_path, f))]
    print(f"Total size of the dataset is {len(images_list)}")

    # Create directory if directory does't exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image in enumerate(sorted(os.listdir(images_path))):
        label = Path(image[:-3] + "txt")
        if not os.path.exists(os.path.join(labels_path, label)):
            print(f"Label file does not exist {label}")
        with open(Path(labels_path, label)) as f:
            objects = []
            for line in f:
                objects.append(line.strip().split())
                im = cv2.imread(os.path.join(images_path , image), cv2.IMREAD_GRAYSCALE)
                for object in  objects:
                    # Top left coordinates of a bounding box
                    (x1, y1) = (int(object[1]), int(object[2]))
                    # Bottom right coordinates of a bounding box
                    (x2, y2) = (int(object[3]), int(object[4]))
                    # Draw rectangle with the give coordinates over the image
                    cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 2)
                os.chdir(output_dir)
                cv2.imwrite("overlay_" + image,im)
    print("done")


if __name__ == "__main__":
    draw_bounding_box()