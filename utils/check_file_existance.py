# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

"""
To check if bbox label files exist for all the test images.
"""
import os

path = "/home/rveeramalli/ufo-segmentation/data/cod_18_07_2016/images"
gt_path = "/home/rveeramalli/ufo-segmentation/data/cod_18_07_2016/ground-truth"

files = os.listdir(path)

count = 0
for file in files:
    if os.path.exists(os.path.join(gt_path, file[:-3] + "txt")):
        count += 1
    else:
        print(f"bbox file does not exist for file:{file}")
print(f"Number of files:{count}")
