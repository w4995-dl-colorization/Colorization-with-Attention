# at root, run this script by python tools/extract_imagenet.py
# relative path for imagenet dir: data/ILSVRC2012_img_train/*.tar
# 1000 tars for each class in total
# each tar can be extracted to get 1300 original size imgs

### USAGE: This script scans through all classes in imagenet and open a sample img in each tar to display them
###        Use mouse left click to append the name of class of that img to a txt file to save for class choices

import cv2
import os
import glob
import ntpath
import skimage.io
import numpy as np
import random
import tarfile
import re

#config path
path = os.path.join(os.getcwd(), 'data', 'ILSVRC2012_img_train')

list_of_tar = glob.glob(path+"/*.tar")

print("In total %d classes of tars scanned" % len(list_of_tar))

#config no. of classes
num_class = 1000 # whole set to select from

list_of_chosen = random.sample(list_of_tar, num_class)

for tarname in list_of_chosen: