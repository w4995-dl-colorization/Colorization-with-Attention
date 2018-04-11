# at root, run this script by python tools/extract_imagenet.py
# relative path for imagenet dir: data/ILSVRC2012_img_train/*.tar
# 1000 tars for each class in total
# each tar can be extracted to get 1300 original size imgs

# USAGE: This script scans through all classes in imagenet, extract the first image of each class tar, output them in to a folder, 1000 items in total

# matplotlib GridSpec

import cv2
import os
import glob
import ntpath
import skimage.io
import numpy as np
import random
import tarfile
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# config path
path = os.path.join(os.getcwd(), 'data', 'ILSVRC2012_img_train')

list_of_tar = glob.glob(path+"/*.tar")

print("In total %d classes of tars scanned" % len(list_of_tar))

# config no. of classes
num_class = 1000  # whole set to select from

list_of_chosen = random.sample(list_of_tar, num_class)

c = 0
for tarname in list_of_chosen:
    print("extracting ", c, " ...")
    tar = tarfile.open(tarname)
    first_member = tar.getmembers()[0]
    tar.extract(first_member, path="data/ILSVRC2012_img_train/sample1000/")
    c += 1

# img = cv2.imread("./color_.jpg")
# imgplot = plt.imshow(img)
# plt.show()
