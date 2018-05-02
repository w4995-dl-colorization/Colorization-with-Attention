# at root, run this script by python tools/classextract_imagenet.py
# relative path for imagenet dir: data/ILSVRC2012_img_train/*.tar
# 1000 tars for each class in total
# each tar can be extracted to get 1300 original size imgs

# USAGE: This script opens the manually selected images folder, (images name like n02009229_4994.JPEG), extract images with the same classes from 1000 class tar to each class folder

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

# config manual sample folder path
manual_path = os.path.join(path, 'manualsample10')

list_of_tar = glob.glob(path+"/*.tar")

list_of_class = glob.glob(manual_path+"/*")

# output e.g. n07753592 , class name
list_of_chosen_dir = list(
    map(lambda x: ntpath.basename(x).split("_")[0], list_of_class))

# output e.g. n07753592.tar , file name
list_of_chosen_tar = list(map(lambda x: x+".tar", list_of_chosen_dir))

print("In total %d classes of tars scanned" % len(list_of_tar))

list_of_chosen = list(zip(list_of_chosen_dir, list_of_chosen_tar))
print("%d Classes found to be extracted." % len(list_of_chosen))

for dirname, tarname in list_of_chosen:
    print("extracting", dirname, path+"/"+tarname)
    tar = tarfile.open(path+"/"+tarname)
    tar.extractall(path=path+"/"+dirname)
    tar.close()
