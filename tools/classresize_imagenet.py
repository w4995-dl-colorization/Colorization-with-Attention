# at root, run this script by python tools/extract_imagenet.py
# relative path for imagenet dir: data/ILSVRC2012_img_train/*.tar
# 1000 tars for each class in total
# each tar can be extracted to get 1300 original size imgs

# USAGE: This script opens the manully selected images folder, figure out the classes and open each class folder and output the resized 256by256by3

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
import resize

# config path
path = os.path.join(os.getcwd(), 'data', 'ILSVRC2012_img_train')

# config manual sample folder path
manual_path = os.path.join(path, 'manualsample10')

list_of_tar = glob.glob(path+"/*.tar")

list_of_class = glob.glob(manual_path+"/*")

# output e.g. n07753592 , class name
list_of_chosen_dir = list(
    map(lambda x: ntpath.basename(x).split("_")[0], list_of_class))

print("In total %d classes of tars scanned" % len(list_of_tar))

print("%d Classes found to be extracted." % len(list_of_chosen_dir))

for dirname in list_of_chosen_dir:
    print("resizing", dirname)
    class_imgs_path = glob.glob(path+"/"+dirname+"/*")
    imgs = skimage.io.imread_collection(class_imgs_path)
    resized_imgs = resize.apply(imgs)
    for img, class_img_path in list(zip(resized_imgs, class_imgs_path)):
        bname = ntpath.basename(class_img_path)
        skimage.io.imsave(path+"/output/rsz_"+bname, img)