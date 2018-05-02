## resize implemented only, -> 256*256
from multiprocessing import Pool
import cv2
import os
import glob
import ntpath
import skimage.io
import numpy as np

# path of dataset
path = os.path.join(os.getcwd(), 'data', 'imagenet_animal')
pathout = os.path.join(os.getcwd(), 'data', 'imagenet_animal_resized')

def resize(img, sz=(256, 256)):
    return cv2.resize(img, sz) #, interpolation=cv2.INTER_AREA

def apply(imgs, sz=(256, 256)):
    p = Pool(6)
    imgs = p.map(resize, imgs)
    return imgs

def readin():
    # all paths of pics
    pnames = glob.glob(path + '/*.JPEG')
    imgs = skimage.io.imread_collection(pnames)
    return pnames, imgs

def writeout(pnames, imgs):
    idx  = 0
    for pname in pnames:
        #base name
        bname = ntpath.basename(pname)
        skimage.io.imsave(pathout+"/"+bname, imgs[idx])
        idx+=1

# pnames, imgs = readin()
# imgs = apply(imgs)
# print(np.array(imgs[0]).shape)
# writeout(pnames, imgs)