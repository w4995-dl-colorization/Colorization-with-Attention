import cv2
from utils import preprocess, decode
import numpy as np
from skimage.io import imsave

images = [cv2.imread('data/output/rsz_n01531178_4510.JPEG')]
data_l, gt_ab_313, prior_color_weight_nongray = preprocess(np.asarray(images, dtype=np.uint8))

reconstructed_img_rgb = decode(data_l[0][None,:,:,:], gt_ab_313[0][None,:,:,:], 0.38)

# print(reconstructed_img_rgb.shape)
imsave('tmp/test2.jpg', reconstructed_img_rgb)