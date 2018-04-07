import numpy as np
import tensorflow as tf
from utils import decode
from net import Net
from net_att import Net_att
from net_densenet import DenseNet
from skimage.io import imsave
import random
import cv2
import re

from evaluation.evaluate_loss import RMSE_lab, PSNR_rgb, saturation_hsv

reg = re.compile(".*/(.*jpg)")

# Select from [no_att, att, end_to_end]
model = 'end_to_end'

# If save the colorized images
SAVE_IMG = False

# Read into images
folder_path = 'data/Opencountry/'
input_file = 'data/test.txt'
# img_names = ['art582.jpg', 'cdmc109.jpg', 'cdmc276.jpg', 'cdmc354.jpg', 'nat190.jpg']
img_names = []
# Select the image names
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        img_name = reg.search(line).group(1)
        img_names.append(img_name)

img_num = len(img_names)
batch_size = 5
assert batch_size <= img_num
remainder = img_num % batch_size


# We add unnecessary files in order to guarantee that each batch has the same
# number of samples
if remainder != 0:
    pad_num = batch_size - remainder
    redundant_img_names = [ img_names[i] for i in random.sample(range(img_num), pad_num) ]
    img_names += redundant_img_names
    img_num += pad_num

height = 256
width = 256

img_list = []

## img collection
img_col = []

for img_name in img_names:
    img = cv2.imread(folder_path+img_name)
    #save ref
    img_col.append(img)

    # Convert image from rgb to gray
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if SAVE_IMG:
        imsave('results/gray_'+img_name, img)

    # Preprocess the image
    img = img.reshape((1, img.shape[0], img.shape[1], 1))
    img = img.astype(dtype=np.float32) / 255.0 * 100 - 50
    img_list.append(img)

# Stack all images to get a 4d image tensor that consists of all images
all_data_l = np.vstack(img_list)

# Construct graph
training_flag = tf.placeholder(tf.bool)
if model == 'end_to_end':
    autocolor = Net_att(train=training_flag)
else:
    autocolor = Net(train=training_flag)
# autocolor = DenseNet(train=training_flag)

data_l = tf.placeholder(tf.float32, (batch_size, height, width, 1))
conv8_313 = autocolor.inference(data_l)


# Load model and run the graph
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'models/'+model+'/model.ckpt')
    reconstructed_img_list = []

    for start_ind in range(0, img_num, batch_size):
        end_ind = start_ind + batch_size
        if end_ind > img_num:
            end_ind = img_num

        batch_data_l = all_data_l[start_ind:end_ind]
        conv8_313_returned = sess.run(conv8_313, feed_dict={training_flag:False, data_l:batch_data_l})

        for i in range(batch_size):
            # Colorize w/ class rebalancing
            # reconstructed_img_rgb  : [height, width, 3], predicted colorized image
            reconstructed_img_rgb = decode(batch_data_l[i][None,:,:,:], conv8_313_returned[i][None,:,:,:], 0.38)
            reconstructed_img_list.append(reconstructed_img_rgb.astype(np.uint8))

            if SAVE_IMG:
                imsave('results/color_'+model+'_'+img_names[start_ind+i], reconstructed_img_rgb)
 
    print(reconstructed_img_list[0].dtype, img_col[0].dtype)
    print(RMSE_lab(reconstructed_img_list, img_col))
    print(PSNR_rgb(reconstructed_img_list, img_col))
    print(saturation_hsv(reconstructed_img_list, img_col))

