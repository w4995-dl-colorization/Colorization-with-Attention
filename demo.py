import numpy as np
import tensorflow as tf
from utils import decode
from net import Net
from net_att import Net_att
from net_densenet import DenseNet
from skimage.io import imsave
import random
import cv2

# Read into images
folder_path = 'data/Opencountry/'
# Select the image names
img_names = ['art582.jpg', 'cdmc109.jpg', 'cdmc276.jpg', 'cdmc354.jpg', 'nat190.jpg']
img_num = len(img_names)
batch_size = 5
remainder = img_num % 5

# We add unnecessary files in order to guarantee that each batch has the same
# number of samples
if remainder != 0:
    redundant_img_names = [ img_names[i] for i in random.sample(range(img_num), remainder) ]
    img_names += redundant_img_names
    img_num += remainder

height = 256
width = 256

img_list = []

for img_name in img_names:
    img = cv2.imread(folder_path+img_name)

    # Convert image from rgb to gray
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imsave('results/gray_'+img_name, img)

    # Preprocess the image
    img = img.reshape((1, img.shape[0], img.shape[1], 1))
    img = img.astype(dtype=np.float32) / 255.0 * 100 - 50
    img_list.append(img)

# Stack all images to get a 4d image tensor that consists of all images
all_data_l = np.vstack(img_list)

# Construct graph
training_flag = tf.placeholder(tf.bool)
autocolor = Net(train=training_flag)
# autocolor = Net_att(train=training_flag)
# autocolor = DenseNet(train=training_flag)
data_l = tf.placeholder(tf.float32, (None, height, width, 1))
conv8_313 = autocolor.inference(data_l)

# Load model and run the graph
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'models/att/model.ckpt')

    for start_ind in range(0, img_num, batch_size):
        end_ind = start_ind + batch_size
        if end_ind > img_num:
            end_ind = img_num

        batch_data_l = all_data_l[start_ind:end_ind]
        conv8_313 = sess.run(conv8_313, feed_dict={training_flag:False, data_l:batch_data_l})

        for i in range(batch_size):
            # Colorize and save the image
            img_rgb = decode(batch_data_l[i][None,:,:,:], conv8_313[i][None,:,:,:], 0.38)
            imsave('results/color_weight_att_'+img_names[start_ind+i], img_rgb)

