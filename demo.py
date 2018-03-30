import numpy as np
import tensorflow as tf
from utils import decode
from net import Net
from net_densenet import DenseNet
from skimage.io import imsave
import cv2

# Read image
img = cv2.imread('data/Opencountry/land516.jpg')

# Convert image from rgb to gray
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imsave('gray_.jpg', img)

# Preprocess the image
img = img.reshape((1, img.shape[0], img.shape[1], 1))
data_l = img.astype(dtype=np.float32) / 255.0 * 100 - 50

# Construct graph
training_flag = tf.placeholder(tf.bool)

autocolor = Net(train=training_flag)
# autocolor = DenseNet(train=training_flag)

conv8_313 = autocolor.inference(data_l)

# Load model and run the graph
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'models/model.ckpt')
    conv8_313 = sess.run(conv8_313, feed_dict={training_flag:False})

# Colorize and save the image
img_rgb = decode(data_l, conv8_313, 0.38)
imsave('color_.jpg', img_rgb)
