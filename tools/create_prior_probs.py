'''
Calculate probability of each bin appearing in the training data
Currently, each bin has to appear at least once in the training
images in order to avoid breaking the training procedure.
'''
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage import color
import random
from skimage.transform import resize

# Read into file paths and randomly shuffle them
filename_lists = []
with open('data/train.txt', 'r') as lists_f:
    for img_f in lists_f:
        img_f = img_f.strip()
        filename_lists.append(img_f)
random.shuffle(filename_lists)

# Load 313 bins on the gamut
# points (313, 2)
points = np.load('resources/pts_in_hull.npy')
points = points.astype(np.float64)
# points (1, 313, 2)
points = points[None, :, :]

# probs (313,)
probs = np.zeros((points.shape[1]), dtype=np.float64)
num = 0

# construct graph
# in_data (50176, 2)
in_data = tf.placeholder(tf.float64, [None, 2])

# in_data (50176, 1, 2)
expand_in_data = tf.expand_dims(in_data, axis=1)

# Calculate l2 distance between the pixel and each bin by broadcasting
# distance (50176, 313)
distance = tf.reduce_sum(tf.square(expand_in_data - points), axis=2)

# Find the index of the bin that is closest
# index (50176, 1)
index = tf.argmin(distance, axis=1)

# Configure/Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

for num, img_f in enumerate(filename_lists):
    img = imread(img_f)
    img = resize(img, (256, 256), preserve_range=True)

    # Make sure the image is rgb format
    if len(img.shape) != 3 or img.shape[2] != 3:
        continue
    img_lab = color.rgb2lab(img)
    img_lab = img_lab.reshape((-1, 3))

    # img_ab (256^2, 2)
    img_ab = img_lab[:, 1:]

    # Run session in order to get the nearest bin for each pixel
    # nd_index (50176, 1)
    nd_index = sess.run(index, feed_dict={in_data: img_ab})
    for i in nd_index:
        i = int(i)
        probs[i] += 1
    print(num)

sess.close()

# Calculate probability of each bin
probs = probs / np.sum(probs)

# Save the result
np.save('probs', probs)
