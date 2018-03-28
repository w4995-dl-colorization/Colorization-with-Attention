import tensorflow as tf

import slim_vgg

inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
model = slim_vgg.vgg_16(inputs)