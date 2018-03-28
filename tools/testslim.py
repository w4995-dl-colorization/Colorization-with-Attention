import tensorflow as tf

import slim_vgg

inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
model = slim_vgg.vgg_16(inputs)

#saver = tf.train.Saver()

#with tf.Session() as sess:
  # Restore variables from disk.
  #saver.restore(sess, "models/vgg16.ckpt")