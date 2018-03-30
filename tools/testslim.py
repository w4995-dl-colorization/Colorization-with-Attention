import tensorflow as tf
import cv2
import numpy as np
import slim_vgg

inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
model, end_points = slim_vgg.vgg_16(inputs)
hm1 = end_points['hm1'] #heatmap tensor
hm2 = end_points['hm2']
hm3 = end_points['hm3']

saver = tf.train.Saver()

#demo input pic
pic = cv2.imread("color_.jpg")
res_pic = cv2.resize(pic,(224, 224), interpolation=cv2.INTER_AREA)
with tf.Session() as sess:
  saver.restore(sess, "models/vgg16.ckpt")
  attention_hm = sess.run(hm1, feed_dict={inputs: [res_pic]})
  res_hm = np.reshape(attention_hm, (56, 56))

  # normalize result heatmap
  sum_hm = np.sum(res_hm, axis=(0, 1))
  # print(sum_hm)
  res_hm /= sum_hm # 0 - 1
  np.set_printoptions(threshold=np.nan)
  norm_hm = cv2.normalize(src=res_hm, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
  norm_hm = 255 - norm_hm
  resized_norm_hm = cv2.resize(norm_hm, (256, 256)) # interpolation=cv2.INTER_AREA
  jet_hm = cv2.applyColorMap(resized_norm_hm, cv2.COLORMAP_JET)

  # superimposed
  gray_pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
  gray_rgb_pic = cv2.cvtColor(gray_pic, cv2.COLOR_GRAY2RGB)
  print(jet_hm.shape, gray_rgb_pic.shape)
  output = cv2.addWeighted(gray_rgb_pic, 0.3, jet_hm, 0.7, 0)
  cv2.imwrite("heatmap.png", output)