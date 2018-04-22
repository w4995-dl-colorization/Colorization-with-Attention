import tensorflow as tf
import cv2
import numpy as np
import slim_vgg

import resize

#demo input pic
folder = "result_images/dog2/"
img = "rsz_n02105412_5057.JPEG"
heatmap = 3
sizes = [56, 28, 7]
pic = cv2.imread(folder+img)


inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
model, end_points = slim_vgg.vgg_16(inputs)
hm = end_points['hm'+str(heatmap)] # heatmap tensor


saver = tf.train.Saver()

#tf global session
sess = tf.Session()


### TF resize not running GPU, too slow compared to the CPU Pool version
## imgs_224 = tf.image.resize_images(imgs, (224, 224, 3))
#print(imgs_224)


## Original code

res_pic = cv2.resize(pic,(224, 224), interpolation=cv2.INTER_AREA)

with tf.Session() as sess:
  saver.restore(sess, "models/vgg16.ckpt")
  attention_hm = sess.run(hm, feed_dict={inputs: [res_pic]})

  size = sizes[heatmap-1]
  res_hm = np.reshape(attention_hm, (size, size)) # 56 or 28 or 7

  # normalize result heatmap
  res_hm /= np.sum(res_hm)

  #np.set_printoptions(threshold=np.nan)
  norm_hm = cv2.normalize(src=res_hm, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
  resized_norm_hm = cv2.resize(norm_hm, (256, 256)) # interpolation=cv2.INTER_AREA
  jet_hm = cv2.applyColorMap(resized_norm_hm, cv2.COLORMAP_JET)

  # superimposed
  gray_pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
  gray_rgb_pic = cv2.cvtColor(gray_pic, cv2.COLOR_GRAY2RGB)
  output = cv2.addWeighted(gray_rgb_pic, 0.3, jet_hm, 0.7, 0)

  cv2.imwrite(folder+"heat_"+str(heatmap)+"_"+img, output)
