import numpy as np
import tensorflow as tf
from utils import decode
from net import Net
from net_att import Net_att
from net_densenet import DenseNet
from skimage.io import imsave
import random
import cv2
import sys
import re

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

height = 256
width = 256

autocolor = Net(train=False)
data_l = tf.placeholder(tf.float32, (1, height, width, 1))
conv8_313 = autocolor.inference(data_l)

# Load model and run the graph
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './models/color_model.ckpt')

    

    # Saving
    export_dir = "./models/export/default/"
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map= {
            "model": tf.saved_model.signature_def_utils.predict_signature_def(
                inputs= {"x": data_l},
                outputs= {"finalnode": conv8_313})
            })
    builder.save()