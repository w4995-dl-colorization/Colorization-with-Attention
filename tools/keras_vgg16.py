#please run in git root/outside tools/
#e.g. python tools/keras_vgg16.py
import tensorflow as tf

model = tf.keras.applications.VGG16(include_top=False, input_shape=(256, 256, 3))

tf.keras.models.save_model(model, "models/vgg_model.h5")