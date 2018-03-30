import numpy as np
import tensorflow as tf
from ops import conv2d, deconv2d

from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
dropout_rate = 0.2


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    # The stride value does not matter
    It is global average pooling without tflearn
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)

    # return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')

class DenseNet():

    def __init__(self, train=True, common_params=None, net_params=None, growth_k=12):
        self.training = train
        self.weight_decay = 0.0
        if common_params:
            self.batch_size = int(common_params['batch_size'])
        if net_params:
            self.weight_decay = float(net_params['weight_decay'])
        self.filters = growth_k

    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def inference(self, x):
        """infer ab probability distribution of images from black-white images

        Args:
          data_l: 4-D tensor [batch_size, height, width, 1],
                  images with only L channel
        Return:
          conv8_313: 4-D tensor [batch_size, height/4, width/4, 313],
                     predicted ab probability distribution of images
        """

        # [batch_size, height/2, width/2, 24]
        x = conv2d('conv_init', x, [7, 7, 1, 2*self.filters], stride=2, wd=self.weight_decay)
        x = Batch_Normalization(x, training=self.training, scope='init_batch')
        x = Relu(x)


        # [batch_size, height/4, width/4, 12]
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')


        # [batch_size, height/8, width/8, 12]
        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        
        x = self.transition_layer(x, scope='trans_2')

        # [batch_size, height/8, width/8, 12]
        # x = self.dense_block(input_x=x, nb_layers=24, layer_name='dense_3')
        # x = self.transition_layer(x, scope='trans_3')


        # [batch_size, height/8, width/8, 108]
        x = self.dense_block(input_x=x, nb_layers=8, layer_name='dense_final')


        # [batch_size, height/8, width/8, 108]
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        #x = Global_Average_Pooling(x)
        print(x.get_shape())

        # [batch_size, height/4, width/4, 256]
        x = deconv2d('conv_final1', x, [4, 4, 108, 256], stride=2, wd=self.weight_decay)
        x = conv2d('conv_final2', x, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
        # x = conv2d('conv_final3', x, [3, 3, 256, 256], stride=1, wd=self.weight_decay)

        # Unary prediction
        # [batch_size, height/4, width/4, 313]
        x = conv2d('conv_final4', x, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)

        return x

    def loss(self, scope, conv8_313, prior_color_weight_nongray, gt_ab_313):
        """loss

        Args:
          conv8_313: 4-D tensor [batch_size, height/4, width/4, 313],
                     predicted ab probability distribution of images
          prior_color_weight_nongray: 4-D tensor [batch_size, height/4, width/4, 313],
                               prior weight for each color bin on the ab gamut
          gt_ab_313: 4-D tensor [batch_size, height/4, width/4, 313],
                     real ab probability distribution of images
        Return:
          new_loss: L_cl(Z_predicted, Z) as in the paper
          g_loss: cross_entropy between predicted and real ab probability
                  distribution of images
        """

        flat_conv8_313 = tf.reshape(conv8_313, [-1, 313])
        flat_gt_ab_313 = tf.reshape(gt_ab_313, [-1, 313])
        g_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_gt_ab_313, logits=flat_conv8_313)) / (self.batch_size)

        dl2c = tf.gradients(g_loss, conv8_313)
        dl2c = tf.stop_gradient(dl2c)

        weight_loss = tf.add_n(tf.get_collection('losses', scope=scope))
        tf.summary.scalar('weight_loss', weight_loss)

        # prior_color_weight_nongray (batch_size, height/4, width/4, 1)
        new_loss = tf.reduce_sum(dl2c * conv8_313 * prior_color_weight_nongray) + weight_loss

        return new_loss, g_loss

