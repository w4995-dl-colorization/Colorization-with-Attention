'''
If use_attention_in_cost==False, this is the same as the Richard Zhang's colorization model.
We call this No-Attention Model.

It use_attention_in_cost==True, cost function is weighted according to attention extracted from
a pretrained VGG-net.
We call this Weighted-Loss Attention Model.
'''

import tensorflow as tf
from ops import conv2d, deconv2d, batch_norm


class Net(object):

    def __init__(self, train=True, common_params=None, net_params=None):
        self.train = train
        self.weight_decay = 0.0
        if common_params:
            self.batch_size = int(common_params['batch_size'])
        if net_params:
            self.weight_decay = float(net_params['weight_decay'])
        self.res_hm1 = None



    def inference(self, data_l):
        """infer ab probability distribution of images from black-white images

        Args:
          data_l: 4-D tensor [batch_size, height, width, 1],
                  images with only L channel

        Return:
          conv8_313: 4-D tensor [batch_size, height/4, width/4, 313],
                     predicted ab probability distribution of images
        """
        with tf.variable_scope('colorization'):
            # conv1 256->128
            conv_num = 1
            temp_conv = conv2d('conv' + str(conv_num), data_l, [3, 3, 1, 64], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 64, 64], stride=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_1', temp_conv, train=self.train)

            # conv2 128->64
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 64, 128], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 128], stride=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_2', temp_conv, train=self.train)

                
            
            # conv3 64->32
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_3', temp_conv, train=self.train)

            # conv4 32->32
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_4', temp_conv, train=self.train)



            # conv5 dilated 32->32
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_5', temp_conv, train=self.train)

            # conv6 dilated 32->32
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_6', temp_conv, train=self.train)

            # conv7 32->32
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_7', temp_conv, train=self.train)

            # conv8 upsampling 32->64
            temp_conv = deconv2d('conv' + str(conv_num), temp_conv, [4, 4, 512, 256], stride=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1

            # Unary prediction
            temp_conv = conv2d('conv' + str(conv_num), temp_conv, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1


            conv8_313 = temp_conv
        return conv8_313

    def hm_weight_mask(self, res_hm):
        """
        Args:
          res_hm: 3-D tensor heatmap extracted from VGG16.
        Return:
          res_hm_mask: softmax version of 3-D tensor heatmap.
        """
        size1 = res_hm.shape[1]
        res_hm = tf.reshape(res_hm, (-1, size1*size1))
        res_hm = tf.nn.softmax(res_hm, axis=1)
        res_hm_mask = tf.reshape(res_hm, (1,-1,size1,size1,1))
        return res_hm_mask

    def loss(self, conv8_313, prior_color_weight_nongray, gt_ab_313, res_hm1, res_hm2, res_hm3, use_attention_in_cost=False):
        """loss

        Args:
          conv8_313: 4-D tensor [batch_size, height/4, width/4, 313],
                     predicted ab probability distribution of images
          prior_color_weight_nongray: 4-D tensor [batch_size, height/4, width/4, 313],
                               prior weight for each color bin on the ab gamut
          gt_ab_313: 4-D tensor [batch_size, height/4, width/4, 313],
                     real ab probability distribution of images
          res_hm1: 3-D tensor [batch_size, height/4, width/4],
                   high-level heatmap extracted from VGG16.
          res_hm2: 3-D tensor [batch_size, height/4, width/4],
                   mid-level heatmap extracted from VGG16.
          res_hm3: 3-D tensor [batch_size, height/4, width/4],
                   low-level heatmap extracted from VGG16.
        Return:
          new_loss: L_cl(Z_predicted, Z) as in the paper
          g_loss: cross_entropy between predicted and real ab probability
                  distribution of images
        """
        # Get the weight loss and record it into summary
        weight_loss = tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('weight_loss', weight_loss)



        flat_conv8_313 = tf.reshape(conv8_313, [-1, 313])
        flat_gt_ab_313 = tf.reshape(gt_ab_313, [-1, 313])
        g_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_gt_ab_313, logits=flat_conv8_313)) / (self.batch_size)

        dl2c = tf.gradients(g_loss, conv8_313)
        dl2c = tf.stop_gradient(dl2c)


        # Create a weighted mask from heatmap and use it to weight loss at each pixel
        res_hm1_mask = self.hm_weight_mask(res_hm1)
        res_hm2_mask = self.hm_weight_mask(res_hm2)
        res_hm3_mask = self.hm_weight_mask(res_hm3)

        # prior_color_weight_nongray (batch_size, height/4, width/4, 1)

        cross_entropy_tensor = dl2c * conv8_313 * prior_color_weight_nongray
        if use_attention_in_cost:
            cross_entropy_tensor *= res_hm3_mask

        new_loss = tf.reduce_sum(cross_entropy_tensor) + weight_loss

        return new_loss, g_loss
