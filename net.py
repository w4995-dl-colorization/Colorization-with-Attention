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

    def count_ht_loss(self, temp_conv, data_l, res_hm1):
        # In test mode, set res_hm1 to constant 0 tensor.
        if res_hm1 == None:
            res_hm1 = tf.constant(0.0, shape=[temp_conv.shape[0], temp_conv.shape[1], temp_conv.shape[2]])
        sum1 = tf.reduce_sum(temp_conv, axis=1)
        sum1 = tf.reduce_sum(sum1, axis=1)
        sum1 = tf.reshape(tf.reduce_sum(sum1, axis=1), (data_l.shape[0],1,1))

        ht_loss1 = tf.nn.l2_loss(tf.reduce_sum(temp_conv, axis=3)/sum1 - res_hm1)
        tf.add_to_collection('ht_losses', ht_loss1)

        return 0


    def inference(self, data_l, res_hm1=None, res_hm2=None):
        """infer ab probability distribution of images from black-white images

        Args:
          data_l: 4-D tensor [batch_size, height, width, 1],
                  images with only L channel
          res_hm: [batch_size, height/4, width/4], heat map
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

            # Heat Map Loss1
            self.count_ht_loss(temp_conv, data_l, res_hm1)
                
            
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

            # Heat Map Loss2
            self.count_ht_loss(temp_conv, data_l, res_hm2)


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

    def loss(self, conv8_313, prior_color_weight_nongray, gt_ab_313, res_hm1):
        """loss

        Args:
          conv8_313: 4-D tensor [batch_size, height/4, width/4, 313],
                     predicted ab probability distribution of images
          prior_color_weight_nongray: 4-D tensor [batch_size, height/4, width/4, 313],
                               prior weight for each color bin on the ab gamut
          gt_ab_313: 4-D tensor [batch_size, height/4, width/4, 313],
                     real ab probability distribution of images
          ht_loss: scalar, the loss caused by heat map.
          res_hm1: 3-D tensor [batch_size, height/4, width/4],
                   heatmap extracted from VGG16.
        Return:
          new_loss: L_cl(Z_predicted, Z) as in the paper
          g_loss: cross_entropy between predicted and real ab probability
                  distribution of images
        """
        # Get the two losses and record them into summary
        weight_loss = tf.add_n(tf.get_collection('losses'))
        ht_loss = tf.add_n(tf.get_collection('ht_losses'))

        tf.summary.scalar('weight_loss', weight_loss)
        tf.summary.scalar('ht_loss', ht_loss)

        # Coefficient that governs the regularization of the two heatmaps
        # Set it to 0 for now to disable this regularization
        beta = 0
        beta_ht_loss = beta*ht_loss


        flat_conv8_313 = tf.reshape(conv8_313, [-1, 313])
        flat_gt_ab_313 = tf.reshape(gt_ab_313, [-1, 313])
        g_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_gt_ab_313, logits=flat_conv8_313)) / (self.batch_size)  + beta_ht_loss

        dl2c = tf.gradients(g_loss, conv8_313)
        dl2c = tf.stop_gradient(dl2c)


        # Create a weighted mask from heatmap and use it to weight loss at each pixel
        size1 = res_hm1.shape[1]
        res_hm1 = tf.reshape(res_hm1, (-1, size1*size1))
        res_hm1 = tf.nn.softmax(res_hm1, axis=1)
        res_hm1_mask = tf.reshape(res_hm1, (1,-1,size1,size1,1))

        # prior_color_weight_nongray (batch_size, height/4, width/4, 1)
        new_loss = tf.reduce_sum(res_hm1_mask * dl2c * conv8_313 * prior_color_weight_nongray) + weight_loss + beta_ht_loss

        return new_loss, g_loss, ht_loss
