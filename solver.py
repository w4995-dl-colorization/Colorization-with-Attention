import numpy as np
import tensorflow as tf
import os
import time
from net import Net
from net_att import Net_att
from net_densenet import DenseNet
from data import DataSet
from datetime import datetime
from utils import decode
from skimage.io import imsave

import cv2
import slim_vgg


class Solver(object):
    def __init__(self, train=True, common_params=None, solver_params=None, net_params=None, dataset_params=None):
        if common_params:
            self.device = common_params['device']
            self.image_size = int(common_params['image_size'])
            self.height = self.image_size
            self.width = self.image_size
            self.batch_size = int(common_params['batch_size'])
            self.num_gpus = 1
            # end_to_end: if use end_to_end attention model or Richard Zhang's model
            self.end_to_end = False if common_params['end_to_end']=='False' else True
            # use_attention_in_cost: if use attention to weight loss in the cost function
            self.use_attention_in_cost = False if common_params['use_attention_in_cost']=='False' else True

        if solver_params:
            self.learning_rate = float(solver_params['learning_rate'])
            self.moment = float(solver_params['moment'])
            self.max_steps = int(solver_params['max_iterators'])
            self.train_dir = str(solver_params['train_dir'])
            self.lr_decay = float(solver_params['lr_decay'])
            self.decay_steps = int(solver_params['decay_steps'])

        self.common_params = common_params
        self.net_params = net_params
        self.train = train
        self.dataset = DataSet(common_params=common_params, dataset_params=dataset_params)


    def construct_graph_for_student(self):
        with tf.device(self.device):
            self.training_flag = tf.placeholder(tf.bool)
            self.res_hm1 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4)))
            self.res_hm2 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4)))
            self.res_hm3 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4)))

            self.data_l = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 1))
            self.gt_ab_313 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 313))
            self.prior_color_weight_nongray = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 1))

            if self.end_to_end == True:
                self.net = Net_att(train=self.training_flag, common_params=self.common_params, net_params=self.net_params)
            else:
                self.net = Net(train=self.training_flag, common_params=self.common_params, net_params=self.net_params)

            # self.net = DenseNet(train=self.training_flag, common_params=self.common_params, net_params=self.net_params)

            self.conv8_313 = self.net.inference(self.data_l)
            new_loss, g_loss = self.net.loss(self.conv8_313, self.prior_color_weight_nongray, self.gt_ab_313, self.res_hm1, self.res_hm2, self.res_hm3, self.use_attention_in_cost)
            tf.summary.scalar('new_loss', new_loss)
            tf.summary.scalar('total_loss', g_loss)

        return new_loss, g_loss


    def construct_graph_for_teacher(self):
        with tf.device(self.device):
            inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
            _, end_points = slim_vgg.vgg_16(inputs)
            # heatmap tensors
            hm1 = end_points['hm1']
            hm2 = end_points['hm2']
            hm3 = end_points['hm3']
        return inputs, hm1, hm2, hm3

    # Normalize attention heat map
    def process_attention(self, attention_hm, size1, size2=64):
        eps = 1e-5
        res_hm = attention_hm.reshape(self.batch_size, size1**2)
        # center heat map
        centered_res_hm = res_hm - res_hm.mean(axis=1).reshape((self.batch_size,1))
        # divide by stdev
        denom_res_hm = np.sqrt((centered_res_hm**2).sum(axis=1)/(size1*size1) + eps).reshape((self.batch_size,1))
        res_hm = centered_res_hm / denom_res_hm
        # reshape
        res_hm = res_hm.reshape((self.batch_size, size1, size1))
        # resize to 64 x 64
        res_hm = np.concatenate([cv2.resize(res_hm[i], (size2, size2))[None, :, :] for i in range(self.batch_size)], axis=0)
        return res_hm


    def train_model(self):

        with tf.device(self.device):

            # Student
            # Construct graph
            new_loss, self.total_loss = self.construct_graph_for_student()

            # Initialize and configure optimizer
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                 self.decay_steps, self.lr_decay, staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=0.99)


            # Compute gradient, moving average of weights and update weights
            grads = opt.compute_gradients(new_loss)
            apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
            variable_averages = tf.train.ExponentialMovingAverage(
                0.999, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_op = tf.group(apply_gradient_op, variables_averages_op)


            # Record values into summary
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='colorization')
            self.summaries.append(tf.summary.scalar('learning_rate', learning_rate))
            for grad, var in grads:
                if grad is not None:
                    self.summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            for var in tf.trainable_variables():
                if var is not None:
                    self.summaries.append(tf.summary.histogram(var.op.name, var))

            summary_op = tf.summary.merge(self.summaries)


            # Initialize and configure student and teacher sessions
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            sess_teacher = tf.Session(config=config)


            # Student: load/create model
            saver_student = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='colorization'))
            ckpt_student = tf.train.get_checkpoint_state('models/model.ckpt')
            if ckpt_student and tf.train.checkpoint_exists(ckpt_student.model_checkpoint_path):
                saver_student.restore(sess, ckpt_student.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            # Teacher: load model
            inputs, hm1, hm2, hm3 = self.construct_graph_for_teacher()
            saver_teacher = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16'))
            saver_teacher.restore(sess_teacher, 'models/vgg16.ckpt')


            # Student: Initialize summary writer
            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

            for step in range(self.max_steps):
                start_time = time.time()

                # Get input data
                images, data_l, gt_ab_313, prior_color_weight_nongray = self.dataset.batch()


                res_hm1 = np.zeros((self.batch_size, 64, 64))
                res_hm2 = np.zeros((self.batch_size, 64, 64))
                res_hm3 = np.zeros((self.batch_size, 64, 64))

                # Extract attention when the end-to-end structure is not used
                if self.use_attention_in_cost:
                    # Teacher: Forward pass to grab/process heat map
                    res_pics = np.concatenate([cv2.resize(img, (224, 224),
                                               interpolation=cv2.INTER_AREA)[None, :, :, :] for img in images], axis=0)

                    attention_hm1, attention_hm2, attention_hm3 = sess_teacher.run((hm1, hm2, hm3), feed_dict={inputs: res_pics})
                    res_hm1 = self.process_attention(attention_hm1, 56, 64)
                    res_hm2 = self.process_attention(attention_hm2, 28, 64)
                    res_hm3 = self.process_attention(attention_hm3, 7, 64)



                # Student: Optimize objective for colorization

                feed_d={self.training_flag:self.train,
                      self.data_l:data_l,
                      self.gt_ab_313:gt_ab_313,
                      self.prior_color_weight_nongray:prior_color_weight_nongray,
                      self.res_hm1:res_hm1,
                      self.res_hm2:res_hm2,
                      self.res_hm3:res_hm3}

                _, loss_value = sess.run([train_op, self.total_loss], feed_dict=feed_d)


                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'


                # Print training info periodically.
                if step % 1 == 0:
                    num_examples_per_step = self.batch_size * self.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / self.num_gpus

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value,
                                         examples_per_sec, sec_per_batch))

                # Record progress periodically.
                if step % 20 == 0:
                    summary_str = sess.run(summary_op, feed_dict=feed_d)
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 100 == 0:
                    checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
                    saver_student.save(sess, checkpoint_path)
