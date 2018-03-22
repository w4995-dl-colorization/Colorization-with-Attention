from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import cv2
import numpy as np
from queue import Queue
from threading import Thread as Process
from utils import preprocess


class DataSet(object):
    """TextDataSet
    process text input file dataset
    text file format:
      image_path
    """

    def __init__(self, common_params=None, dataset_params=None):
        """
        Args:
          common_params: A dict
          dataset_params: A dict
        """
        if common_params:
            self.image_size = int(common_params['image_size'])
            self.batch_size = int(common_params['batch_size'])

        if dataset_params:
            self.data_path = str(dataset_params['path'])
            self.thread_num = int(int(dataset_params['thread_num']))

        # Create record and batch queue for multi-threading
        self.record_queue = Queue(maxsize=10000)
        self.batch_queue = Queue(maxsize=100)

        # Fill in the record_list
        self.record_list = []
        input_file = open(self.data_path, 'r')

        for line in input_file:
            line = line.strip()
            self.record_list.append(line)

        self.record_point = 0
        self.record_number = len(self.record_list)
        self.num_batch_per_epoch = int(self.record_number / self.batch_size)

        # Keep adding record into record_queue
        t_record_producer = Process(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        # (Multi-threads) Read/Process images and batch them
        for i in range(self.thread_num):
            t = Process(target=self.image_customer)
            t.daemon = True
            t.start()

    def record_producer(self):
        """Add records into record_queue
        """
        while True:
            if self.record_point % self.record_number == 0:
                random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    def image_customer(self):
        """ Get record from the record_queue, read images, process them, batch them,
            and put them into the batch_queue
        """
        while True:
            images = []
            for i in range(self.batch_size):
                item = self.record_queue.get()
                image = cv2.imread(item)
                assert len(image.shape)==3 and image.shape[2]==3
                images.append(image)
            images = self.image_process(images)
            images = np.asarray(images, dtype=np.uint8)

            self.batch_queue.put(preprocess(images))

    def image_process(self, batch):
        """ Randomly flip/crop the image
        Args:
          image: a list of 3-D ndarray [height, width, 3], rgb image batch
        Returns:
          image: a list of 3-D ndarray [height, width, 3], processed rgb image batch
        """
        
        def _random_crop(batch, crop_shape, padding=None):
            oshape = np.shape(batch[0])

            if padding:
                oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
            new_batch = []
            npad = ((padding, padding), (padding, padding), (0, 0))
            for i in range(len(batch)):
                new_batch.append(batch[i])
                if padding:
                    new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                              mode='constant', constant_values=0)
                nh = random.randint(0, oshape[0] - crop_shape[0])
                nw = random.randint(0, oshape[1] - crop_shape[1])
                new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
            return new_batch

        def _random_flip_leftright(batch):
            for i in range(len(batch)):
                if bool(random.getrandbits(1)):
                    batch[i] = np.fliplr(batch[i])
            return batch

        batch = _random_flip_leftright(batch)
        batch = _random_crop(batch, [batch[0].shape[0], batch[0].shape[1]], 4)

        return batch

    def batch(self):
        """get images batch from the batch_queue
        Returns:
          images_batch: 4-D ndarray [batch_size, height, width, 3]
        """
        # print(self.record_queue.qsize(), self.batch_queue.qsize())
        images_batch = self.batch_queue.get()
        return images_batch
