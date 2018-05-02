'''
Create two files one consists of picture path for training, the other for testing
'''
import os
from random import shuffle

imagenet_basepath = './data/output/'
file_list = list(os.listdir(imagenet_basepath))
shuffle(file_list)
# Set up the ratio between training set and testing set
split_ratio = 0.995
mid = int(len(file_list) * split_ratio)

# Split files into training and testing
train_file_list = file_list[:mid]
test_file_list = file_list[mid:]

with open('data/train.txt', 'w') as f:
    for pic_path in train_file_list:
        if pic_path[0] != '.':
            image = os.path.abspath(imagenet_basepath + pic_path)
            f.write(image + '\n')

with open('data/test.txt', 'w') as f:
    for pic_path in test_file_list:
        if pic_path[0] != '.':
            image = os.path.abspath(imagenet_basepath + pic_path)
            f.write(image + '\n')
