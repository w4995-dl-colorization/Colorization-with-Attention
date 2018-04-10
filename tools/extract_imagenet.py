# at root, run this script by python tools/extract_imagenet.py
# relative path for imagenet dir: data/ILSVRC2012_img_train/*.tar
# 1000 tars for each class in total
# each tar can be extracted to get 1300 original size imgs

### USAGE: this script take random <num_class> tars and extract them to the corresponding name folder

import os
import glob
import ntpath
import random
import tarfile

#config path
path = os.path.join(os.getcwd(), 'data', 'ILSVRC2012_img_train')

list_of_tar = glob.glob(path+"/*.tar")

print("In total %d classes of tars scanned" % len(list_of_tar))

#config no. of classes
num_class = 10

list_of_chosen = random.sample(list_of_tar, num_class)

for tarname in list_of_chosen:
    print("extracting %s" % tarname)
    bname = ntpath.basename(tarname)
    dirname = bname.split('.')[0]
    tar = tarfile.open(tarname)
    tar.extractall(path=path+"/"+dirname)
    tar.close()