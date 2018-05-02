# Colorization with Attention

### Results
![presentation](https://github.com/w4995-dl-colorization/Colorization-with-Attention/blob/master/images/results.jpg)
The images are (from left to right): Ground-truth, generated by No Attention Model, generated by Weighted-loss Attention Model, generated by End-to-end Attention Model.

### Instructions

#### Download a dataset

MIT CVCL Opencountry dataset at: http://cvcl.mit.edu/database.htm

ImageNet at: http://www.image-net.org/

#### Use Pretrained Model 
1. Download our pre-trained models(Three different models) [here](https://drive.google.com/file/d/1yI0vt6dv_xVKFKWX2-p609sNfPmABrV8/view?usp=sharing), unzip it and put the model/ folder into the top level of the repo.

Note: The pretrained model is only trained using a subset of ImageNet (~50,000 animal images) so it
does not give guarantee to performance on other type of images.

2. Test (Follow the instruction in demo.py to set up parameters)

```
python3 demo.py
```

#### Train your own model

1. Transform your training data to text_record file(The ratio of splitting training/testing can be set up in create_imagenet_list.py)
```
python3 tools/create_imagenet_list.py
```

2. Calculate the training data prior-probs(reference to tools/create_prior_probs.py)
Note: Currently, each bin has to appear at least once in the training images in order to avoid breaking the training procedure. If training images do not cover all the colors, you can use
the existing "prior_probs.npy" in the resource folder by skipping this step.
```
python3 tools/create_prior_probs.py
```

3. Write your own train-configure file and put it in conf/ (refer to conf/train.cfg for an example)

4. Train

CPU:

```
python3 tools/train.py -c conf/train.cfg
```

GPU:
```
python3 tools/train.py -c conf/train_gpu.cfg
```

5. Test (Follow the instruction in demo.py to set up parameters)

```
python3 demo.py
```

### Reference
[Richard Zhang, Phillip Isola, Alexei A. Efros. Colorful Image Colorization, ECCV2016.](https://arxiv.org/abs/1603.08511)

[Saumya Jetley, Nicholas A. Lord, Namhoon Lee and Philip H. S. Torr, learn to pay attention, ICLR 2018](https://arxiv.org/abs/1804.02391)

### Acknowledgement
The code in this repo is built on top of the work at:

https://github.com/nilboy/colorization-tf

https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py

### Tree Structure
├── conf            (configure files)

├── data            (dataset)

├── evaluate.py     (Run quantitative measurement of errors of the prediction)

├── evaluation

│   ├── evaluate_loss.py (evaluate colorization quantitatively)

├── data.py         (handling data processing for colorization)

├── demo.py         (demo for testing correctness)

├── models          (store trained models)

├── net.py          (no_attention/weighted-loss attention model)

├── net_att.py      (end-to-end attention model)

├── net_densenet.py (densenet no_attention model to be fixed)

├── ops.py          (customized tf layers)

├── resources       

│   ├── prior_probs.npy (empiricial probability used for class rebalancing extracted from imagenet)

│   └── pts_in_hull.npy (values of each bin on ab color space)

├── slim_vgg.py         (teacher CNN vgg-net for classification)

├── solver.py           (graph and session for training)

├── testslim.py         (Can be used to extract attention map)

├── tools

│   ├── create_imagenet_list.py (create training list for colorization)

│   ├── create_prior_probs.py   (create empiricial probability used for class rebalancing from current dataset)

│   └── train.py                (wrapper for training)

└── utils.py                    (helper functions)

