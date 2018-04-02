
### Train

##### Download MIT CVCL Opencountry dataset
http://cvcl.mit.edu/database.htm

#### Train customer data

1. Transform your training data to text_record file
```
python3 tools/create_imagenet_list.py
```

2. Calculate the training data prior-probs(reference to tools/create_prior_probs.py)
Note: Currently, each bin has to appear at least once in the training images in order to avoid breaking the training procedure. If training images do not cover all the colors, you can use
the existing "prior_probs.npy" in the resource folder by skipping this step.
```
python3 tools/create_prior_probs.py
```

3. Write your own train-configure file and put it in conf/ (reference to conf/train.cfg for an example)

4. Train

CPU:

```
python3 tools/train.py -c conf/train.cfg
```

GPU:
```
python3 tools/train.py -c conf/train_gpu.cfg
```

5. Test

    ```
    python3 demo.py
    ```


### Acknowledgement
The code in this repo is built on top of the work at:
https://github.com/nilboy/colorization-tf

https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py

### Tree Structure
├── conf            (configure files)

├── data            (dataset)

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

├── testslim.py

├── tools

│   ├── create_imagenet_list.py (create training list for colorization)

│   ├── create_prior_probs.py   (create empiricial probability used for class rebalancing from current dataset)

│   ├── cropnresize.py          (preprocess images for attention)

│   ├── testslim.py             (test attention extraction)

│   └── train.py                (wrapper for training)

└── utils.py                    (helper functions)
