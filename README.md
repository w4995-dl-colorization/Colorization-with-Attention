
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
