
###Train

##### Download MIT CVCL Opencountry dataset
http://cvcl.mit.edu/database.htm

#### Train customer data

1. transform your training data to text_record file
```
python tools/create_imagenet_list.py
```

2. calculate the training data prior-probs(reference to tools/create_prior_probs.py)
```
python tools/create_prior_probs.py
```

3. write your own train-configure file and put it in conf/ (reference to conf/train.cfg for an example)

4. train (python tools/train.py -c $your_configure_file)
```
python tools/train.py -c conf/train.cfg
```

### test demo

1. Download pretrained model(<a>https://drive.google.com/file/d/0B-yiAeTLLamRWVVDQ1VmZ3BxWG8/view?usp=sharing</a>)

	```
	mv color_model.ckpt models/model.ckpt
	```
2. Test

	```
	python demo.py
	```

### Acknowledgement
The code in this repo is built on top of the work at:
https://github.com/nilboy/colorization-tf
