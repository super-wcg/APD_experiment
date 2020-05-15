# APD
[Attribute-aware Pedestrian Detection in a Crowd](https://arxiv.org/pdf/1910.09188.pdf)

## Installation

The project need python2(if you want used python3, you need modify some code, but it is very easy.)

To run the demo, the following requirements are needed.
```
numpy
matplotlib
torch >= 0.4.1
glob
argparse
[DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)
```

## Model
[CSID-92.pth](https://pan.baidu.com/s/16wCiO4qhOVprzThYL8V-LQ)(code:qo85) is a model trained on [cityperson dataset](https://bitbucket.org/shanshanzhang/citypersons/src/default/).
Put model in './models'.

Prepare CityPersons dataset as the original codes doing

* For citypersons, we use the training set (2975 images) for training and test on the validation set (500 images), we assume that images and annotations are stored in  `./data/citypersons`, and the directory structure is

```
*DATA_PATH
	*annotations
		*anno_train.mat
		*anno_val.mat
	*images
		*train
		*val
```

## Train
The demo code and the trained is only for cityperson dataset.
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --one_gpu 1 --init_lr 2e-5
```

## Test
```
python test_merge_APD.py
```


## Demo
The demo code and the trained is only for cityperson dataset.
```
python demo.py --img_list 'images/*.pth'
```


refer to:https://github.com/super-wcg/APD