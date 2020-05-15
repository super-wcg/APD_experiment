import os
import time
import torch
import json
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from net.network1 import CSPNet1, CSPNet_mod
from net.network import CSPNet
from config import Config
from dataloader.loader import *
from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate
from util.nms_wrapper import nms
import numpy as np
from multiprocessing import Process, Manager

import argparse
import glob
from peddla import peddla_net
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from tqdm import tqdm

config = Config()
config.train_path = './data/citypersons'
config.test_path = './data/citypersons'
config.gpu_ids = [0]
config.onegpu = 2
config.size_train = (640, 1280)
config.size_test = (1024, 2048)
config.init_lr = 1e-5
config.num_epochs = 150
config.offset = True
config.val = True
config.val_frequency = 1

def preprocess(image, mean, std):
    img = (image - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis, ...])

def parse_det(hm, wh, reg, density=None, diversity=None, score=0.1,down=4):
    # hm = _nms(hm, kernel=2)
    seman = hm[0, 0].cpu().numpy()
    height = wh[0, 0].cpu().numpy()
    offset_y = reg[0, 0, :, :].cpu().numpy()
    offset_x = reg[0, 1, :, :].cpu().numpy()
    density = density[0, 0].cpu().numpy()
    diversity = diversity[0].cpu().numpy()
    y_c, x_c = np.where(seman > score)
    maxh = int(down * seman.shape[0])
    maxw = int(down * seman.shape[1])
    boxs = []
    dens = []
    divers = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41 * h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, maxw), min(y1 + h, maxh), s])
            dens.append(density[y_c[i], x_c[i]])
            divers.append(diversity[:, y_c[i], x_c[i]])
        boxs = np.asarray(boxs, dtype=np.float32)
        dens = np.asarray(dens, dtype=np.float32)
        divers = np.asarray(divers, dtype=np.float32)
        keep = a_nms(boxs, 0.5, dens, divers)
        boxs = boxs[keep, :]
    else:
        boxs = np.asarray(boxs, dtype=np.float32)
    return boxs

def a_nms(dets, thresh, density, diversity):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        thresh_update = min(max(thresh, density[i]), 0.75)

        temp_tag = diversity[i]
        temp_tags = diversity[order[1:]]
        diff = np.sqrt(np.power((temp_tag - temp_tags), 2).sum(1))
        Flag_4 = diff > 0.95

        thresh_ = np.ones_like(ovr) * 0.5
        thresh_[Flag_4] = thresh_update
        inds = np.where(ovr <= thresh_)[0]
        order = order[inds + 1]

    return keep

def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model


# dataset
print('Dataset...')
if config.val:
    testtransform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testdataset = CityPersons(path=config.train_path, type='val', config=config,
                              transform=testtransform, preloaded=True)
    testloader = DataLoader(testdataset, batch_size=1)

# net
print('Net...')
net1 = CSPNet1().cuda()
# net = CSPNet().cuda()
# To continue training
# net.load_state_dict(torch.load('./ckpt/CSPNet-47.pth'))
net1.load_state_dict(torch.load('./models/CSPNet-8.pth'))
# net.load_state_dict(torch.load('./models/CSPNet-11.pth'))
#if len(config.gpu_ids) > 1:
net1 = nn.DataParallel(net1, device_ids=config.gpu_ids)
# net = nn.DataParallel(net, device_ids=config.gpu_ids)

# BGR
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# args = parse_args()
num_layers = 34
heads = {'hm': 1, 'wh': 1, 'reg': 2, 'aed': 4}
model = peddla_net(num_layers, heads, head_conv=256, down_ratio=4).cuda().eval()

# load model
model = load_model(model, 'final.pth')
# model.load_state_dict(torch.load('./ckpt/CSID-5.pth'))

def val(log=None):
    net1.eval()
    model.eval()
    # net.eval()

    print('Perform validation...')
    res = []
    t3 = time.time()
    for i, data in enumerate(testloader, 0):
        tt1 = time.time()
        inputs = data[0].cuda()
        
        img = plt.imread(data[1][0]).astype(np.float32)
        img_pre = preprocess(img[:, :, ::-1], mean, std)
        img_pre = img_pre.cuda()
        with torch.no_grad():
            pos1, height1, offset1 = net1(inputs)
            #  pos, height, offset = net(inputs)
            output = model(img_pre)[-1]
        output['hm'].sigmoid_()
        hm, wh, reg, attr = output['hm'], output['wh'], output['reg'], output['aed']

        density = attr.pow(2).sum(dim=1, keepdim=True).sqrt()
        diversity = torch.div(attr, density)
        boxes = parse_det(hm, wh, reg, density=density, diversity=diversity, score=0.01, down=4)
        
        # boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.01, down=4, nms_thresh=0.5)
        boxes1 = parse_det_offset(pos1.cpu().numpy(), height1.cpu().numpy(), offset1.cpu().numpy(), config.size_test, score=0.01, down=4, nms_thresh=0.5)
        
        bb = list(boxes) + list(boxes1)
        # bb = list(boxes)
        boxes = np.asarray(bb, dtype=np.float32)
        keep = nms(boxes, 0.5, usegpu=False, gpu_id=0)
        boxes = boxes[keep, :]
        if len(boxes) > 0:
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]

            for box in boxes:
                temp = dict()
                temp['image_id'] = i+1
                temp['category_id'] = 1
                temp['bbox'] = box[:4].tolist()
                temp['score'] = float(box[4])
                res.append(temp)
        tt2 = time.time()
        print('\r%d/%d, %f' % (i + 1, len(testloader), tt2 - tt1)),
        sys.stdout.flush()
    print('')

    if config.teacher:
        print('Load back student params')
        net.module.load_state_dict(student_dict)

    with open('./_temp_val.json', 'w') as f:
        json.dump(res, f)

    MRs = validate('./eval_city/val_gt.json', './_temp_val.json')
    t4 = time.time()
    print('Summerize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    if log is not None:
        log.write("%.7f %.7f %.7f %.7f\n" % tuple(MRs))
    print('Validation time used: %.3f' % (t4 - t3))
    return MRs[0]


if __name__ == '__main__':
    val()
