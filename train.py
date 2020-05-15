import argparse
from peddla import peddla_net
import numpy as np
import torch
from eval_city.eval_script.eval_demo import validate
import json
import time
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from dataloader.loader import *
from config import Config
import torch.nn.functional as function
from torch import autograd


def parse_args():

    parser = argparse.ArgumentParser(description='Train SiamAF')
    # parser.add_argument('--train_size', type=list, default=[640, 1280], dest='train_size')  # cityperson
    parser.add_argument('--train_size', type=list, default=[640, 1200], dest='train_size')    # eurocity
    parser.add_argument('--gpu_ids', type=list, default=[0], dest='gpu_ids')
    parser.add_argument('--one_gpu', type=int, default=2, dest='onegpu')
    parser.add_argument('--init_lr', type=float, default=2e-4, dest='init_lr')
    parser.add_argument('--epochs', type=int, default=500, dest='epochs')
    parser.add_argument('--data_path', type=str, default='./data/citypersons', dest='data_path')
    parser.add_argument('--save_epoch', type=int, default=1, dest='val_frequency')
    parser.add_argument('--img_list', type=str, default='files of image list')
    
    args = parser.parse_args()
    return args

def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    return ema_model

def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    # input_softmax = function.softmax(input_logits, dim=1)
    # target_softmax = function.softmax(target_logits, dim=1)
    # num_classes = input_logits.size()[1]
    return function.mse_loss(input_logits, target_logits, size_average=False) / 4

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


def train():
    args = parse_args()
    config = Config()

    # dataset
    print('Dataset...')
    traintransform = Compose(
        [ColorJitter(brightness=0.5), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    traindataset = CityPersons(path=args.data_path, type='train', config=config,
                           transform=traintransform, caffemodel=True)
    trainloader = DataLoader(traindataset, batch_size=args.onegpu*len(args.gpu_ids))
    
    # net
    print("Net...")
    num_layers = 34
    heads = {'hm': 1, 'wh': 1, 'reg': 2, 'aed': 4}
    model = peddla_net(num_layers, heads, head_conv=256, down_ratio=4).cuda()
     
    # load model
    # model = load_model(model, 'final.pth')
    # tea_model = load_model(model, 'final.pth')
    model.load_state_dict(torch.load('./models/CSID-92.pth'))
    # torch.cuda.empty_cache()

    teacher_dict = model.state_dict()

    # position
    center = cls_pos().cuda()
    height = reg_pos().cuda()
    offset = offset_pos().cuda()
    aed = aed_pos().cuda()

    # optimizer
    params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append({'params': p})
        else:
            print(n)
    
    # multi GPUS parallel 
    model = nn.DataParallel(model, device_ids=args.gpu_ids)

    # optimizer
    optimizer = optim.Adam(params, lr=args.init_lr)

    # batchsize
    batchsize = args.onegpu * len(args.gpu_ids)
    train_batches = len(trainloader)

    def criterion(output, label):
        cls_loss = center(output['hm'].sigmoid_(), label[0])
        reg_loss = height(output['wh'], label[1])
        off_loss = offset(output['reg'], label[2])
        aed_loss = aed(output['aed'], label[3])
        return cls_loss, reg_loss, off_loss, aed_loss
    
    print('Training start')
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')

    best_loss = np.Inf
    best_loss_epoch = 0

    best_mr = 100
    best_mr_epoch = 0

    for epoch in range(args.epochs):
        print('----------')
        print('Epoch %d begin' % (epoch + 1))
        t1 = time.time()

        epoch_loss = 0.0
        model.train()

        for i, data in enumerate(trainloader, 0):
            t3 = time.time()
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = [l.cuda().float() for l in labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # heat map
            outputs = model(inputs)
            outputs = outputs[-1]
            
            '''
            stu = []
            for s in model.named_parameters():
                stu.append(s[1])
            for s in tea_model.parameters():
                s.weight.data.
            '''

            # loss
            cls_loss, reg_loss, off_loss, aed_loss = criterion(outputs, labels)
            loss = cls_loss + reg_loss + off_loss + aed_loss

            # back-prop
            loss.backward()

            # update param
            optimizer.step()
            for k, v in model.module.state_dict().items():
                if k.find('num_batches_tracked') == -1:
                    teacher_dict[k] = config.alpha * teacher_dict[k] + (1 - config.alpha) * v
                else:
                    teacher_dict[k] = 1 * v
            
            # print statistics
            batch_loss = loss.item()
            batch_cls_loss = cls_loss.item()
            batch_reg_loss = reg_loss.item()
            batch_off_loss = off_loss.item()
            batch_aed_loss = aed_loss.item()

            t4 = time.time()
            print('\r[Epoch %d/150, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, aed: %.6f, Time: %.3f sec        ' %
                  (epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss, batch_reg_loss, batch_off_loss, batch_aed_loss, t4-t3)),
            epoch_loss += batch_loss
        print('')

        t2 = time.time()
        epoch_loss /= len(trainloader)
        print('Epoch %d end, AvgLoss is %.6f, Time used %.1f sec.' % (epoch+1, epoch_loss, int(t2-t1)))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_loss_epoch = epoch + 1
        print('Epoch %d has lowest loss: %.7f' % (best_loss_epoch, best_loss))

        if epoch + 1 > 0 and (epoch + 1) % args.val_frequency == 0:
            def val():
                model.eval()
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                t3 = time.time()
                res = []
                i = 1
                # file_lists = sorted(glob.glob(args.img_list))
                file_lists = open('val.txt')
                for file in tqdm(file_lists.readlines()):
                    torch.cuda.synchronize()
                    seg = file[:-2].split('_')
                    img = plt.imread('/mnt/raid/Talor/CSP_res2net/data/citypersons/images/val/' + seg[0] + '/' + file[:-2]).astype(np.float32)
                    img_pre = preprocess(img[:, :, ::-1], mean, std)
                    img_pre = img_pre.cuda()

                    with torch.no_grad():
                        output = model(img_pre)[-1]
                    output['hm'].sigmoid_()
                    hm, wh, reg, attr = output['hm'], output['wh'], output['reg'], output['aed']

                    density = attr.pow(2).sum(dim=1, keepdim=True).sqrt()
                    diversity = torch.div(attr, density)
                    boxes = parse_det(hm, wh, reg, density=density, diversity=diversity, score=0.01, down=4)

                    if len(boxes) > 0:
                        boxes[:, [2, 3]] -= boxes[:, [0, 1]]

                        for box in boxes:
                            temp = dict()
                            temp['image_id'] = i
                            temp['category_id'] = 1
                            temp['bbox'] = box[:4].tolist()
                            temp['score'] = float(box[4])
                            res.append(temp)
                    i = i + 1

                with open('./_temp_val.json', 'w') as f:
                    json.dump(res, f)

                MRs = validate('./eval_city/val_gt.json', './_temp_val.json')
                t4 = time.time()
                print('Summerize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
                    % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
                print('Validation time used: %.3f' % (t4 - t3))
                return MRs[0]
            cur_mr = val()
            if cur_mr < best_mr:
                best_mr = cur_mr
                best_mr_epoch = epoch + 1
            print('Epoch %d has lowest MR: %.7f' % (best_mr_epoch, best_mr))
            
        print('Save checkpoint...')
        filename = './ckpt/CSID-%d.pth' % (epoch+1)

        torch.save(model.module.state_dict(), filename)

        print('%s saved.' % filename)

if __name__ == "__main__":
    train()
