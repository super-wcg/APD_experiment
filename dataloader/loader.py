from __future__ import division
import sys
import random
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import data_augment
from load_data import get_citypersons, get_eurocity
import matplotlib.pyplot as plt

class CityPersons(Dataset):
    def __init__(self, path, type, config, preloaded=False, transform=None, caffemodel=False):

        # self.dataset = get_citypersons(root_dir=path, type=type)
        self.dataset = get_eurocity(root_dir=path, type=type)
        self.dataset_len = len(self.dataset)
        self.type = type

        if self.type == 'train' and config.train_random:
            random.shuffle(self.dataset)
        self.config = config
        self.transform = transform
        self.caffemodel = caffemodel

        if self.type == 'train':
            self.preprocess = RandomResizeFix(size=config.size_train, scale=(0.4, 1.5))
        else:
            self.preprocess = None

        self.preloaded = preloaded

        if self.preloaded:
            self.img_cache = []
            for i, data in enumerate(self.dataset):
                if self.caffemodel:
                    self.img_cache.append(cv2.imread(data['filepath']))
                else:
                    self.img_cache.append(Image.open(data['filepath']))
                print('%d/%d\r' % (i+1, self.dataset_len)),
                sys.stdout.flush()
            print('')

    def __getitem__(self, item):

        if self.caffemodel:
            # input is BGR order, not normalized
            img_data = self.dataset[item]
            if self.preloaded:
                img = self.img_cache[item]
            else:
                # img = cv2.imread(img_data['filepath'])
                img = plt.imread(img_data['filepath'])

            if self.type == 'train':
                img_data, x_img = data_augment.augment(self.dataset[item], self.config, img)

                gts = img_data['bboxes'].copy()
                igs = img_data['ignoreareas'].copy()

                y_center, y_height, y_offset, y_id = self.calc_gt_center(gts, igs, radius=2, stride=self.config.down)

                x_img = x_img.astype(np.float32)
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                img = (x_img[:, :, ::-1] - mean) / std
                # x_img = torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis, ...])
                # x_img -= [103.939, 116.779, 123.68]
                x_img = torch.from_numpy(img).permute([2, 0, 1])

                return x_img, [y_center, y_height, y_offset, y_id]

            else:
                x_img = img.astype(np.float32)
                x_img -= [103.939, 116.779, 123.68]
                x_img = torch.from_numpy(x_img).permute([2, 0, 1])

                return x_img

        else:
            # input is RGB order, and normalized
            img_data = self.dataset[item]
            if self.preloaded:
                img = self.img_cache[item]
            else:
                img = Image.open(img_data['filepath'])

            if self.type == 'train':
                gts = img_data['bboxes'].copy()
                igs = img_data['ignoreareas'].copy()

                x_img, gts, igs = self.preprocess(img, gts, igs)

                y_center, y_height, y_offset, y_id = self.calc_gt_center(gts, igs, radius=2, stride=self.config.down)
                                        
                if self.transform is not None:
                    x_img = self.transform(x_img)

                return x_img, [y_center, y_height, y_offset, y_id]

            else:
                if self.transform is not None:
                    x_img = self.transform(img)
                else:
                    x_img = img

                return [x_img, img_data['filepath']]

    def __len__(self):
        return self.dataset_len

    def calc_gt_center(self, gts, igs, radius=2, stride=4):

        def gaussian(kernel):
            sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
            s = 2*(sigma**2)
            dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
            return np.reshape(dx, (-1, 1))

        scale_map = np.zeros((2, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        offset_map = np.zeros((3, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        pos_map = np.zeros((3, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        den_map = np.zeros((2, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        pos_map[1, :, :, ] = 1  # channel 1: 1-value mask, ignore area will be set to 0

        if len(igs) > 0:
            igs = igs / stride
            for ind in range(len(igs)):
                x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
                pos_map[1, y1:y2, x1:x2] = 0

        if len(gts) > 0:
            gts = gts / stride
            for ind in range(len(gts)):
                res = 0
                for t in range(len(gts)):
                    if ind != t:
                        iou = self.iou(gts[ind], gts[t])
                        if iou > res:
                            res = iou

                x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
                # c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
                c_xtl, c_ytl = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
                c_xtl, c_ybl = int((gts[ind, 0] + gts[ind, 2]) / 2), int(np.ceil((gts[ind, 1] + gts[ind, 3]) / 2.0))
                c_xtr, c_ytr = int(np.ceil((gts[ind, 0] + gts[ind, 2]) / 2.0)), int((gts[ind, 1] + gts[ind, 3]) / 2)
                c_xtr, c_ybr = int(np.ceil((gts[ind, 0] + gts[ind, 2]) / 2.0)), int(np.ceil((gts[ind, 1] + gts[ind, 3]) / 2.0))

                dx = gaussian(x2-x1)
                dy = gaussian(y2-y1)
                gau_map = np.multiply(dy, np.transpose(dx))

                pos_map[0, y1:y2, x1:x2] = np.maximum(pos_map[0, y1:y2, x1:x2], gau_map)  # gauss map
                pos_map[1, y1:y2, x1:x2] = 1  # 1-mask map
                # pos_map[2, c_y, c_x] = 1  # center map
                pos_map[2, c_ytl, c_xtl] = 1
                pos_map[2, c_ybl, c_xtl] = 1
                pos_map[2, c_ytr, c_xtr] = 1
                pos_map[2, c_ybr, c_xtr] = 1

                # den_map[0, c_y, c_x] = res
                # den_map[1, c_y, c_x] = 1
                den_map[0, c_ytl, c_xtl] = res
                den_map[0, c_ybl, c_xtl] = res
                den_map[0, c_ytr, c_xtr] = res
                den_map[0, c_ybr, c_xtr] = res
                # den_map[1, c_ytl, c_xtl] = 1
                # den_map[2, c_ybl, c_xtl] = 1
                # den_map[3, c_ytr, c_xtr] = 1
                # den_map[4, c_ybr, c_xtr] = 1
                den_map[1, c_ytl, c_xtl] = 1
                den_map[1, c_ybl, c_xtl] = 1
                den_map[1, c_ytr, c_xtr] = 1
                den_map[1, c_ybr, c_xtr] = 1

                # scale_map[0, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
                # scale_map[1, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = 1
                scale_map[0, c_ytl:c_ytl+1, c_xtl:c_xtl+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
                scale_map[1, c_ytl:c_ytl+1, c_xtl:c_xtl+1] = 1  # 1-mask
                scale_map[0, c_ybl:c_ybl+1, c_xtl:c_xtl+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
                scale_map[1, c_ybl:c_ybl+1, c_xtl:c_xtl+1] = 1
                scale_map[0, c_ytr:c_ytr+1, c_xtr:c_xtr+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
                scale_map[1, c_ytr:c_ytr+1, c_xtr:c_xtr+1] = 1
                scale_map[0, c_ybr:c_ybr+1, c_xtr:c_xtr+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
                scale_map[1, c_ybr:c_ybr+1, c_xtr:c_xtr+1] = 1

                # offset_map[0, c_y, c_x] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5  # height-Y offset
                # offset_map[1, c_y, c_x] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5  # width-X offset
                # offset_map[2, c_y, c_x] = 1  # 1-mask
                offset_map[0, c_ytl, c_xtl] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_ytl - 0.5  # height-Y offset
                offset_map[1, c_ytl, c_xtl] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_xtl - 0.5  # width-X offset
                offset_map[2, c_ytl, c_xtl] = 1  # 1-mask
                offset_map[0, c_ybl, c_xtl] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_ybl - 0.5  # height-Y offset
                offset_map[1, c_ybl, c_xtl] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_xtl - 0.5  # width-X offset
                offset_map[2, c_ybl, c_xtl] = 1  # 1-mask
                offset_map[0, c_ytr, c_xtr] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_ytr - 0.5  # height-Y offset
                offset_map[1, c_ytr, c_xtr] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_xtr - 0.5  # width-X offset
                offset_map[2, c_ytr, c_xtr] = 1  # 1-mask
                offset_map[0, c_ybr, c_xtr] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_ybr - 0.5  # height-Y offset
                offset_map[1, c_ybr, c_xtr] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_xtr - 0.5  # width-X offset
                offset_map[2, c_ybr, c_xtr] = 1  # 1-mask

        return pos_map, scale_map, offset_map, den_map

    def iou(self, box1, box2):
        in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
        in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
        inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
        iou = inter / union
        return iou


class RandomResizeFix(object):
    """
    Args:
        size: expected output size of each edge
        scale: scale factor
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, scale=(0.4, 1.5), interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale

    def __call__(self, img, gts, igs):
        # resize image
        w, h = img.size
        ratio = np.random.uniform(self.scale[0], self.scale[1])
        n_w, n_h = int(ratio * w), int(ratio * h)
        img = img.resize((n_w, n_h), self.interpolation)
        gts = gts.copy()
        igs = igs.copy()

        # resize label
        if len(gts) > 0:
            gts = np.asarray(gts, dtype=float)
            gts *= ratio

        if len(igs) > 0:
            igs = np.asarray(igs, dtype=float)
            igs *= ratio

        # random flip
        w, h = img.size
        if np.random.randint(0, 2) == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(gts) > 0:
                gts[:, [0, 2]] = w - gts[:, [2, 0]]
            if len(igs) > 0:
                igs[:, [0, 2]] = w - igs[:, [2, 0]]

        if h >= self.size[0]:
            # random crop
            img, gts, igs = self.random_crop(img, gts, igs, self.size, limit=16)
        else:
            # random pad
            img, gts, igs = self.random_pave(img, gts, igs, self.size, limit=16)

        return img, gts, igs

    @staticmethod
    def random_crop(img, gts, igs, size, limit=8):
        w, h = img.size
        crop_h, crop_w = size

        if len(gts) > 0:
            sel_id = np.random.randint(0, len(gts))
            sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
            sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
        else:
            sel_center_x = int(np.random.randint(0, w - crop_w + 1) + crop_w * 0.5)
            sel_center_y = int(np.random.randint(0, h - crop_h + 1) + crop_h * 0.5)

        crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
        crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
        diff_x = max(crop_x1 + crop_w - w, int(0))
        crop_x1 -= diff_x
        diff_y = max(crop_y1 + crop_h - h, int(0))
        crop_y1 -= diff_y
        cropped_img = img.crop((crop_x1, crop_y1, crop_x1 + crop_w, crop_y1 + crop_h))

        # crop detections
        if len(igs) > 0:
            igs[:, 0:4:2] -= crop_x1
            igs[:, 1:4:2] -= crop_y1
            igs[:, 0:4:2] = np.clip(igs[:, 0:4:2], 0, crop_w)
            igs[:, 1:4:2] = np.clip(igs[:, 1:4:2], 0, crop_h)
            keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & ((igs[:, 3] - igs[:, 1]) >= 8)
            igs = igs[keep_inds]

        if len(gts) > 0:
            before_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
            gts[:, 0:4:2] -= crop_x1
            gts[:, 1:4:2] -= crop_y1
            gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_w)
            gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_h)

            after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

            keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit) & (after_area >= 0.5 * before_area)
            gts = gts[keep_inds]

        return cropped_img, gts, igs

    @staticmethod
    def random_pave(img, gts, igs, size, limit=8):
        img = np.asarray(img)
        h, w = img.shape[0:2]
        pave_h, pave_w = size
        # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
        paved_image = np.ones((pave_h, pave_w, 3), dtype=img.dtype) * np.mean(img, dtype=int)
        pave_x = int(np.random.randint(0, pave_w - w + 1))
        pave_y = int(np.random.randint(0, pave_h - h + 1))
        paved_image[pave_y:pave_y + h, pave_x:pave_x + w] = img
        # pave detections
        if len(igs) > 0:
            igs[:, 0:4:2] += pave_x
            igs[:, 1:4:2] += pave_y
            keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & ((igs[:, 3] - igs[:, 1]) >= 8)
            igs = igs[keep_inds]

        if len(gts) > 0:
            gts[:, 0:4:2] += pave_x
            gts[:, 1:4:2] += pave_y
            keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
            gts = gts[keep_inds]

        return Image.fromarray(paved_image), gts, igs

