from __future__ import division
import os
import numpy as np
from scipy import io as scio
import json


def get_citypersons(root_dir='data/cityperson', type='train'):
    all_img_path = os.path.join(root_dir, 'images')
    all_anno_path = os.path.join(root_dir, 'annotations')
    rows, cols = 1024, 2048
    
    # all_img_path = '/mnt/raid/Talor/CSP_res2net/' + all_img_path[2:]
    # all_anno_path = '/mnt/raid/Talor/CSP_res2net/' + all_anno_path[2:]

    anno_path = os.path.join(all_anno_path, 'anno_' + type + '.mat')
    res_path = os.path.join('data/cache/cityperson', type)
    image_data = []
    annos = scio.loadmat(anno_path)
    index = 'anno_' + type + '_aligned'
    valid_count = 0
    iggt_count = 0
    box_count = 0

    for l in range(len(annos[index][0])):
        anno = annos[index][0][l]
        cityname = anno[0][0][0][0].encode()
        imgname = anno[0][0][1][0].encode()
        gts = anno[0][0][2]
        img_path = os.path.join(all_img_path, type + '/' + cityname + '/' + imgname)
        boxes = []
        ig_boxes = []
        vis_boxes = []
        for i in range(len(gts)):
            label, x1, y1, w, h = gts[i, :5]
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            w, h = min(int(w), cols - x1 - 1), min(int(h), rows - y1 - 1)
            xv1, yv1, wv, hv = gts[i, 6:]
            xv1, yv1 = max(int(xv1), 0), max(int(yv1), 0)
            wv, hv = min(int(wv), cols - xv1 - 1), min(int(hv), rows - yv1 - 1)

            if label == 1 and h >= 50:
                box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
                boxes.append(box)
                vis_box = np.array([int(xv1), int(yv1), int(xv1) + int(wv), int(yv1) + int(hv)])
                vis_boxes.append(vis_box)
            else:
                ig_box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
                ig_boxes.append(ig_box)
        boxes = np.array(boxes)
        vis_boxes = np.array(vis_boxes)
        ig_boxes = np.array(ig_boxes)

        if len(boxes) > 0:
            valid_count += 1
        annotation = {}
        annotation['filepath'] = img_path
        box_count += len(boxes)
        iggt_count += len(ig_boxes)
        annotation['bboxes'] = boxes
        annotation['vis_bboxes'] = vis_boxes
        annotation['ignoreareas'] = ig_boxes
        image_data.append(annotation)

    return image_data

def get_eurocity(root_dir='data/cityperson', type='train'):
    f1 = json.load(open('/mnt/raid/Talor/APD_test/day_train_all.json'))

    all_img_path = "/mnt/raid/Talor/dataset/eurocity"

    image_data = []
    images = f1["images"]
    anno = f1["annotations"]
    i = 0
    rows, cols = 1024, 1920

    for img in images:
        img_path = os.path.join(all_img_path, img['file_name'])
        boxes = []
        ig_boxes = []
        index = img['id']

        while(i < len(anno) and anno[i]['image_id'] == index):
            bbox = anno[i]['bbox']
            x1, y1 = max(int(bbox[0]), 0), max(int(bbox[1]), 0)
            w, h = min(int(bbox[2]), cols - x1 - 1), min(int(bbox[3]), rows - y1 - 1)
            box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
            boxes.append(box)
            i += 1
        boxes = np.array(boxes)

        annotation = {}
        annotation['filepath'] = img_path
        annotation['bboxes'] = boxes
        annotation['ignoreareas'] = ig_boxes
        image_data.append(annotation)
    return image_data
