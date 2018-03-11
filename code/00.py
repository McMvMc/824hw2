from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import cv2
import numpy as np
import visdom

from datasets.factory import get_imdb

def vis_detections(im, dets, n = 1):
    """Visual debugging of detections."""
    for i in range(n):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
    return im


im_idx = 2018
SERVER_IP = 'http://127.0.0.1'
PORT_NUM = '8097'

imdb = get_imdb('voc_2007_trainval')
im_name = imdb.image_index[im_idx]
im_path = imdb.image_path_at(im_idx)

im = cv2.imread(im_path)
bbx = imdb.roidb[im_idx]['boxes']
im_bbx = vis_detections(im, bbx, 20)

im2 = cv2.imread(im_path)
gt_bbx = imdb.gt_roidb()[im_idx]['boxes']
im_gt_bbx = vis_detections(im2, gt_bbx)

vis = visdom.Visdom(server=SERVER_IP, port=PORT_NUM)
vis.image(np.transpose(im_bbx,(2,0,1)))
vis.image(np.transpose(im_gt_bbx,(2,0,1)))
