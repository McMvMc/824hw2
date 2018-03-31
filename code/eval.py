from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import torch
import cv2
import cPickle
import numpy as np

import network
from wsddn import WSDDN
from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms

from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir



def im_detect(net, image, rois):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data, im_scales = net.get_image_blob(image)
    rois = np.hstack((np.zeros((rois.shape[0],1)),rois*im_scales[0]))
    im_info = np.array(
        [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        dtype=np.float32)

    cls_prob = net(im_data, rois, im_info)
    scores = cls_prob.data.cpu().numpy()
    boxes = rois[:, 1:5] / im_info[0][2]

    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def eval_net(name, net, imdb, max_per_image=300, thresh=0.05, visualize=False,
             logger=None, step=None):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes+1)]

    output_dir = get_output_dir(imdb, name)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    roidb = imdb.roidb

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        rois = imdb.roidb[i]['boxes']
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, rois)
        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        if visualize:
            # im2show = np.copy(im[:, :, (2, 1, 0)])
            im2show = np.copy(im)

        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes+1):
            newj = j-1
            inds = np.where(scores[:, newj] > thresh)[0]
            cls_scores = scores[inds, newj]
            cls_boxes = boxes[inds, newj * 4:(newj + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if visualize:
                im2show = vis_detections(im2show, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc(average=False)

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))

        if visualize and np.random.rand()<0.01:
            # TODO: Visualize here using tensorboard
            # TODO: use the logger that is an argument to this function
            print('Visualizing')
            #cv2.imshow('test', im2show)
            #cv2.waitKey(1)

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    aps = imdb.evaluate_detections(all_boxes, output_dir)
    return aps