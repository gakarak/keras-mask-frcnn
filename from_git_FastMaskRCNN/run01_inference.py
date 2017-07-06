#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
__author__ = 'ar'


import functools
import os, sys
import time
import numpy as np
from time import gmtime, strftime
import tensorflow as tf
import tensorflow.contrib.slim as slim
import libs.configs.config_v1 as cfg
import libs.datasets.coco as coco
import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.nets_factory as network
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
from train.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, _get_init_fn, get_var_list_to_restore

import libs.layers.mask

resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS

import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8,allow_growth=True)
        # model_saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = coco.read('./data/coco/records/coco_train2014_00000-of-00033.tfrecord')
            with tf.control_dependencies([image, gt_boxes, gt_masks]):
                image, gt_boxes, gt_masks = coco_preprocess.preprocess_image(image, gt_boxes, gt_masks, is_training=False)
            # pathModel = './output/mask_rcnn/coco_resnet50_model.ckpt-49000'
            # pathModel = './models/coco_resnet50_model.ckpt-499999'
            # pathModel = './output/mask_rcnn/checkpoint'
            pathModel = './output/mask_rcnn/coco_resnet50_model.ckpt-158000'
            pathModelMeta = '{0}.meta'.format(pathModel)
            #
            # model_saver = tf.train.import_meta_graph(pathModelMeta)
            # model_saver.restore(sess, pathModel)
            # (1) prebuild model
            # logits, end_points = resnet50(image, 1000, is_training=False)
            # end_points['inputs'] = image
            # pyramid = pyramid_network.build_pyramid('resnet50', end_points)
            # outputs = pyramid_network.build_heads(pyramid, ih, iw, num_classes=81, base_anchors=9, is_training=False, gt_boxes=gt_boxes)
            logits, end_points, pyramid_map = network.get_network(FLAGS.network, image, weight_decay=FLAGS.weight_decay, is_training=False)
            outputs = pyramid_network.build(end_points, ih, iw, pyramid_map,
                                            num_classes=81,
                                            base_anchors=9,
                                            is_training=True,
                                            gt_boxes=gt_boxes, gt_masks=gt_masks,
                                            loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0])
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            #
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            # (2) load trained model
            # FLAGS.checkpoint_exclude_scopes = 'pyramid'
            # FLAGS.checkpoint_include_scopes = 'resnet_v1_50'
            # vars_to_restore = get_var_list_to_restore()
            # for var in vars_to_restore:
            #     print('\trestoring ', var.name)
            # model_saver = tf.train.Saver(vars_to_restore)
            model_saver = tf.train.Saver()
            model_saver.restore(sess=sess, save_path=pathModel)
            # print('Restored %d(%d) vars from %s' % (len(vars_to_restore), len(tf.global_variables()), pathModel))
            #
            outputs['image_np'] = image
            outputs['gt_boxes_np'] = gt_boxes
            outputs['gt_boxes'] = gt_boxes
            outputs['gt_masks'] = gt_masks
            outputs['num_instances'] = num_instances
            outputs['img_id'] = img_id
            #
            for dgbi in range(15):
                tret = sess.run(outputs)
                # [HARD-DEBUG]
                dbg_gt_boxes = tret['gt_boxes']
                dbg_gt_masks = tret['gt_masks']
                dbg_rois = tret['roi']['box']
                dbg_mask_height = 28
                dbg_mask_width  = 28
                dbg_num_classes = 81
                dbg_num_instances = tret['num_instances']
                dbg_img_id = tret['img_id']
                msk_encode = libs.layers.mask.encode(
                    gt_masks=dbg_gt_masks,
                    gt_boxes=dbg_gt_boxes,
                    rois=dbg_rois,
                    num_classes=dbg_num_classes,
                    mask_height=dbg_mask_height,
                    mask_width=dbg_mask_width
                )
                msk_encode_l = msk_encode[0]
                msk_encode_1 = msk_encode[1]
                msk_encode_2 = msk_encode[2]
                plt.figure()
                for ii in range(msk_encode_l.shape[0]):
                    lst_bg = []
                    lst_fg = []
                    tlbl = msk_encode_l[ii]
                    if tlbl<0:
                        plt.subplot(2, 2, 1)
                        plt.plot(np.sum(msk_encode_1[ii], axis=(0, 1)))
                        plt.subplot(2, 2, 2)
                        plt.plot(np.sum(msk_encode_2[ii], axis=(0, 1)))
                        lst_bg.append('id={0}/{1}'.format(ii, tlbl))
                    else:
                        plt.subplot(2, 2, 3)
                        plt.plot(np.sum(msk_encode_1[ii], axis=(0, 1)))
                        plt.subplot(2, 2, 4)
                        plt.plot(np.sum(msk_encode_2[ii], axis=(0, 1)))
                        lst_fg.append('id={0}/{1}'.format(ii, tlbl))
                plt.subplot(2, 2, 1), plt.legend(lst_bg)
                plt.subplot(2, 2, 3), plt.legend(lst_fg)
                # plt.show()
                #
                tmsk = dbg_gt_masks.copy()
                for ii in range(dbg_gt_masks.shape[0]):
                    tmsk[ii] *= (ii+1)
                tmsk = np.sum(tmsk, axis=0)
                #
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(0.5 * tret['image_np'][0] + 0.5)
                for ii in range(dbg_gt_masks.shape[0]):
                    tbox = dbg_gt_boxes[ii]
                    txy = (tbox[0], tbox[1])
                    tw = tbox[2] - tbox[0]
                    th = tbox[3] - tbox[1]
                    plt.gcf().gca().add_artist(plt.Rectangle(txy, tw, th, edgecolor='r', fill=False))
                plt.subplot(1, 2, 2)
                plt.imshow(tmsk)
                for ii in range(dbg_gt_masks.shape[0]):
                    tbox = dbg_gt_boxes[ii]
                    txy = (tbox[0], tbox[1])
                    tw = tbox[2] - tbox[0]
                    th = tbox[3] - tbox[1]
                    plt.gcf().gca().add_artist(plt.Rectangle(txy, tw, th, edgecolor='r', fill=False))
                #
                plt.show()
                print ('-')
            #
            realBBox = tret['gt_boxes_np']
            imagef32 = tret['image_np']
            imageu8 = (127.5 * (imagef32 + 1.0)).astype(np.uint8)
            #
            numTOP = 15
            arrROI = tret['roi']
            roiScore = arrROI['score'].reshape(-1)
            if len(roiScore)<numTOP:
                numTOP = len(roiScore)
            bestIdx = np.argsort(-roiScore)[:numTOP]
            bestBBox = arrROI['box'][bestIdx, :]
            #
            bestMsk = tret['mask']['mask'][bestIdx]
            bestCls = tret['mask']['cls'][bestIdx]
            bestScor = tret['mask']['score'][bestIdx]
            #
            plt.imshow(imageu8[0])
            ax = plt.gca()
            for ii in range(numTOP):
                x0 = bestBBox[ii, 0]
                y0 = bestBBox[ii, 1]
                dx = bestBBox[ii, 2] - x0
                dy = bestBBox[ii, 3] - y0
                ax.add_patch(patches.Rectangle((x0, y0), dx, dy, fill=False, edgecolor='red', linewidth=3))
                ax.text(x0, y0, 'cls={0}, score={1}'.format(bestCls[ii], bestScor[ii]))
            for ii in range(realBBox.shape[0]):
                x0 = realBBox[ii, 0]
                y0 = realBBox[ii, 1]
                dx = realBBox[ii, 2] - 1* x0
                dy = realBBox[ii, 3] - 1* y0
                ax.add_patch(patches.Rectangle((x0, y0), dx, dy, fill=False, edgecolor='yellow', linewidth=3))
            #
            plt.show()
            print ('---')