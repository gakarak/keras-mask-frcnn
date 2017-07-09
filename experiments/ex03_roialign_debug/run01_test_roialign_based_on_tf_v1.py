#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.engine.topology import Layer
import keras.engine.topology as ktop
import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Flatten, MaxPool2D, Conv2D

import skimage.io as skio

#####################################
class RoiAligngConv_V1(Layer):
    '''ROI pooling (Align) layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        if self.dim_ordering == 'th':
            raise Exception('[RoiAligngConv_V1] Sorry, Theano backend curently not supported!')
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiAligngConv_V1, self).__init__(**kwargs)
    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]
    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            # return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels
            return None, self.num_rois, 4
    def call(self, x, mask=None):
        assert(len(x) == 2)
        img = x[0]
        rois = x[1]
        input_shape = K.cast(K.shape(img), 'float32')
        siz_h = K.cast(input_shape[1], tf.float32)
        siz_w = K.cast(input_shape[2], tf.float32)
        # tmp_bboxes = tf.zeros((self.num_rois, 4), tf.float32)
        tmp_bboxes = []
        tmp_bidx = [0] * self.num_rois
        # tmp_bboxes.append((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
        for roi_idx in range(self.num_rois-0):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            x1 = x / (siz_w - 1.)
            y1 = y / (siz_h - 1.)
            x2 = (x + w) / (siz_w - 1.)
            y2 = (y + h) / (siz_h - 1.)
            tmp_bboxes.append([y1, x1, y2, x2])
            # tmp_bboxes[roi_idx, 0] = x1
            # tmp_bboxes[roi_idx, 1] = y1
            # tmp_bboxes[roi_idx, 2] = x2
            # tmp_bboxes[roi_idx, 3] = y2
        tmp_bboxes = K.stack(tmp_bboxes)
        # k_bboxes = K.concatenate(tmp_bboxes, axis=1)
        ret = tf.image.crop_and_resize(img,
                                       tmp_bboxes,
                                       crop_size=[self.pool_size, self.pool_size],
                                       box_ind=tmp_bidx) #FIXME: explicit bix-index if 1-batch training
        ret = K.reshape(ret, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        # ret = K.zeros((self.num_rois, self.pool_size, self.pool_size, self.nb_channels), dtype='float32')
        # ret = K.reshape(ret, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        # ret = K.zeros((self.num_rois, self.pool_size, self.pool_size, self.nb_channels), dtype='float32')
        # ret = K.reshape(tmp_bboxes, (1, self.num_rois, 4))
        return ret

#####################################
def buildTestNet(inpShape=(256,256,1), num_roi=8, pool_size=64):
    inpDataImg = Input(shape=inpShape)
    inpDataROI = Input(shape=(num_roi, 4))
    x = inpDataImg
    # for ii in range(2):
    #     x = Conv2D(filters=16, kernel_size=(5,5), padding='same')(x)
    #     x = MaxPool2D(pool_size=(2,2))(x)
    x = RoiAligngConv_V1(pool_size=pool_size, num_rois=num_roi)([x, inpDataROI])
    # x = MaxPool2D(pool_size=(2, 2))(x)
    # x = MaxPool2D(pool_size=(2, 2))(x)
    # x = MaxPool2D(pool_size=(2, 2))(x)
    retModel = Model(inputs=[inpDataImg, inpDataROI], outputs=x)
    return retModel

#####################################
if __name__ == '__main__':
    # fimg = '../../data/doge2.jpg'
    fimg = '../../data/make_america_1.jpg'
    img = skio.imread(fimg, as_grey=True)

    dataImg = np.reshape(img, [1] + list(img.shape) + [1]).astype(np.float32)
    #
    nrow, ncol = img.shape
    numROI = 8
    poolSize = 128
    dataROI = np.expand_dims(np.array([[1,1, ncol-2, nrow-2] for xx in range(numROI)]), axis=0).astype(np.float32)
    inpShape = dataImg.shape[1:]
    #
    model = buildTestNet(inpShape=inpShape, num_roi=numROI, pool_size=poolSize)
    model.summary()
    model.compile(optimizer='sgd', loss='mae')
    tret = model.predict_on_batch([dataImg, dataROI])
    print ('-')