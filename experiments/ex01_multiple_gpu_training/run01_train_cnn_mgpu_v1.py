#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import tensorflow as tf

import lib_multi_gpu as mgpu

import time
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D

###############################
def buildModel(inpShape=(32,32,3), numCls=10, numConv=4, numConvRep=2, convFltSize=(3,3), convFltNum=8, actfun='relu', denseNumHidden=None):
    # (1) input
    # inpLayer = InputLayer(input_shape=inpShape)
    inpLayer = Input(inpShape)
    x = inpLayer
    # (2) conv-part
    sizFlt = convFltSize
    numFlt = convFltNum
    for ii in range(numConv):
        for rr in range(numConvRep):
            x = Conv2D(filters=(numFlt * (2 ** ii)),
                       kernel_size=sizFlt,
                       strides=(1, 1),
                       padding='same',
                       activation=actfun)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    # (3) classifier part
    if denseNumHidden is not None:
        if isinstance(denseNumHidden, list):
            for numHidden in denseNumHidden:
                x = Dense(units=numHidden, activation=actfun)(x)
        else:
            x = Dense(units=denseNumHidden, activation=actfun)(x)
    #
    x = Dense(units=numCls, activation='softmax')(x)
    model = Model(inputs=inpLayer, outputs=x)
    return model

###############################
if __name__ == '__main__':
    isDebug = False
    batchSize = 256
    numEpochs = 10
    lstGPU = mgpu.get_available_gpus()
    numGPUs = len(lstGPU)
    print ('** Available GPUs: [{0}]'.format(lstGPU))
    #
    # (1) Prepare dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    num_classes = len(np.unique(y_train))
    # (1.1) convert to categorical
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    #
    # (2) build model and place on GPU
    img_shape = x_train.shape[1:]
    model = buildModel(inpShape=img_shape, numCls=num_classes, denseNumHidden=128)
    model.summary()
    model = mgpu.make_parallel(model=model)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    # (3) draw model
    if isDebug:
        tmp_fimg_model='tmp_model_visualisation.png'
        keras.utils.plot_model(model, to_file=tmp_fimg_model, show_shapes=True)
        plt.imshow(skio.imread(tmp_fimg_model))
        plt.show()
    # (4) train model
    t0 = time.time()
    model.fit(x_train, y_train,
              batch_size=batchSize, # * numGPUs,
              epochs=numEpochs,
              validation_data=(x_test, y_test),
              shuffle=True)
    dt = time.time() - t0
    print ('training time: {0} s, #epochs = {1}, #GPUs = {2}'.format(dt, numEpochs, numGPUs))
