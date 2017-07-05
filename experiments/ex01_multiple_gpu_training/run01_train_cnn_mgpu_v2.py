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
    isDebug   = False
    batchSize = 256
    numEpochs = 10
    numReps   = 5
    lstGPU    = mgpu.get_available_gpus()
    numGPUs   = len(lstGPU)
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
    img_shape = x_train.shape[1:]
    #
    # (2) draw model
    if isDebug:
        model = buildModel(inpShape=img_shape, numCls=num_classes, denseNumHidden=128)
        model.summary()
        tmp_fimg_model='tmp_model_visualisation.png'
        keras.utils.plot_model(model, to_file=tmp_fimg_model, show_shapes=True)
        plt.imshow(skio.imread(tmp_fimg_model))
        plt.show()
    # (3) train model
    res_loss_trn = []
    res_loss_val = []
    res_acc_trn = []
    res_acc_val = []
    res_dt = []
    for irep in range(numReps):
        # (3.1) build model and place on GPU
        model = buildModel(inpShape=img_shape, numCls=num_classes, denseNumHidden=128)
        model = mgpu.make_parallel(model=model)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        # (3.2) train model
        t0 = time.time()
        model.fit(x_train, y_train,
                  batch_size=batchSize, # * numGPUs,
                  epochs=numEpochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
        dt = time.time() - t0
        ret_train = model.evaluate(x_train, y_train)
        ret_test  = model.evaluate(x_test,  y_test)
        res_dt.append(dt)
        res_loss_trn.append(ret_train[0])
        res_loss_val.append(ret_test[0])
        res_acc_trn.append(ret_train[1])
        res_acc_val.append(ret_test[1])
    #
    res_dt = np.array(res_dt)
    res_loss_trn = np.array(res_loss_trn)
    res_loss_val = np.array(res_loss_val)
    res_acc_trn = np.array(res_acc_trn)
    res_acc_val = np.array(res_acc_val)
    for irep in range(numReps):
        print ('\t{0}/{1} : dt={2} (s), loss(t/v)={3}/{4}, acc(t/v)={5}/{6}'
               .format(irep, numReps, res_dt[irep],
                       res_loss_trn[irep], res_loss_val[irep],
                       res_acc_trn[irep], res_acc_val[irep]))
    print ('#GPUs = {0}, #Epochs = {1}, BatchSize = {2}'.format(numGPUs, numEpochs, batchSize))
    print ('<dt>={0} (s), <loss>(t/v)={1}/{2}, <acc>(t/v)={3}/{4}'
           .format(res_dt.mean(),
                   res_loss_trn.mean(), res_loss_val.mean(),
                   res_acc_trn.mean(), res_acc_val.mean()))
