#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

# code from github-repo: https://github.com/kuza55/keras-extras/

from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf
from tensorflow.python.client import device_lib as tf_dev_lib

###############################
def get_available_gpus():
    local_device_protos = tf_dev_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

###############################
"""
    Keras multible-GPU training: data parallelism approach   
    @:param model - keras model
    @:param gpu_count - number of GPU, if None - the use all available GPU (by default)
"""
def make_parallel(model, gpu_count=None):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)
    if gpu_count is None:
        list_gpus = get_available_gpus()
    else:
        list_gpus = ['/gpu:%d' % xx for xx in range(gpu_count)]
    num_gpus = len(list_gpus)
    if num_gpus<1:
        raise Exception('Invalid number available GPUs for training: #{0}'.format(num_gpus))
    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])
    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i, gpu_d in enumerate(list_gpus):
        with tf.device(gpu_d):
            with tf.name_scope('gpu_tower_%d' % i) as scope:
                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])
    # merge outputs on CPU
    print ("""
**** Keras trained in a Multi-GPU mode (Data parallelism) ****
#GPUs = {0}, GPUs: [{1}]
    """.format(num_gpus, list_gpus))
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
        return Model(input=model.inputs, output=merged)

if __name__ == '__main__':
    pass