#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import tensorflow as tf
from tensorflow.python.client import device_lib as tf_dev_lib

import lib_multi_gpu as mgpu

###############################
def get_available_gpus():
    local_device_protos = tf_dev_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

###############################
if __name__ == '__main__':
    lstGPU = get_available_gpus()
    print ('-')