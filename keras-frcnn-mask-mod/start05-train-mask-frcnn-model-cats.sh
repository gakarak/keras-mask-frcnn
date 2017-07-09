#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

##wdir="${PWD}/data/VOCdevkit/VOC2012"
##wdir="${PWD}/data/VOCdevkit"

fidx='/home/ar/prj_datamola/frcnn_test_dataset_1/generated_data/info-idx-all.txt'

runpy='mask_frcnn_train.py'

python ${runpy} -o simple -p ${fidx}