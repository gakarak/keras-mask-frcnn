#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

##wdir="${PWD}/data/VOCdevkit/VOC2012"
wdir="${PWD}/data/VOCdevkit"

runpy='train_frcnn.py'

python ${runpy} -p ${wdir}