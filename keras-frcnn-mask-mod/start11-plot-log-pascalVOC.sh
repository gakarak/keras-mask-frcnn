#!/bin/bash

##wdir='/media/data/datasets/detection_frcnn/data_MSCOCO/raw-data/train2014-idx-simple.txt-out-train'
wdir='/media/data/datasets/detection_frcnn/data_PascalVOC/VOCdevkit-out-train'
runpy='plot_logs_v1.py'


python ${runpy} ${wdir}

