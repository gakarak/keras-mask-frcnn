#!/bin/bash

if [ -f ~/pyenv/env-keras2/bin/activate ]; then
    source ~/pyenv/env-keras2/bin/activate
fi

runpy="${PWD}/../frcnn_mod_train.py"
pydir=`dirname ${runpy}`

export PYTHONPATH="${pydir}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

##dataDir='/home/ar/prj_datamola/keras-frcnn.git/data/VOCdevkit'
dataDir='/media/data/datasets/detection_frcnn/data_MSCOCO/raw-data/train2014-idx-simple.txt'
##dataTyp='pascal_voc'
dataTyp='simple'

outDir="${dataDir}-out-train"
mkdir -p ${outDir}

# (1) find trained model
inpModel=`find ${outDir} -name '*_best.h5' | sed 's/\ //g' | head -n 1`
echo ":: [${inpModel}]"

if [ -n "${inpModel}" ]; then
    extOpt="--input_weight_path ${inpModel}"
    echo ":: found input model [${inpModel}]"
    echo ":: ext-option = [${extOpt}]"
else
    echo ":: cant find trained model in dir [${outDir}]"
fi

python ${runpy} ${extOpt} -p ${dataDir} -o ${dataTyp} -w ${outDir}

