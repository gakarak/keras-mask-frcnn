#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import cv2
import skimage.io as skio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#################################################
def buildImageWithRotScaleAroundCenter(pimg, pcnt, pangDec, pscale, pcropSize, isDebug=False, pborderMode = cv2.BORDER_REPLICATE):
    # (1) precalc parameters
    angRad = (np.pi / 180.) * pangDec
    cosa = np.cos(angRad)
    sina = np.sin(angRad)
    # (2) prepare separate affine transformation matrices
    matShiftB = np.array([[1., 0., -pcnt[0]], [0., 1., -pcnt[1]], [0., 0., 1.]])
    matRot = np.array([[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]])
    matShiftF = np.array([[1., 0., +pcnt[0]], [0., 1., +pcnt[1]], [0., 0., 1.]])
    matScale = np.array([[pscale, 0., 0.], [0., pscale, 0.], [0., 0., 1.]])
    matShiftCrop = np.array([[1., 0., pcropSize[0] / 2.], [0., 1., pcropSize[1] / 2.], [0., 0., 1.]])
    # matTotal_OCV = matShiftF.dot(matRot.dot(matScale.dot(matShiftB)))
    # (3) build total-matrix
    matTotal = matShiftCrop.dot(matRot.dot(matScale.dot(matShiftB)))
    if isDebug:
        print ('(1) mat-shift-backward = \n{0}'.format(matShiftB))
        print ('(2) mat-scale = \n{0}'.format(matScale))
        print ('(3) mat-rot = \n{0}'.format(matRot))
        print ('(4) mat-shift-forward = \n{0}'.format(matShiftF))
        print ('(5) mat-shift-crop = \n{0}'.format(matShiftCrop))
        print ('---\n(*) mat-total = \n{0}'.format(matTotal))
    # (4) warp image with total affine-transform
    if (pimg.ndim>2) and (pimg.shape[-1]>3):
        pimg0 = pimg[:, :, :3]
        pimg1 = pimg[:, :, -1]
        imgRet0 = cv2.warpAffine(pimg0, matTotal[:2, :], pcropSize, flags=cv2.INTER_CUBIC, borderMode=pborderMode)
        imgRet1 = cv2.warpAffine(pimg1, matTotal[:2, :], pcropSize, flags=cv2.INTER_NEAREST, borderMode=pborderMode)
        imgRet = np.dstack((imgRet0, imgRet1))
    else:
        imgRet = cv2.warpAffine(pimg, matTotal[:2, :], pcropSize, borderMode=pborderMode)
    return imgRet

#################################################
def getRandomInRange(vrange, pnum=None):
    vmin,vmax = vrange
    if pnum is None:
        trnd = np.random.rand()
    else:
        trnd = np.random.rand(pnum)
    ret = vmin + (vmax-vmin)*trnd
    return ret

#################################################
def generateRandomizedScene(pathBG, pathIdxFG, p_rangeSizes=(64, 256), p_rangeAngle=(0, 36), p_rangeNumSamples=(3, 16)):
    imgBG = skio.imread(pathBG)
    mskBGS = np.zeros(imgBG.shape[:2])
    mskBGI = np.zeros(imgBG.shape[:2])
    numSamples = np.random.randint(p_rangeNumSamples[0], p_rangeNumSamples[1])
    cntInstance = 0
    pass

#################################################
if __name__ == '__main__':
    fidxFG = '/home/ar/prj_datamola/frcnn_test_dataset_1/02_data_resized_512x512_mask/idx.txt'
    fidxBG = '/home/ar/prj_datamola/frcnn_test_dataset_1/03_data_backgrounds/idx.txt'
    #
    def_range_sizes=(64, 256)
    def_range_angle=(0,  24)
    #
    dataFG = pd.read_csv(fidxFG, sep=',')
    dataBG = pd.read_csv(fidxBG, sep=',')
    numImgsFG = len(dataFG)
    numImgsBG = len(dataBG)
    wdirFG = os.path.dirname(fidxFG)
    wdirBG = os.path.dirname(fidxBG)
    arrLbl = dataFG['clsid'].as_matrix()
    # (1) FG
    pathImgsFG = dataFG['path'].as_matrix()
    pathImgsFG = [os.path.join(wdirFG, xx) for xx in pathImgsFG]
    # (2) BG
    pathImgsBG = dataBG['path'].as_matrix()
    pathImgsBG = [os.path.join(wdirBG, xx) for xx in pathImgsBG]
    #
    for ipathBG, pathBG in enumerate(pathImgsBG):
        timgBG = skio.imread(pathBG)
        for ipathFG, pathFG in enumerate(pathImgsFG):
            #
            timgFG = skio.imread(pathFG)
            tmskFG = timgFG[:, :, 3]
            minSiz = float(np.min(timgFG.shape[:2]))
            (mskPC, mskR) = cv2.minEnclosingCircle(np.array(np.where(tmskFG > 128)).transpose())
            #
            rndSiz = int(getRandomInRange(def_range_sizes))
            rndAng = getRandomInRange(def_range_angle)
            pScale = float(rndSiz)/(2.*mskR)
            newImg = buildImageWithRotScaleAroundCenter(timgFG, mskPC[::-1], rndAng, pScale, (rndSiz, rndSiz))
            #
            plt.subplot(2, 2, 1)
            plt.imshow(timgFG)
            plt.plot(mskPC[1], mskPC[0], 'o')
            plt.gcf().gca().add_artist(plt.Circle(mskPC[::-1], mskR, edgecolor='r', fill=False))
            #
            plt.subplot(2, 2, 2), plt.imshow(tmskFG), plt.plot(mskPC[1], mskPC[0], 'o')
            plt.plot(mskPC[1], mskPC[0], 'o')
            plt.gcf().gca().add_artist(plt.Circle(mskPC[::-1], mskR, edgecolor='r', fill=False))
            #
            plt.subplot(2, 2, 3),
            plt.imshow(newImg)
            #
            plt.subplot(2, 2, 4),
            plt.imshow(newImg[:,:,3])
            plt.show()

            print ('-')