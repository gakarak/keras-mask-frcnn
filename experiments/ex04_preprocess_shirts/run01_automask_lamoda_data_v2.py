#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import cv2
import os
import skimage.io as skio
# from skimage.morphology import disk
# from skimage.filters.rank import median
import skimage.morphology as skmorph
import skimage.filters as skflt
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import KMeans
import sklearn.metrics as skmt

import pandas as pd

import multiprocessing as mp
import multiprocessing.pool
from concurrent.futures import ThreadPoolExecutor

def proc_fun(params):
    isDebug = False
    pimg   = params[0]
    ipimg  = params[1]
    numImg = params[2]
    print ('[{0}/{1}] : {2}'.format(ipimg, numImg, pimg))
    img = skio.imread(pimg)
    imgNew = []
    for ii in range(img.shape[-1]):
        imgNew.append(skflt.rank.median(img[:,:,ii], skmorph.disk(3)))
    img = np.dstack(imgNew)
    # procSPX = cv2.ximgproc.createSuperpixelSLIC(img, cv2.ximgproc.SLIC, region_size=36, ruler=60.)
    # procSPX = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2], 5000, 4)
    # procSPX.iterate(img, 1000)
    procSPX = cv2.ximgproc.createSuperpixelLSC(img.copy(), region_size=18)
    procSPX.iterate(20)
    segments_spx = procSPX.getLabels().copy()
    procSPX.clear()
    del procSPX
    #
    imgDSC = img.reshape(-1,3)
    numSPX = len(np.unique(segments_spx))
    dscSPX = np.zeros((numSPX,6))
    for ispx in range(numSPX):
        tdsc = img[segments_spx==ispx]
        if len(tdsc)>0:
            tm = np.mean(tdsc,axis=0)
            tv = np.std(tdsc, axis=0)
            dscSPX[ispx, :3] = tm
            dscSPX[ispx, 3:] = tv
    #
    kmeans = KMeans(n_clusters=2).fit(dscSPX)
    idxBG = 0
    dscC0 = dscSPX[kmeans.labels_ == idxBG, :3]
    dscC1 = dscSPX[kmeans.labels_ != idxBG, :3]
    dstC0 = skmt.pairwise_distances(dscC0, [[255., 255., 255.]], metric='l1').mean()
    dstC1 = skmt.pairwise_distances(dscC1, [[255., 255., 255.]], metric='l1').mean()
    # swap class-label if mean-distance near to 'white' color
    if dstC1<dstC0:
        idxBG = 1
    tmsk = np.zeros(img.shape[:2])
    for ispx in range(numSPX):
        tmsk[segments_spx == ispx] = int(kmeans.labels_[ispx] != idxBG)
    # print ('-')
    foutImg = '{0}-msk.png'.format(pimg)
    timgMasked = np.dstack((img, (255.*tmsk).astype(np.uint8)))
    # segments_slic = slic(img, n_segments=3000, compactness=20, max_iter=30)
    if isDebug:
        plt.subplot(1, 3, 1)
        plt.imshow(segments_spx)
        plt.subplot(1, 3, 2)
        plt.imshow(mark_boundaries(img, segments_spx))
        plt.subplot(1, 3, 3)
        plt.imshow(timgMasked)
        plt.show()
    else:
        skio.imsave(foutImg, timgMasked)

if __name__ == '__main__':

    fidx = '/home/ar/data/PRJ_DATAMOLA/shirts01_lamoda.ru/idx.txt'
    wdir = os.path.dirname(fidx)
    dataCSV = pd.read_csv(fidx)
    pathImg = dataCSV['path'].as_matrix()
    pathImg = [os.path.join(wdir, xx) for xx in pathImg]
    numImg = len(pathImg)
    #
    procPool = mp.Pool(processes=1)
    for ipimg, pimg in enumerate(pathImg):
        procPool.apply_async(proc_fun, [(pimg, ipimg, numImg)])
    #
    print ('----- WAIT THREADS -----')
    procPool.close()
    procPool.join()
    # print ('-')


