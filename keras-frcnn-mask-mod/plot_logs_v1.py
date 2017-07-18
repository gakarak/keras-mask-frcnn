#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
import glob
import skimage.io as skio

#############################
def press_event(event):
    global currentLogIdx, numLogs
    print('\t--> press-event: {0}'.format(event.key))
    sys.stdout.flush()
    if event.key == 'k':
        currentLogIdx -= 1
        if currentLogIdx < 0:
            currentLogIdx = numLogs - 1
    if event.key == 'l':
        currentLogIdx += 1
        if currentLogIdx >= numLogs:
            currentLogIdx = 0


if __name__ == '__main__':
    wdir='/mnt/data1T2/datasets2/mscoco/test-logs'
    if len(sys.argv) > 1:
        wdir = sys.argv[1]
    else:
        print ('*** Usage: {0} [/path/to/dir-with-logs]'.format(os.path.basename(sys.argv[0])))
    if not os.path.isdir(wdir):
        raise Exception('\tCant find directory with logs! [{0}]'.format(wdir))
    currentLogIdx = 0
    numLogs = 0
    #
    logSuffix = '*-trainlog*.txt'
    # logSuffix = '*-trainlog.txt'
    listLogs = glob.glob('{0}/*{1}'.format(wdir, logSuffix))
    numLogs = len(listLogs)
    if numLogs < 1:
        raise Exception('\tCant find LOG file in format [*{0}] in directory [{1}]'.format(logSuffix, wdir))
    plt.figure()
    ptrFigure = plt.gcf()
    ptrFigure.canvas.mpl_connect('key_press_event', press_event)
    cnt = 0
    #
    while True:
        flog = listLogs[currentLogIdx]
        bname = os.path.basename(flog)
        ptrFigure.canvas.set_window_title('Current: [{0}]'.format(bname))
        lstKeys = ['bboxes', 'regr', 'cls', 'acc', 'loss']

        if os.path.isfile(flog):
            data = pd.read_csv(flog)
            dataMap = {}
            for kk in lstKeys:
                tmpMap = {}
                for dkey in data.keys():
                    if dkey.count(kk):
                        tmpMap[dkey] = data[dkey].as_matrix()
                if len(tmpMap.keys())>0:
                    dataMap[kk] = tmpMap
            numPlots = len(dataMap.keys())
            plt.clf()
            for ii, (kk,vv) in enumerate(dataMap.items()):
                plt.subplot(1, numPlots, ii+1)
                for pdata in vv.values():
                    plt.plot(pdata)
                plt.grid(True)
                plt.legend(vv.keys(), loc='best')
                plt.title('::{0}'.format(kk))
            plt.show(block=False)
            plt.pause(5)
            print (':: update: [{0}], current log: [{1}]'.format(cnt, bname))
            cnt += 1
        else:
            print ('*** WARNING *** cant find log-file [{0}]'.format(flog))
