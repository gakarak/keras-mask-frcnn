from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_mfrcnn import config
import keras_mfrcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_mfrcnn import roi_helpers

import matplotlib.pyplot as plt

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))
    return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

numClasses = len(C.class_mapping)
numClassesFG = len(C.class_mapping) - 1

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (1024, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)
maskout = nn.maskout(feature_map_input, roi_input, C.num_rois, nb_classes=numClassesFG, trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_maskout = Model([feature_map_input, roi_input], maskout)
# model_maskout_only = Model([feature_map_input, roi_input], maskout)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights(C.model_path_best, by_name=True)
model_classifier.load_weights(C.model_path_best, by_name=True)
model_maskout.load_weights(C.model_path_best, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')
model_maskout.compile(optimizer='sgd', loss='mse') #FIXME: fake 'compiling' parameters

all_imgs = []

classes = {}

bbox_threshold = 0.5

isDebug = False
isRounded = True

visualise = True

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    # if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
    # 	continue
    if not img_name.lower().endswith(('.jpg')):
        continue
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path,img_name)

    img = cv2.imread(filepath)

    X, ratio = format_img(img, C)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)


    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}
    masks = {}

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
        retMask = model_maskout.predict([F, ROIs]).reshape([C.num_rois, C.maskSize, C.maskSize, numClassesFG])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_num = np.argmax(P_cls[0, ii, :])
            cls_name = class_mapping[cls_num]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []
                masks[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]


            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            tmsk = retMask[ii]
            x1y1x2y2 = [C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)]
            bboxes[cls_name].append(x1y1x2y2)
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
            masks[cls_name].append(tmsk[:,:,cls_num])

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs, idx_selected = roi_helpers.non_max_suppression_fast(bbox,
                                                                    np.array(probs[key]),
                                                                    overlap_thresh=0.5,
                                                                    isRounded=isRounded)
        cnt = 0
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            box_mask = masks[key][idx_selected[jk]]

            dx = int(real_x2 - real_x1)
            dy = int(real_y2 - real_y1)
            if dx<2:
                dx = 2
            if dy<2:
                dy = 2

            box_mask_r = cv2.resize(box_mask, (dx, dy))
            tout_msk = np.zeros(img.shape[:2], dtype=np.uint8)
            tout_prv = img.copy()
            dy_ = min(dy, box_mask_r.shape[0])
            dx_ = min(dx, box_mask_r.shape[1])
            try:
                tout_msk[real_y1:real_y1+dy_, real_x1:real_x1+dx_] = (255.*box_mask_r[:dy_, :dx_]).astype(np.uint8)
                tout_prv[:,:,2] = tout_msk
            except Exception as err:
                print ('!!!ERROR!!! : [{0}]'.format(err))
                continue

            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
            cv2.rectangle(tout_msk, (real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)
            cv2.rectangle(tout_prv, (real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)

            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            #
            #
            cv2.rectangle(tout_msk, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(tout_prv, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            #
            cv2.rectangle(tout_msk, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(tout_prv, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            #
            cv2.putText(tout_msk, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            cv2.putText(tout_prv, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            fout_msk = '{0}-{1:03d}-{2}-msk.png'.format(filepath, cnt, key)
            fout_prv = '{0}-{1:03d}-{2}-prv.png'.format(filepath, cnt, key)
            cv2.imwrite(fout_msk, tout_msk)
            cv2.imwrite(fout_prv, tout_prv)
            cnt +=1
            print ('-')
    print('Elapsed time = {}'.format(time.time() - st))
    print(all_dets)
    if isDebug:
        plt.imshow(img)
        plt.show()
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
