import cv2
import numpy as np
import copy
import os


def augment_mask(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)


    pathImg = img_data_aug['filepath']
    pathMsk_I = '{0}-mski.png'.format(pathImg[:-8])
    pathMsk_S = '{0}-msks.png'.format(pathImg[:-8])

    img = cv2.imread(pathImg)
    # FIXME: remove intencity shift for visualization in image-viewers applications, real classes start fro "1", "0" - for background
    msk_i = cv2.imread(pathMsk_I, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    msk_i[msk_i>0] -= 128
    msk_s = cv2.imread(pathMsk_S, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    msk_s[msk_s > 0] -= 127

    if augment:
        rows, cols = img.shape[:2]

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            msk_i = cv2.flip(msk_i, 1)
            msk_s = cv2.flip(msk_s, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            msk_i = cv2.flip(msk_i, 0)
            msk_s = cv2.flip(msk_s, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if config.rot_90:
            angle = np.random.choice([0, 90, 180, 270], 1)[0]
            if angle == 270:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 0)
                msk_i = np.transpose(msk_i, (1, 0))
                msk_i = cv2.flip(msk_i, 0)
                msk_s = np.transpose(msk_s, (1, 0))
                msk_s = cv2.flip(msk_s, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
                msk_i = cv2.flip(msk_i, -1)
                msk_s = cv2.flip(msk_s, -1)
            elif angle == 90:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 1)
                msk_i = np.transpose(msk_i, (1, 0, 2))
                msk_i = cv2.flip(msk_i, 1)
                msk_s = np.transpose(msk_s, (1, 0, 2))
                msk_s = cv2.flip(msk_s, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img, msk_i, msk_s
