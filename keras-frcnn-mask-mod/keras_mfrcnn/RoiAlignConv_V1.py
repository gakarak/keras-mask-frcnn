from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiAligngConv_V1(Layer):
    '''ROI pooling (Align) layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        if self.dim_ordering == 'th':
            raise Exception('[RoiAligngConv_V1] Sorry, Theano backend curently not supported!')
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiAligngConv_V1, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)
        siz_h = K.cast(input_shape[0], tf.float32)
        siz_w = K.cast(input_shape[1], tf.float32)
        tmp_bboxes = []
        tmp_bidx = [0] * self.num_rois
        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            x1 = x / (siz_w - 1.)
            y1 = y / (siz_h - 1.)
            x2 = (x + w) / (siz_w - 1.)
            y2 = (y + h) / (siz_h - 1.)
            tmp_bboxes.append([y1, x1, y2, x2])
        # k_bboxes = K.concatenate(tmp_bboxes, axis=1)
        ret = tf.image.crop_and_resize(img,
                                       tmp_bboxes,
                                       crop_size=[self.pool_size, self.pool_size],
                                       box_ind=tmp_bidx) #FIXME: explicit bix-index if 1-batch training
        ret = K.reshape(ret, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return ret
        #
        # outputs = []
        #
        # for roi_idx in range(self.num_rois):
        #
        #     x = rois[0, roi_idx, 0]
        #     y = rois[0, roi_idx, 1]
        #     w = rois[0, roi_idx, 2]
        #     h = rois[0, roi_idx, 3]
        #
        #     row_length = w / float(self.pool_size)
        #     col_length = h / float(self.pool_size)
        #
        #     num_pool_regions = self.pool_size
        #
        #     #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
        #     # in theano. The theano implementation is much less efficient and leads to long compile times
        #
        #     if self.dim_ordering == 'th':
        #         raise Exception('[RoiAligngConv_V1] Theano backend is not supported yet')
        #     elif self.dim_ordering == 'tf':
        #         # (1) subpixel cropping!
        #         # x = K.cast(x, 'int32')
        #         # y = K.cast(y, 'int32')
        #         # w = K.cast(w, 'int32')
        #         # h = K.cast(h, 'int32')
        #
        #         # rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
        #         rs = tf.image.crop_and_resize()
        #         outputs.append(rs)
        #
        # final_output = K.concatenate(outputs, axis=0)
        # final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        #
        # if self.dim_ordering == 'th':
        #     final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        # else:
        #     final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))
        #
        # return final_output
