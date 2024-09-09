import math

class Config:
    def __init__(self):
        self.verbose = True
        self.network = 'vgg'
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False
        self.anchor_box_scales = [64, 128, 256]
        self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
        self.im_size = 300
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0
        self.num_rois = 4
        self.rpn_stride = 16
        self.balanced_classes = False
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5
        self.class_mapping = None
        self.model_path = None