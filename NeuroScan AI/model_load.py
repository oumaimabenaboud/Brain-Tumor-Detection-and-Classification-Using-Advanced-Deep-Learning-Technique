import os
import pickle
import tensorflow as tf
from keras.layers import Input
from tensorflow.keras.models import Model
import cv2
from preprocessing import nn_base, classifier_layer, rpn_layer, Config, rpn_to_roi, format_img, apply_regr, non_max_suppression_fast, get_real_coordinates
import numpy as np
import keras.backend as K


def load_model(config_path):
    # Load the configuration
    with open(config_path, 'rb') as f_in:
        C = pickle.load(f_in)

    # Modify config settings
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    # Define input shapes
    num_features = 512
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    # Input tensors
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # Define the base network (VGG here, but you can use Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # Define RPN (Region Proposal Network) based on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)

    # Define the classifier
    classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

    # Compile models
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    # Load weights
    print(f'Loading weights from {C.model_path}')
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    # Compile models
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    return model_rpn, model_classifier_only, C


def get_predictions(image_path, model_rpn, model_classifier_only, C, class_mapping, bbox_threshold=0.7, overlap_thresh=0.7):
    # Load and format the image
    img = cv2.imread(image_path)
    X, ratio = format_img(img, C)  # Assuming you have a format_img function to preprocess the image
    X = np.transpose(X, (0, 2, 3, 1))

    # Get output from the RPN and feature maps
    [Y1, Y2, F] = model_rpn.predict(X)

    # Get the proposed regions (R)
    R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=overlap_thresh)

    # Convert from (x1, y1, x2, y2) to (x, y, w, h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # Apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # Pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        # Predict using the classifier
        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs], verbose=0)

        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
            print("P_cls shape: ", P_cls.shape)
            print("P_cls values: ", P_cls[0, ii, :])

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    # Apply Non-Max Suppression (NMS)
    all_dets = []
    for key in bboxes:
        bbox = np.array(bboxes[key])
        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            real_x1, real_y1, real_x2, real_y2 = get_real_coordinates(ratio, x1, y1, x2, y2)

            # Add details of detections
            all_dets.append({
                "class": key,
                "prob": int(100 * new_probs[jk]),
                "bbox": (real_x1, real_y1, real_x2, real_y2)
            })

    return all_dets
