#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Reuben
# @Date:   2025-02-24 12:08PM EST 2025
# @Last Modified by:   Reuben
# @Last Modified time: 2025-02-24 12:08PM EST 2025
# Features:
# 1. Input the image path, intrinsic matrix, and distortion coefficients and camera installation height
# 2. Detect the lane lines using the trained model assigned
# 3. Choose the suitable lanes for Vanishing point calculation
# 3. Calculate the vanishing point
# 4. Calculate the camera pitch angle and yaw angle
# 5. Calculate the camera extrinsic matrix
# 6. Verify the vanishing point by plotting the camera trajectory
# 7. Output the result

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import time
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger
from lanenet_model.lanenet_postprocess import COLOR_MAP


CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lane_vp_calibration')

RESIZE_IMAGE_HEIGHT = CFG.AUG.TRAIN_CROP_SIZE[1]
RESIZE_IMAGE_WIDTH = CFG.AUG.TRAIN_CROP_SIZE[0]

# filter the lanes those are not straight enough
STRAIGHT_FIT_PARAM_THRESHOLD = [0.01, 0.1]

def init_args():
    parser = argparse.ArgumentParser(description="Lane Vanishing Point Calibration")
    parser.add_argument("--image_path", type=str, default="data/route28/00000.jpg", help="Path to the image")
    parser.add_argument("--weights_path", type=str, default="model/lane_detection_model.pth", help="Path to the trained model")
    parser.add_argument("--config_path", type=str, default="data/lane_vp_calib_config.json", help="Path to the config file")
    parser.add_argument("--output_dir", type=str, default="output/", help="Path to the output file")
    return parser.parse_args()


def load_calibration_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        # verify the config
        if "front-camera-intrinsic" not in config:
            raise ValueError("Invalid config file, front-camera-intrinsic not found")
        if "param" not in config["front-camera-intrinsic"]:
            raise ValueError("Invalid config file, param not found")
        if "cam_K" not in config["front-camera-intrinsic"]["param"]:
            raise ValueError("Invalid config file, cam_K not found")
        if "cam_dist" not in config["front-camera-intrinsic"]["param"]:
            raise ValueError("Invalid config file, cam_dist not found")
        return config

def image_preprocessing(image_path, cam_K, cam_dist):
    # Return the preprocessed image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # TODO: calculate the undistorted image
    undistorted_image = image   

    original_image = undistorted_image.copy()

    resized_image = cv2.resize(undistorted_image, (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    normalized_image = resized_image / 127.5 - 1.0

    # debug: set the bottom 1/3 of the image to 0
    resized_height = resized_image.shape[0]
    resized_image[int(resized_height * 0.75):, :] = 0

    cv2.imwrite("resized_image.jpg", resized_image)

    return normalized_image, original_image

def lane_detection_step1(weights_path, image_path, cam_K, cam_dist):
    # Return the detected binary segmentation image and the instance segmentation image

    # Initialize tensorflow session and model once
    tf.reset_default_graph()
    
    LOG.info('Start reading image and preprocessing')
    normalized_image, original_image = image_preprocessing(image_path, cam_K, cam_dist)
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    try:
        with sess.as_default():
            saver.restore(sess=sess, save_path=weights_path)

            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [normalized_image]}
            )
            t_cost = time.time() - t_start
            LOG.info('Finished lane inference, cost time: {:.5f}s'.format(t_cost))
            print("binary_seg_image.shape:")
            print(binary_seg_image.shape)
            print("instance_seg_image.shape:")
            print(instance_seg_image.shape)
    except Exception as e:
        LOG.error('Error in lane detection step 1: {}'.format(e))
        return None, None
    return original_image,binary_seg_image, instance_seg_image


def lane_detection_step2(original_image, binary_seg_image, instance_seg_image):
    # find the suitable lanes for vanishing point calculation on the 2D image

    h_orig, w_orig = original_image.shape[:2]

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=original_image, # 源图像
            with_lane_fit=False,
            with_2d_lane_fit=True
        )
    if postprocess_result['fit_params'] is None or len(postprocess_result['fit_params']) == 0:
        LOG.error('No any lane detected')
        return postprocess_result
    
    print("postprocess_result['fit_params']:")
    print(postprocess_result['fit_params'])

    straight_lanes = []
    straight_lane_fit_params = []
    straight_lane_colors = []
    for lane_index in range(len(postprocess_result['fit_params'])):
        fit_param = postprocess_result['fit_params'][lane_index]
        resized_lane_coords = postprocess_result['resized_lane_coords'][lane_index]
        lane_color = COLOR_MAP[lane_index]

        # Convert tuple coordinates to numpy array
        # Original format: (y_coords, x_coords)
        lane_points = np.vstack(resized_lane_coords).T  # Convert to Nx2 array [x, y]
        
        # filter the lanes those are not straight enough
        if abs(fit_param[0]) > STRAIGHT_FIT_PARAM_THRESHOLD[0]:
            continue
        # 车的行进方向应该与车道线平行，所以车道线的斜率应该保持在一定的区间范围内
        # 斜率范围
        #if abs(fit_param[1]) > STRAIGHT_FIT_PARAM_THRESHOLD[1]:
        #    continue
        # TODO: 过滤车道线覆盖的区域大小
        # TODO: the other filter conditions

        straight_fit_param = np.polyfit(lane_points[:, 0], lane_points[:, 1], 1)
        straight_lanes.append(lane_points)
        straight_lane_fit_params.append(straight_fit_param)
        straight_lane_colors.append(lane_color)
        # draw the fit straight lane on the original image
        y_start = 0
        y_end = h_orig
        x_start = straight_fit_param[0] * y_start + straight_fit_param[1]
        x_end = straight_fit_param[0] * y_end + straight_fit_param[1]
        pt1 = (int(x_start), int(y_start))
        pt2 = (int(x_end), int(y_end))
        # 修改颜色格式为OpenCV需要的BGR元组
        my_color = tuple(map(int, lane_color.tolist()))  # 转换RGB到BGR
        cv2.line(original_image, pt1, pt2, my_color, 2)

    print("straight_lanes number:")
    print(len(straight_lanes))
    # TODO: 计算消失点


    postprocess_result["straight_lanes"] = straight_lanes
    postprocess_result["straight_lane_fit_params"] = straight_lane_fit_params
    postprocess_result["straight_lane_colors"] = straight_lane_colors
    return postprocess_result


def vanishing_point_calculation(vp_coordinates, lane_lines):
    # calculate the vanishing point
    vp = None
    return vp

def camera_pose_calculation(vp, config):
    # calculate the camera pitch angle and yaw angle
    pitch_angle = None
    yaw_angle = None
    return pitch_angle, yaw_angle


def camera_extrinsic_matrix_calculation(pitch_angle, yaw_angle, config):
    # calculate the camera extrinsic matrix
    extrinsic_matrix = None
    return extrinsic_matrix


def verify_vanishing_point(vp, config):
    # verify the vanishing point by plotting the camera trajectory
    pass



def show_result(original_image, result, result_file_path):
    empty_image = np.zeros((RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 3), dtype=np.uint8)

    mask_image = result["mask_image"]
    lane_embedding_feats_image = result["lane_embedding_feats_image"]
    connect_image = result["connect_components_analysis_ret"]
    morphological_image = result["morphological_ret"]
    instance_seg_image2 = result["instance_seg_result"]
    binary_seg_image2 = result["binary_seg_result"]

    if mask_image is None:
        LOG.warning("Failed to get mask image")
        mask_image = empty_image
    if lane_embedding_feats_image is None:
        LOG.warning("Failed to get lane embedding feats image")
        lane_embedding_feats_image = empty_image
    if connect_image is None:
        LOG.warning("Failed to get connect image")
        connect_image = empty_image
    if morphological_image is None:
        LOG.warning("Failed to get morphological image")
        morphological_image = empty_image
    if instance_seg_image2 is None:
        LOG.warning("Failed to get instance seg image")
        instance_seg_image2 = empty_image
    if binary_seg_image2 is None:
        LOG.warning("Failed to get binary seg image")
        binary_seg_image2 = empty_image

    # Save result
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))

    # Plot the mask image in the top-left subplot
    axs[0, 0].imshow(original_image[:, :, (2, 1, 0)])
    axs[0, 0].set_title("src_image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(empty_image[:, :, (2, 1, 0)])
    axs[0, 1].set_title("empty_image")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(mask_image[:, :, (2, 1, 0)])
    axs[0, 2].set_title("mask_image")
    axs[0, 2].axis("off")

    axs[0, 3].imshow(lane_embedding_feats_image[:, :, (2, 1, 0)])
    axs[0, 3].set_title("lane_embedding_feats_image")
    axs[0, 3].axis("off")

    axs[1, 0].imshow(binary_seg_image2 * 255, cmap="gray")
    axs[1, 0].set_title("binary_seg_image")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(morphological_image * 255, cmap="gray")
    axs[1, 1].set_title("morphological_image")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(connect_image * 255, cmap="gray")
    axs[1, 2].set_title("connect_image")
    axs[1, 2].axis("off")

    axs[1, 3].imshow(instance_seg_image2[:, :, (2, 1, 0)])
    axs[1, 3].set_title("instance_seg_image")
    axs[1, 3].axis("off")

    # Adjust layout to prevent overlapping titles or labels
    plt.tight_layout()

    # Save the composite figure to the specified file
    plt.savefig(result_file_path, dpi=800)
    plt.close()

    LOG.info("Result saved to: {}".format(result_file_path))        
    pass


 


def main():
    args = init_args()
    calibration_config = load_calibration_config(args.config_path)
    print(calibration_config)

    original_image, binary_seg_image, instance_seg_image = lane_detection_step1(
        args.weights_path, args.image_path, 
        calibration_config["front-camera-intrinsic"]["param"]["cam_K"], 
        calibration_config["front-camera-intrinsic"]["param"]["cam_dist"])
    if binary_seg_image is None or instance_seg_image is None:
        LOG.error('Error in lane detection step 1')
        return
    
    postprocess_result = lane_detection_step2(original_image, binary_seg_image, instance_seg_image)
    if postprocess_result is None:
        LOG.error('Error in lane detection step 2')
        return

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    result_file_path = args.output_dir + 'output.jpg'
    show_result(original_image, postprocess_result, result_file_path)     
    pass

if __name__ == "__main__":
    # command line example:
    # python lane_vp_calibration.py --image_path data/route28/00000.jpg --weights_path model/lane_detection_model.pth --config_path data/lane_vp_calib_config.json --output_dir output/
    main()