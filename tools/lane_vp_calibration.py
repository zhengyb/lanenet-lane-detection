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
# 坐标系定义：（全部为右手坐标系）
# 像素坐标系Cp - O(0,0)在图像左上角; U:水平向右 V:竖直向下
# 图像坐标系Cf - O(0,0)在光轴上，; X:水平向右 Y:竖直向下
# 理想相机坐标系Cd - O(0,0,0)在光心; X:水平向右 Y:垂直向下 Z:水平向前
# 相机坐标系Cc - O(0,0,0)在光心; 理想坐标系经过旋转得到
# 世界坐标系Cw(自车坐标系Cego) - O(0,0,0)车头前沿横向中心地面点；X:车辆前进方向 Y:车辆左侧 Z:垂直地面向上
# 旋转角度：
# - 俯仰角pitch - 相机坐标系Cc相对于理想相机坐标系Cd的俯仰角，绕Xd轴旋转，正值表示向上倾斜
# - 偏航角yaw - 相机坐标系Cc相对于理想相机坐标系Cd的偏航角，绕Yd轴旋转
# - 横滚角roll - 相机坐标系Cc相对于理想相机坐标系Cd的横滚角，绕Zd轴旋转

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
LOG = init_logger.get_logger(log_file_name_prefix="lane_vp_calibration")

RESIZE_IMAGE_HEIGHT = CFG.AUG.TRAIN_CROP_SIZE[1]
RESIZE_IMAGE_WIDTH = CFG.AUG.TRAIN_CROP_SIZE[0]

# filter the lanes those are not straight enough
STRAIGHT_FIT_PARAM_THRESHOLD = [0.01, 0.1]


def init_args():
    parser = argparse.ArgumentParser(description="Lane Vanishing Point Calibration")
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/route28/00000.jpg",
        help="Path to the image",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="weights/tusimple_lanenet_zyb0213/best_model_miou0.6358.ckpt-270",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="data/lane_vp_calib_config.json",
        help="Path to the config file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/", help="Path to the output file"
    )
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


def image_preprocessing(image_path, cam_K_config, cam_dist_config):
    # 从配置中提取并转换参数格式
    cam_K = np.array(cam_K_config["data"], dtype=np.float32).reshape(3,3)
    cam_dist = np.array(cam_dist_config["data"], dtype=np.float32)  # 直接转换为一维数组
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    undistorted_image = cv2.undistort(image, cam_K, cam_dist)
    original_image = undistorted_image.copy()
    
    resized_image = cv2.resize(
        undistorted_image,
        (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    normalized_image = resized_image / 127.5 - 1.0

    # debug: set the bottom 1/3 of the image to 0
    #resized_height = resized_image.shape[0]
    #resized_image[int(resized_height * 0.75) :, :] = 0

    cv2.imwrite("resized_image.jpg", resized_image)

    return normalized_image, original_image


def lane_detection_step1(weights_path, image_path, cam_K, cam_dist):
    # Return the detected binary segmentation image and the instance segmentation image

    # Initialize tensorflow session and model once
    tf.reset_default_graph()

    LOG.info("Start reading image and preprocessing")
    normalized_image, original_image = image_preprocessing(image_path, cam_K, cam_dist)
    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[1, RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 3],
        name="input_tensor",
    )

    net = lanenet.LaneNet(phase="test", cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(
        input_tensor=input_tensor, name="LaneNet"
    )

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = (
        CFG.GPU.GPU_MEMORY_FRACTION
    )
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = "BFC"

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope="moving_avg"):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY
        )
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    try:
        with sess.as_default():
            saver.restore(sess=sess, save_path=weights_path)

            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [normalized_image]},
            )
            t_cost = time.time() - t_start
            LOG.info("Finished lane inference, cost time: {:.5f}s".format(t_cost))
            print("binary_seg_image.shape:")
            print(binary_seg_image.shape)
            print("instance_seg_image.shape:")
            print(instance_seg_image.shape)
    except Exception as e:
        LOG.error("Error in lane detection step 1: {}".format(e))
        return None, None
    return original_image, binary_seg_image, instance_seg_image


def lane_detection_step2(original_image, binary_seg_image, instance_seg_image):
    # find the suitable lanes for vanishing point calculation on the 2D image

    h_orig, w_orig = original_image.shape[:2]

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    postprocess_result = postprocessor.postprocess(
        binary_seg_result=binary_seg_image[0],
        instance_seg_result=instance_seg_image[0],
        source_image=original_image,  # 源图像
        with_lane_fit=False,
        with_2d_lane_fit=True,
    )
    if (
        postprocess_result["fit_params"] is None
        or len(postprocess_result["fit_params"]) == 0
    ):
        LOG.error("No any lane detected")
        return postprocess_result

    print("postprocess_result['fit_params']:")
    print(postprocess_result["fit_params"])

    straight_lanes = []
    straight_lane_fit_params = []
    straight_lane_colors = []
    for lane_index in range(len(postprocess_result["fit_params"])):
        fit_param = postprocess_result["fit_params"][lane_index]
        resized_lane_coords = postprocess_result["resized_lane_coords"][lane_index]
        lane_color = COLOR_MAP[lane_index]

        # Convert tuple coordinates to numpy array
        # Original format: (y_coords, x_coords)
        lane_points = np.vstack(resized_lane_coords).T  # Convert to Nx2 array [x, y]

        # filter the lanes those are not straight enough
        if abs(fit_param[0]) > STRAIGHT_FIT_PARAM_THRESHOLD[0]:
            continue
        # 车的行进方向应该与车道线平行，所以车道线的斜率应该保持在一定的区间范围内
        # 斜率范围
        # if abs(fit_param[1]) > STRAIGHT_FIT_PARAM_THRESHOLD[1]:
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
    vp_list = vanishing_point_calculation(straight_lane_fit_params)
    print("vp_list:")
    print(vp_list)
    # 在原图上绘制消失点
    # for vp in vp_list:
    #    cv2.circle(original_image, (int(vp[2][0]), int(vp[2][1])), 20, (0, 0, 255), -1)

    filtered_vp_mean, filtered_vp_list = filter_vp_list(vp_list)
    print("filtered_vp_mean:")
    print(filtered_vp_mean)
    print("filtered_vp_list:")
    print(filtered_vp_list)
    if filtered_vp_mean is not None:
        # 在原图上绘制过滤后的消失点
        cv2.circle(
            original_image,
            (int(filtered_vp_mean[0]), int(filtered_vp_mean[1])),
            20,
            (0, 0, 255),
            -1,
        )
        if filtered_vp_list is not None:
            for vp in filtered_vp_list:
                cv2.circle(
                    original_image, (int(vp[2][0]), int(vp[2][1])), 10, (0, 0, 255), -1
                )
                cv2.imwrite("original_image_with_vp.jpg", original_image)

    postprocess_result["straight_lanes"] = straight_lanes
    postprocess_result["straight_lane_fit_params"] = straight_lane_fit_params
    postprocess_result["straight_lane_colors"] = straight_lane_colors
    postprocess_result["filtered_vp_mean"] = filtered_vp_mean
    postprocess_result["filtered_vp_list"] = (
        filtered_vp_list  # [[straight_lane_index1, straight_lane_index2, [vp_point_x, vp_point_y]]]
    )

    return postprocess_result


def filter_vp_list(vp_list, distance_threshold=20):
    # 1. 计算所有消失点的平均值
    # 2. 计算所有消失点到平均值的距离
    # 3. 过滤掉距离平均值较远的消失点
    # 4. 重新计算平均值
    # 5. 返回过滤后的消失点

    vp_points = [vp[2] for vp in vp_list]
    # 计算所有消失点的平均值
    vp_mean = np.mean(vp_points, axis=0)
    # 计算所有消失点到平均值的距离
    vp_distance = np.linalg.norm(vp_points - vp_mean, axis=1)
    # 过滤掉距离平均值较远的消失点
    filtered_vp_list = [
        vp_list[i] for i in range(len(vp_list)) if vp_distance[i] < distance_threshold
    ]
    if len(filtered_vp_list) == 0:
        return None, None
    # 重新计算平均值
    filtered_vp_points = [vp[2] for vp in filtered_vp_list]
    filtered_vp_mean = np.mean(filtered_vp_points, axis=0)
    # 计算所有消失点到平均值的距离
    # filtered_vp_distance = np.linalg.norm(filtered_vp_list - filtered_vp_mean, axis=1)

    return filtered_vp_mean, filtered_vp_list


def vanishing_point_calculation(lane_line_fit_params):
    # calculate the vanishing point of each two lanes
    vp = []
    for i in range(len(lane_line_fit_params)):
        for j in range(i + 1, len(lane_line_fit_params)):
            vp.append(
                [
                    i,
                    j,
                    vanishing_point_calculation_two_lanes(
                        lane_line_fit_params[i], lane_line_fit_params[j]
                    ),
                ]
            )
    return vp


def vanishing_point_calculation_two_lanes(lane_line_fit_param1, lane_line_fit_param2):
    # calculate the vanishing point of two lanes

    # 计算两条直线的交点
    # 直线1: x = m1 * y + b1
    # 直线2: x = m2 * y + b2
    # 交点: y = (b2 - b1) / (m1 - m2)
    # 交点: x = m1 * y + b1

    y = (lane_line_fit_param2[1] - lane_line_fit_param1[1]) / (
        lane_line_fit_param1[0] - lane_line_fit_param2[0]
    )
    x = lane_line_fit_param1[0] * y + lane_line_fit_param1[1]

    return x, y


def camera_pose_calculation(vp, cam_K, image_width=1920, image_height=1080):
    """
    通过消失点计算相机姿态角（俯仰角和偏航角）
    :param vp: 消失点坐标（x,y）像素坐标系，需为去畸变后的坐标
    :param cam_K: 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    :param image_width: 图像宽度（用于合理性检查）
    :param image_height: 图像高度（用于合理性检查）
    :return: (pitch_deg, yaw_deg) 俯仰角和偏航角（单位：度）
    """
    # 输入参数校验
    if vp is None:
        LOG.error("Vanishing point is None")
        return None, None
    
    if cam_K.shape != (3, 3):
        LOG.error(f"Invalid camera matrix shape: {cam_K.shape}")
        return None, None
    
    try:
        # 参数安全提取
        fx = float(cam_K[0, 0])
        fy = float(cam_K[1, 1])
        cx = float(cam_K[0, 2])
        cy = float(cam_K[1, 2])
        
        # 消失点坐标验证
        vp_x, vp_y = float(vp[0]), float(vp[1])
        if not (0 <= vp_x <= image_width and 0 <= vp_y <= image_height):
            LOG.warning(f"Vanishing point ({vp_x:.1f}, {vp_y:.1f}) out of image bounds")
        
        # 坐标系转换说明：
        # 图像坐标系：原点在左上角，X向右，Y向下
        # 3D相机坐标系：X向右，Y向下，Z向前
        # 因此需要将Y轴偏移取反以符合常规3D坐标系方向
        pixel_offset_x = vp_x - cx
        pixel_offset_y = cy - vp_y  # 注意此处取反
        
        # 安全除法处理（防止焦距为0）
        if fy <= 0 or fx <= 0:
            raise ValueError(f"Invalid focal length fx={fx}, fy={fy}")
        
        # 计算视角角度（使用arctan2保证象限正确）
        # 俯仰角：正值表示相机向下倾斜 (符合汽车前视相机安装特性)
        # 偏航角：正值表示相机向右偏转
        pitch_rad = np.arctan2(pixel_offset_y, fy)
        yaw_rad = np.arctan2(pixel_offset_x, fx)
        
        # 角度范围约束（根据物理限制）
        MAX_PITCH = np.radians(85)  # ±85度限制
        MAX_YAW = np.radians(85)
        pitch_rad = np.clip(pitch_rad, -MAX_PITCH, MAX_PITCH)
        yaw_rad = np.clip(yaw_rad, -MAX_YAW, MAX_YAW)
        
        # 转换为角度值
        pitch_deg = np.degrees(pitch_rad)
        yaw_deg = np.degrees(yaw_rad)
        
        LOG.info(f"Calculated angles: pitch={pitch_deg:.2f}°, yaw={yaw_deg:.2f}° "
                f"| VP=({vp_x:.1f}, {vp_y:.1f})")
        return pitch_deg, yaw_deg
    
    except Exception as e:
        LOG.error(f"Pose calculation failed: {str(e)} | "
                 f"Params: fx={fx}, fy={fy}, cx={cx}, cy={cy}, "
                 f"vp=({vp_x}, {vp_y})")
        return None, None


def camera_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """
    TBD
    计算包含从理想相机坐标系到实际相机坐标系的旋转矩阵
    :param pitch_deg: 俯仰角（度）正值表示向上倾斜
    :param yaw_deg: 偏航角（度）正值表示向右偏转
    :param roll_deg: 横滚角（度）正值表示向右倾斜
    :return: 4x4外参矩阵
    """
    # 角度转弧度
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    roll = np.radians(roll_deg)

    # 修正后的旋转矩阵（按Roll-Pitch-Yaw顺序）
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    R = R_yaw @ R_pitch @ R_roll
    return R
    
    
def extrinsic_matrix_calculation(pitch_deg, yaw_deg, roll_deg=0.0, height_offset=1.0, dy=0.0, dx=0.0):
    """
    从自车坐标系到实际相机坐标系的映射矩阵
    pitch_deg: 俯仰角（度）正值表示向上倾斜
    yaw_deg: 偏航角（度）正值表示向右偏转
    roll_deg: 横滚角（度）正值表示向右倾斜
    height_offset: 相机高度相对于自车坐标系原点的偏移量, 默认1.0m
    dy: 相机相对于自车坐标系原点的偏移量, 向左偏移为正  TBD?
    dx: 相机相对于自车坐标系原点的偏移量, 向前偏移为正  TBD?

    return 4x4外参矩阵
    """

    # 从世界坐标系到理想相机坐标系的旋转矩阵
    R_world_to_ideal = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ])
    R_world_to_ideal2 = np.concatenate([R_world_to_ideal, np.zeros((1, 3))], axis=0)

    # 从世界坐标系到理想相机坐标系的平移向量
    T_world_to_ideal = np.array([
        [dy, -height_offset, dx, 1] # TBD?
    ]).reshape(4, 1)

    # 构建从世界坐标系到理想相机坐标系的4x4外参矩阵
    M0 = np.concatenate([R_world_to_ideal2, T_world_to_ideal], axis=1)

    # 构建从理想相机坐标系到实际相机坐标系的旋转矩阵
    R_ideal_to_actual = camera_rotation_matrix(pitch_deg, yaw_deg, roll_deg)

    # 构建从理想相机坐标系到实际相机坐标系的4x4外参矩阵
    R_ideal_to_actual2 = np.concatenate([R_ideal_to_actual, np.zeros((1, 3))], axis=0)

    # 构建从理想相机坐标系到实际相机坐标系的平移向量
    T_ideal_to_actual = np.array([[0, 0, 0, 1]]).reshape(4, 1)

    M1 = np.concatenate([R_ideal_to_actual2, T_ideal_to_actual], axis=1)

    # 构建从世界坐标系到实际相机坐标系的4x4外参矩阵
    M = M1 @ M0
    

    return M
    

def convert_image_to_ipm(image, extrinsic_matrix, cam_K):
    """
    将图像转换为IPM图像
    :param image: 输入图像
    :param extrinsic_matrix: 外参矩阵 (需要是3x3单应性矩阵)
    :param cam_K: 相机内参矩阵
    :return: IPM图像
    """
    # 将4x4外参矩阵转换为3x3单应性矩阵
    # 取外参矩阵的旋转部分与内参组合
    R = extrinsic_matrix[:3, :3]
    H = cam_K @ R @ np.linalg.inv(cam_K)
    
    # 确保矩阵数据类型符合OpenCV要求
    H = H.astype(np.float32)
    
    return cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))


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
        args.weights_path,
        args.image_path,
        calibration_config["front-camera-intrinsic"]["param"]["cam_K"],
        calibration_config["front-camera-intrinsic"]["param"]["cam_dist"],
    )
    if binary_seg_image is None or instance_seg_image is None:
        LOG.error("Error in lane detection step 1")
        return

    postprocess_result = lane_detection_step2(
        original_image, binary_seg_image, instance_seg_image
    )
    if postprocess_result is None:
        LOG.error("Error in lane detection step 2")
        return

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    result_file_path = args.output_dir + "output.jpg"
    show_result(original_image, postprocess_result, result_file_path)

    if postprocess_result["filtered_vp_mean"] is not None:
        vp_x, vp_y = postprocess_result["filtered_vp_mean"]
        cam_K = np.array(
                calibration_config["front-camera-intrinsic"]["param"]["cam_K"]["data"]
            ).reshape(3, 3)
        pitch_deg, yaw_deg = camera_pose_calculation(
            (vp_x, vp_y),
            cam_K
        )
        print(f"Camera Orientation: Pitch={pitch_deg:.2f}°, Yaw={yaw_deg:.2f}°")

        M = extrinsic_matrix_calculation(pitch_deg, yaw_deg, height_offset=1.0, dy=-0.3, dx=-2.0)
        print(M)
        ipm_image = convert_image_to_ipm(original_image, M, cam_K)
        cv2.imwrite(args.output_dir + "ipm_image.jpg", ipm_image)

    # TODO: calculate the Pitch and Yaw using multiple images


if __name__ == "__main__":
    # command line example:
    # python lane_vp_calibration.py --image_path data/route28/vp_calib4.jpg --weights_path model/lane_detection_model.pth --config_path data/lane_vp_calib_config.json --output_dir output/
    main()
