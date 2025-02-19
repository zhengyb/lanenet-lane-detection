#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger
from tools.utils import make_instance_seg_img_visuable

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

output_dir="/app/test/"

# Command line in the container:
# python tools/test_lanenet.py --image_path /app/data/tusimple_test_image/0.jpg --weights_path /app/weights/tusimple_lanenet/tusimple_lanenet.ckpt --with_lane_fit True
# python tools/test_lanenet.py --image_path /app/data/lab_test/lab_lanes0213/1.jpg --weights_path /app/weights/tusimple_lanenet/tusimple_lanenet.ckpt --with_lane_fit True --data_source lab_lenovo


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--with_lane_fit', type=args_str2bool, help='If need to do lane fit', default=False)
    parser.add_argument('--data_source', type=str, help='The data source: tusimple or lab_lenovo', default='tusimple')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path, with_lane_fit=True, data_source='tusimple',
                 save_dir=None, save_name=None):
    """

    :param image_path:
    :param weights_path:
    :param with_lane_fit:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    # Initialize tensorflow session and model once
    tf.reset_default_graph()
    
    LOG.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # print("image.shape:")
    # print(image.shape)

    # if data_source == 'lab_lenovo':
    # change the top-half of the image to black
    # image[:int(image.shape[0]*0.75), :, :] = 0

    image_vis = image
    # 将图像缩放为512x256. Be careful, the resized sizes are hard-coded in many places in the project.
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)

    print("image.shape:")
    print(image.shape)
    # if data_source == 'lab_lenovo':
    if False:
        # change the top-half of the image to black
        image[:int(256 *0.75), :, :] = 0
        pass

    cv2.imwrite(output_dir + '1-image_resize.jpg', image)

    # 标准化到[-1, 1]范围
    image = image / 127.5 - 1.0
    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    if data_source == 'tusimple':
        ipm_remap_file_path = './data/tusimple_ipm_remap.yml'
    elif data_source == 'lab_lenovo':
        ipm_remap_file_path = './data/lab_lenovo_ipm_remap_1920_1080.yml'
    else:
        raise ValueError(f"Unsupported data source: {data_source}")


    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG, ipm_remap_file_path=ipm_remap_file_path)

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

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        # 运行500次，计算平均时间
        loop_times = 1 #500
        for i in range(loop_times):
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
        t_cost = time.time() - t_start
        t_cost /= loop_times
        LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))


        # save the binary_seg_image and instance_seg_image as images
        cv2.imwrite(output_dir + '2-binary_seg_image.jpg', binary_seg_image[0] * 255)
        # (each_value + 1.0 ) * 127.5
        # deep copy the instance_seg_image
        seg_image = np.copy(instance_seg_image[0])
        print("seg_image.shape:")
        print(seg_image.shape)
        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            seg_image[:, :, i] = minmax_scale(seg_image[:, :, i])
        seg_image2 = np.array(seg_image, np.uint8)

        #seg_image = make_instance_seg_img_visuable(seg_image)
        cv2.imwrite(output_dir + '3-instance_seg_image.jpg', seg_image2)

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis, # 源图像
            with_lane_fit=with_lane_fit,
            #data_source='tusimple' # 数据集, Why?
            data_source=data_source
        )
        mask_image = postprocess_result['mask_image']

        if mask_image is None:
            LOG.warning('Failed to get mask image')
            mask_image = np.zeros((256, 512, 3), dtype=np.uint8)

        #source_image_with_lane = postprocess_result['source_image']
        if with_lane_fit:
            lane_params = postprocess_result['fit_params']
            LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
            for i in range(len(lane_params)):
                LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        result_file_path = output_dir + 'output.jpg'
        postprocessor.save_postprocess_result(image_vis, postprocess_result, result_file_path)

        # Save the composite figure to the specified file
        #plt.savefig(output_dir + 'output.jpg', dpi=300)

    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.image_path, args.weights_path, with_lane_fit=args.with_lane_fit, data_source=args.data_source)
