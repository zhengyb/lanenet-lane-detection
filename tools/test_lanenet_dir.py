#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
from pathlib import Path
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix="lanenet_test_dir")


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def init_args():
    """
    # Usage:
    # python3 test_lanenet_dir.py --input_dir /app/data/route50/route50-0214/ --output_dir /app/data/route50/route50-0214-retsult/ --weights_path /app/weights/tusimple_lanenet_zyb0213/best_model_miou0.6358.ckpt-270  --with_lane_fit False --data_source lab_lenovo
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, help="The input directory containing images"
    )
    parser.add_argument(
        "--output_dir", type=str, help="The output directory for saving results"
    )
    parser.add_argument("--weights_path", type=str, help="The model weights path")
    parser.add_argument(
        "--with_lane_fit",
        type=bool,
        help="If you want to get lane fit result",
        default=False,
    )
    parser.add_argument(
        "--data_source", type=str, help="The data source to test", default="tusimple"
    )
    parser.add_argument(
        "--limit_images",
        type=int,
        help="Limit the maximum number of images to process. Default is 0 that means no limit",
        default=0,
    )

    return parser.parse_args()


def test_lanenet_directory(
    input_dir,
    output_dir,
    weights_path,
    with_lane_fit=True,
    data_source="tusimple",
    limit_images=0,
):
    """
    Test LaneNet on a directory of images using the existing test_lanenet function
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of image files and sort them
    input_dir = Path(input_dir)
    image_files = sorted(list(input_dir.glob("*.[jJ][pP][gG]")), key=lambda x: x.name)

    LOG.info("Found {} images in {}".format(len(image_files), input_dir))

    # Initialize tensorflow session and model once
    tf.reset_default_graph()

    # Initialize LaneNet
    input_tensor = tf.placeholder(
        dtype=tf.float32, shape=[1, 256, 512, 3], name="input_tensor"
    )
    net = lanenet.LaneNet(phase="test", cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(
        input_tensor=input_tensor, name="LaneNet"
    )

    if data_source == "tusimple":
        ipm_remap_file_path = "./data/tusimple_ipm_remap.yml"
    elif data_source == "lab_lenovo":
        ipm_remap_file_path = "./data/lab_lenovo_ipm_remap_1920_1080.yml"
    else:
        raise ValueError(f"Unsupported data source: {data_source}")

    # Initialize postprocessor
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(
        cfg=CFG, ipm_remap_file_path=ipm_remap_file_path
    )

    # Set session configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = (
        CFG.GPU.GPU_MEMORY_FRACTION
    )
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = "BFC"

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
    # saver = tf.train.Saver()

    time_start = time.time()
    try:
        with sess.as_default():
            saver.restore(sess=sess, save_path=weights_path)

            # Process each image
            for idx, image_file in enumerate(image_files, 1):
                if limit_images > 0 and idx > limit_images:
                    break
                LOG.info(
                    "[{}/{}] Processing image: {}".format(
                        idx, len(image_files), image_file.name
                    )
                )

                try:
                    # Read and preprocess image
                    image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
                    if image is None:
                        LOG.error("Failed to read image: {}".format(image_file))
                        continue

                    original_image = image.copy()
                    image = cv2.resize(
                        image, (512, 256), interpolation=cv2.INTER_LINEAR
                    )
                    image = image / 127.5 - 1.0

                    # Inference
                    binary_seg_image, instance_seg_image = sess.run(
                        [binary_seg_ret, instance_seg_ret],
                        feed_dict={input_tensor: [image]},
                    )

                    # Postprocess
                    postprocess_result = postprocessor.postprocess(
                        binary_seg_result=binary_seg_image[0],
                        instance_seg_result=instance_seg_image[0],
                        source_image=original_image,
                        with_lane_fit=with_lane_fit,
                        data_source=data_source,
                    )

                    result_file_path = output_dir / (image_file.stem + "_result.jpg")
                    postprocessor.save_postprocess_result(
                        original_image, postprocess_result, result_file_path
                    )

                except Exception as e:
                    LOG.error(
                        "Failed to process {}: {}".format(image_file.name, str(e))
                    )
                    continue
    except Exception as e:
        LOG.error("Failed to process {}: {}".format(image_file.name, str(e)))
    finally:
        plt.close("all")
        tf.reset_default_graph()


    time_end = time.time()

    LOG.info("Processing complete")
    if limit_images == 0:
        LOG.info("Time taken: {} seconds for {} images, {} seconds per_image".format((time_end - time_start), len(image_files), (time_end - time_start) / len(image_files)))
    else:
        LOG.info("Time taken: {} seconds for {} images, {} seconds per_image".format((time_end - time_start), limit_images, (time_end - time_start) / limit_images))


def main():
    """
    main function
    """
    # Initialize arguments
    args = init_args()

    # Test lanenet on the input directory
    test_lanenet_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weights_path=args.weights_path,
        with_lane_fit=args.with_lane_fit,
        data_source=args.data_source,
        limit_images=args.limit_images,
    )


if __name__ == "__main__":
    main()
