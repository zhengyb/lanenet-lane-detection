#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午7:31
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : generate_tusimple_dataset.py
# @IDE: PyCharm Community Edition
"""
generate tusimple training dataset
"""
import argparse
import glob
import json
import os
import os.path as ops
import shutil

import cv2
import numpy as np
import random


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='The origin path of unzipped tusimple dataset')

    return parser.parse_args()


def process_json_file(json_file_path, src_dir, ori_dst_dir, binary_dst_dir, instance_dst_dir):
    """

    :param json_file_path:
    :param src_dir: origin clip file path
    :param ori_dst_dir:
    :param binary_dst_dir:
    :param instance_dst_dir:
    :return:
    """
    assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

    image_nums = len(os.listdir(ori_dst_dir))

    with open(json_file_path, 'r') as file:
        for line_index, line in enumerate(file):
            info_dict = json.loads(line)

            image_dir = ops.split(info_dict['raw_file'])[0]
            image_dir_split = image_dir.split('/')[1:]
            image_dir_split.append(ops.split(info_dict['raw_file'])[1])
            image_name = '_'.join(image_dir_split)
            image_path = ops.join(src_dir, info_dict['raw_file'])
            assert ops.exists(image_path), '{:s} not exist'.format(image_path)

            h_samples = info_dict['h_samples']
            lanes = info_dict['lanes']

            image_name_new = '{:s}.png'.format('{:d}'.format(line_index + image_nums).zfill(4))

            src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            dst_binary_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
            dst_instance_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)

            for lane_index, lane in enumerate(lanes):
                assert len(h_samples) == len(lane)
                lane_x = []
                lane_y = []
                for index in range(len(lane)):
                    if lane[index] == -2:
                        continue
                    else:
                        ptx = lane[index]
                        pty = h_samples[index]
                        lane_x.append(ptx)
                        lane_y.append(pty)
                if not lane_x:
                    continue
                lane_pts = np.vstack((lane_x, lane_y)).transpose()
                lane_pts = np.array([lane_pts], np.int64)

                cv2.polylines(dst_binary_image, lane_pts, isClosed=False,
                              color=255, thickness=5)
                cv2.polylines(dst_instance_image, lane_pts, isClosed=False,
                              color=lane_index * 50 + 20, thickness=5)

            dst_binary_image_path = ops.join(binary_dst_dir, image_name_new)
            dst_instance_image_path = ops.join(instance_dst_dir, image_name_new)
            dst_rgb_image_path = ops.join(ori_dst_dir, image_name_new)

            cv2.imwrite(dst_binary_image_path, dst_binary_image)
            cv2.imwrite(dst_instance_image_path, dst_instance_image)
            cv2.imwrite(dst_rgb_image_path, src_image)

            print('Process {:s} success'.format(image_name))


def gen_train_sample(src_dir, b_gt_image_dir, i_gt_image_dir, image_dir):
    """
    generate sample index file
    :param src_dir:
    :param b_gt_image_dir:
    :param i_gt_image_dir:
    :param image_dir:
    :return:
    """

    with open('{:s}/training/train.txt'.format(src_dir), 'w') as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.png'):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                print('图像对: {:s}损坏'.format(image_name))
                continue
            else:
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path)
                file.write(info + '\n')
    return


def process_tusimple_dataset(src_dir):
    """

    :param src_dir:
    :return:
    """
    traing_folder_path = ops.join(src_dir, 'training')
    testing_folder_path = ops.join(src_dir, 'testing')

    os.makedirs(traing_folder_path, exist_ok=True)
    os.makedirs(testing_folder_path, exist_ok=True)


    print("Copying json label files...")
    for json_label_path in glob.glob('{:s}train_set/label*.json'.format(src_dir)):
        print("json label file: ", json_label_path)
        json_label_name = ops.split(json_label_path)[1]

        print("Copying json label file: ", json_label_path)
        shutil.copyfile(json_label_path, ops.join(traing_folder_path, json_label_name))

    print("Copying testing json label files...")
    for json_label_path in glob.glob('{:s}test_set/test*.json'.format(src_dir)):
        json_label_name = ops.split(json_label_path)[1]

        print("Copying json label file: ", json_label_path)
        shutil.copyfile(json_label_path, ops.join(testing_folder_path, json_label_name))

    gt_image_dir = ops.join(traing_folder_path, 'gt_image')
    gt_binary_dir = ops.join(traing_folder_path, 'gt_binary_image')
    gt_instance_dir = ops.join(traing_folder_path, 'gt_instance_image')

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)

    print("Processing json label files for traing_folder_path...")
    for json_label_path in glob.glob('{:s}/*.json'.format(traing_folder_path)):
       process_json_file(json_label_path, src_dir + '/train_set', gt_image_dir, gt_binary_dir, gt_instance_dir)

    print("Generating train sample...")
    gen_train_sample(src_dir, gt_binary_dir, gt_instance_dir, gt_image_dir)
    print("Done")

    return


def split_train_val_dataset(src_dir, val_ratio=0.1):
    train_txt = ops.join(src_dir, 'training', 'train.txt')
    val_txt = ops.join(src_dir, 'training', 'val.txt')

    original_train_txt = ops.join(src_dir, 'training', 'original_train.txt')

    # verify if the original_train_txt is valid
    if not ops.exists(original_train_txt):
        print("{original_train_txt} not found")
        return
    
    with open(original_train_txt, 'r') as file:
        lines = file.readlines()

    # shuffle the lines
    random.shuffle(lines)

    # split the lines into train and val
    train_lines = lines[:int(len(lines) * (1 - val_ratio))]
    val_lines = lines[int(len(lines) * (1 - val_ratio)):]

    with open(train_txt, 'w') as file:
        file.writelines(train_lines)

    with open(val_txt, 'w') as file:
        file.writelines(val_lines)

    print("Split train and val dataset done")
    # output the number of lines in train and val
    print("Number of lines in original_train.txt: ", len(lines))
    print("Number of lines in train.txt: ", len(train_lines))
    print("Number of lines in val.txt: ", len(val_lines))

if __name__ == '__main__':
    # Command line in the container: python ./tools/generate_tusimple_dataset.py --src_dir /app/data/TUSimple/
    args = init_args()

    original_train_txt = ops.join(args.src_dir, 'training', 'original_train.txt')
    train_txt = ops.join(args.src_dir, 'training', 'train.txt')
    val_txt = ops.join(args.src_dir, 'training', 'val.txt')

    # process the tusimple dataset
    if False:
        os.remove(original_train_txt)
        os.remove(train_txt)
        os.remove(val_txt)
        process_tusimple_dataset(args.src_dir)

    # rename the train.txt to original_train.txt
    if ops.exists(train_txt) and not ops.exists(original_train_txt):
        os.rename(train_txt, original_train_txt)

        split_train_val_dataset(args.src_dir, val_ratio=0.1)


