import numpy as np
from enum import Enum

from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


class CameraName(Enum):
    INVALID = "invalid_camera"
    TUSIMPLE = "tusimple"
    ACCORD_LENOVO = "accord_lenovo"
    CARLA = "carla_camera"
    BEV = "bev_camera"

    def __str__(self):
        return self.value


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    if min_val == max_val:
        output_arr = np.zeros_like(input_arr)
    else:
        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def make_instance_seg_img_visuable(instance_seg_image):

    instance_seg_result = np.copy(instance_seg_image)
    for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
        instance_seg_result[:, :, i] = minmax_scale(instance_seg_result[:, :, i])
    embedding_image = np.array(instance_seg_result, np.uint8)
    return embedding_image


