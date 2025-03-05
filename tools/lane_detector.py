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
from tools.utils import make_instance_seg_img_visuable, CameraName
from tools.camera_geometry import CameraGeometry


CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')
LANENET_WIDTH = 512
LANENET_HEIGHT = 256

# filter the lanes those are not straight enough
STRAIGHT_FIT_PARAM_THRESHOLD = [0.01, 2]

class LaneDetector():
    def __init__(self, cam_geom=CameraGeometry(CameraName.CARLA, field_of_view_deg=45), 
                 model_path='./weights/tusimple_lanenet_zyb0213/best_model_miou0.6358.ckpt-270',
                 debug=False):
        self.debug = debug
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        self.model_path = model_path
        self.width = LANENET_WIDTH
        self.height = LANENET_HEIGHT
        self.device = None # TODO: cuda or cpu
        self.input_tensor = None
        self.binary_seg_ret = None
        self.instance_seg_ret = None
        self.postprocessor = None
        self.sess = None
        self.saver = None
        self.model = self._init_model()

    def _init_model(self):
        # Initialize tensorflow session and model once
        tf.reset_default_graph()
        # Initialize LaneNet
        self.input_tensor = tf.placeholder(
            dtype=tf.float32, shape=[1, self.height, self.width, 3], name="input_tensor"
        )
        net = lanenet.LaneNet(phase="test", cfg=CFG)
        binary_seg_ret, instance_seg_ret = net.inference(
            input_tensor=self.input_tensor, name="LaneNet"
        )
        self.binary_seg_ret = binary_seg_ret
        self.instance_seg_ret = instance_seg_ret

        # Initialize postprocessor
        camera_name = self.cg.camera_name
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(
            cfg=CFG, ipm_remap_file_path=self._get_ipm_remap_file_path(camera_name)
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

        self.sess = tf.Session(config=sess_config)
        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope="moving_avg"):
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY
            )
            variables_to_restore = variable_averages.variables_to_restore()

        # define saver
        self.saver = tf.train.Saver(variables_to_restore)
        with self.sess.as_default():
            self.saver.restore(self.sess, self.model_path)
        pass

    def _get_ipm_remap_file_path(self, camera_name):
        # ipm remap file path: "./data/camera_<name>_ipm_remap.yml"
        return f'./data/camera_{camera_name}_ipm_remap.yml'
        
    def read_imagefile_to_array(self, filename, rotate_180=False):
        # preprocess image
        image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        if image is None:
            LOG.error("Failed to read image: {}".format(filename))
            return None
        if rotate_180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        # TODO: undistort image 
        original_image = image.copy()
        resize_image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        resize_image = resize_image / 127.5 - 1.0
        return resize_image, original_image   

    def detect_from_file(self, filename):        
        if self.cg.camera_name == "accord_camera":
            rotate_180 = True
        else:
            rotate_180 = False
        img_array, original_image = self.read_imagefile_to_array(filename, rotate_180)
        return self.detect(img_array, original_image)

    def _predict(self, img):
        # Inference
        input_tensor = self.input_tensor
        binary_seg_image, instance_seg_image = self.sess.run(
                        [self.binary_seg_ret, self.instance_seg_ret],
                        feed_dict={input_tensor: [img]},
                    )
        if binary_seg_image is None or len(binary_seg_image) == 0:
            return None, None
        return binary_seg_image[0], instance_seg_image[0]

    def _postprocess(self, binary_seg_image, instance_seg_image, original_image,
                     with_lane_fit=False,
                     data_source="tusimple",
                     with_2d_lane_fit=False
                     ):
        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image,
            instance_seg_result=instance_seg_image,
            source_image=original_image,
            with_lane_fit=with_lane_fit,
            data_source=data_source,
            with_2d_lane_fit=with_2d_lane_fit
        )
        return postprocess_result

    def detect(self, img_array, original_image):
        binary_seg_image, instance_seg_image = self._predict(img_array)
        postprocess_result = self._postprocess(binary_seg_image, instance_seg_image, original_image,
                                               with_lane_fit=True, 
                                               data_source="TODO",
                                               with_2d_lane_fit=False)
        # TODO:
        if self.debug:
            result_file_path = "./output/result.jpg"
            self.postprocessor.save_postprocess_result(
                original_image, postprocess_result, result_file_path
            )
        return postprocess_result
    
        # save the postprocess result to an image file



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



def filter_vp_list(vp_list, distance_threshold=20):
    # 1. 计算所有消失点的平均值
    # 2. 计算所有消失点到平均值的距离
    # 3. 过滤掉距离平均值较远的消失点
    # 4. 重新计算平均值
    # 5. 返回过滤后的消失点

    if len(vp_list) == 0:
        return None, []

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


class CalibLaneDetector(LaneDetector):
    def __init__(self, cam_geom=CameraGeometry(camera_name=CameraName.CARLA, field_of_view_deg=45), 
                 model_path='./weights/tusimple_lanenet_zyb0213/best_model_miou0.6358.ckpt-270',
                 debug=False):
        super().__init__(cam_geom, model_path, debug) 
        self.estimated_pitch_deg = 0
        self.estimated_yaw_deg = 0
        self.mean_residuals_thresh = 15
        self.update_cam_geometry()
        self.pitch_yaw_history = []
        self.calibration_success = False

    def detect4calibration(self, img_array, original_image):
        # Fit lane lines in 2D image
        binary_seg_image, instance_seg_image = self._predict(img_array)
        postprocess_result = self._postprocess(binary_seg_image, instance_seg_image, original_image,
                                               with_lane_fit=False, 
                                               data_source="TODO",
                                               with_2d_lane_fit=True)

        return postprocess_result
    
    def detect_vanishing_point(self, filename):
        color_map = lanenet_postprocess.COLOR_MAP
        # Detect vanishing point in 3D space
        if self.cg.camera_name == "accord_camera":
            rotate_180 = True
        else:
            rotate_180 = False
        img_array, original_image = self.read_imagefile_to_array(filename, rotate_180)
        postprocess_result = self.detect4calibration(img_array, original_image)

        if (
            postprocess_result["fit_params"] is None
            or len(postprocess_result["fit_params"]) == 0
        ):
            LOG.error("No any lane detected")
            return None, postprocess_result

        if self.debug:
            print("postprocess_result['fit_params']:")
            print(postprocess_result["fit_params"])

        # Filter out lanes that are not straight enough
        straight_lanes = []
        straight_lane_fit_params = []
        straight_lane_colors = []
        for lane_index in range(len(postprocess_result["fit_params"])):
            fit_param = postprocess_result["fit_params"][lane_index]
            resized_lane_coords = postprocess_result["resized_lane_coords"][lane_index]
            lane_color = color_map[lane_index]

            # Convert tuple coordinates to numpy array
            # Original format: (y_coords, x_coords)
            lane_points = np.vstack(resized_lane_coords).T  # Convert to Nx2 array [x, y]

            # filter the lanes those are not straight enough
            if abs(fit_param[0]) > STRAIGHT_FIT_PARAM_THRESHOLD[0]:
                continue
            # 车的行进方向应该与车道线平行，所以车道线的斜率应该保持在一定的区间范围内
            # 斜率范围
            if abs(fit_param[1]) > STRAIGHT_FIT_PARAM_THRESHOLD[1]:
                continue
            # TODO: 过滤车道线覆盖的区域大小
            # TODO: the other filter conditions

            straight_fit_param = np.polyfit(lane_points[:, 0], lane_points[:, 1], 1)
            straight_lanes.append(lane_points)
            straight_lane_fit_params.append(straight_fit_param)
            straight_lane_colors.append(lane_color)
            if self.debug:
                # draw the fit straight lane on the original image
                y_start = int(self.cg.image_height / 3)
                y_end = int(self.cg.image_height * 0.9) # original image height
                x_start = straight_fit_param[0] * y_start + straight_fit_param[1]
                x_end = straight_fit_param[0] * y_end + straight_fit_param[1]
                pt1 = (int(x_start), int(y_start))
                pt2 = (int(x_end), int(y_end))
                # 修改颜色格式为OpenCV需要的BGR元组
                my_color = tuple(map(int, lane_color.tolist()))  # 转换RGB到BGR
                cv2.line(original_image, pt1, pt2, my_color, 5)
                print("pt1: {}, pt2: {}".format(pt1, pt2))

        if self.debug:
            print("straight_lanes number:")
            print(len(straight_lanes))

        # Calculate vanishing point
        vp_list = vanishing_point_calculation(straight_lane_fit_params)
        if self.debug:
            print("vp_list:")
            print(vp_list)
            # 在原图上绘制消失点
            for vp in vp_list:
                cv2.circle(original_image, (int(vp[2][0]), int(vp[2][1])), 5, (255, 0, 0), -1)

        filtered_vp_mean, filtered_vp_list = filter_vp_list(vp_list)
        if self.debug:
            print("filtered_vp_mean:")
            print(filtered_vp_mean)
            print("filtered_vp_list:")
            print(filtered_vp_list)
        
        if filtered_vp_mean is not None and self.debug:
            # 在原图上绘制过滤后的消失点
            cv2.circle(original_image, (int(filtered_vp_mean[0]), int(filtered_vp_mean[1])), 10, (0, 0, 255), -1)
            cv2.imwrite("original_image_with_vp.jpg", original_image)

        postprocess_result["straight_lanes"] = straight_lanes
        postprocess_result["straight_lane_fit_params"] = straight_lane_fit_params
        postprocess_result["straight_lane_colors"] = straight_lane_colors
        postprocess_result["filtered_vp_mean"] = filtered_vp_mean
        postprocess_result["filtered_vp_list"] = (
            filtered_vp_list  # [[straight_lane_index1, straight_lane_index2, [vp_point_x, vp_point_y]]]
        )

        if self.debug:
            result_file_path = "./output/vanishing_point_result.jpg"
            self.postprocessor.save_postprocess_result(
                original_image, postprocess_result, result_file_path
            )

        return filtered_vp_mean, postprocess_result

    def add_to_pitch_yaw_history(self, pitch, yaw):
        self.pitch_yaw_history.append([pitch, yaw])
        if len(self.pitch_yaw_history) > 50:
            py = np.array(self.pitch_yaw_history)
            mean_pitch = np.mean(py[:,0])
            mean_yaw = np.mean(py[:,1])
            self.estimated_pitch_deg = np.rad2deg(mean_pitch)
            self.estimated_yaw_deg = np.rad2deg(mean_yaw)
            self.update_cam_geometry()
            self.calibration_success = True
            self.pitch_yaw_history = []
            print("yaw, pitch = ", self.estimated_yaw_deg, self.estimated_pitch_deg)

    def get_pitch_yaw_from_vp(self, u_i, v_i):
        # get pitch and yaw in radian from vanishing point in one image
        K_inv = self.cg.inverse_intrinsic_matrix
        p_infinity = np.array([u_i, v_i, 1])
        r3 = K_inv @ p_infinity    
        r3 /= np.linalg.norm(r3)
        yaw = -np.arctan2(r3[0], r3[2])
        pitch = np.arcsin(r3[1])    

        if self.debug:
            print("pitch degree: {}".format(np.rad2deg(pitch)))
            print("yaw degree: {}".format(np.rad2deg(yaw)))

        return pitch, yaw
    
    def update_cam_geometry(self):
        self.cg = CameraGeometry(
            camera_name = self.cg.camera_name,
            height = self.cg.height, 
            roll_deg = self.cg.roll_deg,
            image_width = self.cg.image_width,
            image_height = self.cg.image_height, 
            field_of_view_deg = self.cg.field_of_view_deg,
            pitch_deg = self.estimated_pitch_deg, 
            yaw_deg = self.estimated_yaw_deg )
        self.cut_v, self.grid = self.cg.precompute_grid()


if __name__ == "__main__":
    test_image_path = "./data/carla_vp_calib01.png"
    carla_cam_geom = CameraGeometry(
        camera_name = CameraName.CARLA,
        height = 1.3,
        roll_deg = 0,
        image_width = 1024,
        image_height = 512, 
        field_of_view_deg = 45
    )
    calib_lane_detector = CalibLaneDetector(carla_cam_geom, debug=True)
    filtered_vp_mean, postprocess_result = calib_lane_detector.detect_vanishing_point(test_image_path)

    # 计算pitch和yaw
    pitch, yaw = calib_lane_detector.get_pitch_yaw_from_vp(filtered_vp_mean[0], filtered_vp_mean[1])

    
    trafo_cam_to_road = calib_lane_detector.cg.trafo_cam_to_road
    print("Before calibration, trafo_cam_to_road:")
    print(trafo_cam_to_road)

    for i in range(51):
        calib_lane_detector.add_to_pitch_yaw_history(pitch, yaw)


    carlibed_cam_geom = calib_lane_detector.cg
    trafo_cam_to_road = carlibed_cam_geom.trafo_cam_to_road
    print("After calibration, trafo_cam_to_road:")
    print(trafo_cam_to_road)

    ln1_pt1_roadXYZ = carlibed_cam_geom.uv_to_roadXYZ_roadframe_iso8855(583, 170)
    ln1_pt2_roadXYZ = carlibed_cam_geom.uv_to_roadXYZ_roadframe_iso8855(981, 460)
    
    ln2_pt1_roadXYZ = carlibed_cam_geom.uv_to_roadXYZ_roadframe_iso8855(522, 170)
    ln2_pt2_roadXYZ = carlibed_cam_geom.uv_to_roadXYZ_roadframe_iso8855(141, 460)

    # 只保留2位小数
    ln1_pt1_roadXYZ = np.round(ln1_pt1_roadXYZ, 2)
    ln1_pt2_roadXYZ = np.round(ln1_pt2_roadXYZ, 2)
    ln2_pt1_roadXYZ = np.round(ln2_pt1_roadXYZ, 2)
    ln2_pt2_roadXYZ = np.round(ln2_pt2_roadXYZ, 2)

    print("ln1_pt1_roadXYZ:")
    print(ln1_pt1_roadXYZ)
    print("ln1_pt2_roadXYZ:")
    print(ln1_pt2_roadXYZ)
    print("ln2_pt1_roadXYZ:")
    print(ln2_pt1_roadXYZ)
    print("ln2_pt2_roadXYZ:")
    print(ln2_pt2_roadXYZ)
    
    time1 = time.time()
    carlibed_cam_geom.precompute_bidirectional_mapping()
    time2 = time.time()
    print("precompute_bidirectional_mapping time: {}".format(time2 - time1))

    ln1_pt1_roadXYZ_fast = np.round(carlibed_cam_geom.uv_to_roadxy_iso8855_fast(583, 170), 2) 
    ln1_pt2_roadXYZ_fast = np.round(carlibed_cam_geom.uv_to_roadxy_iso8855_fast(981, 460), 2)
    ln2_pt1_roadXYZ_fast = np.round(carlibed_cam_geom.uv_to_roadxy_iso8855_fast(522, 170), 2)
    ln2_pt2_roadXYZ_fast = np.round(carlibed_cam_geom.uv_to_roadxy_iso8855_fast(141, 460), 2)

    print("Please check the following results:")
    print("ln1_pt1_roadXYZ_fast:")
    print(ln1_pt1_roadXYZ_fast)
    print("ln1_pt2_roadXYZ_fast:")
    print(ln1_pt2_roadXYZ_fast)
    print("ln2_pt1_roadXYZ_fast:")
    print(ln2_pt1_roadXYZ_fast)
    print("ln2_pt2_roadXYZ_fast:")
    print(ln2_pt2_roadXYZ_fast)

    uv_coords = carlibed_cam_geom.roadxy_iso8855_to_uv_fast(ln1_pt1_roadXYZ_fast[0], ln1_pt1_roadXYZ_fast[1])
    print("uv_coords: u = {}, v = {}".format(uv_coords[0], uv_coords[1]))

    uv_coords = [
        [583, 170],
        [981, 460],
        [522, 170],
        [141, 460]
    ]
    road_coords = carlibed_cam_geom.uv_coords_to_roadxy_iso8855_fast(uv_coords)
    print("road_coords:")
    print(road_coords)
