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
LOG = init_logger.get_logger(log_file_name_prefix="lanenet_test")
LANENET_WIDTH = 512
LANENET_HEIGHT = 256

# filter the lanes those are not straight enough
STRAIGHT_FIT_PARAM_THRESHOLD = [0.01, 10]


class LaneDetector:
    def __init__(
        self,
        cam_geom=CameraGeometry(CameraName.CARLA, field_of_view_deg=45),
        model_path="./weights/tusimple_lanenet_zyb0213/best_model_miou0.6358.ckpt-270",
        debug=False,
    ):
        self.debug = debug
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        self.model_path = model_path
        self.width = LANENET_WIDTH
        self.height = LANENET_HEIGHT
        self.device = None  # TODO: cuda or cpu
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
        return f"./data/camera_{camera_name}_ipm_remap.yml"

    def read_imagefile_to_array(self, filename, rotate_180=False):
        # preprocess image
        image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        if image is None:
            LOG.error("Failed to read image: {}".format(filename))
            return None, None
        image, original_image = self.preprocess_image(image, rotate_180)
        return image, original_image
    
    def preprocess_image(self, image, rotate_180=False):
        if rotate_180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        # TODO: undistort image
        resize_image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        resize_image = resize_image / 127.5 - 1.0
        return resize_image, image

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

    def _postprocess(
        self,
        binary_seg_image,
        instance_seg_image,
        original_image,
        with_lane_fit=False,
        data_source="tusimple",
        with_2d_lane_fit=False,
    ):
        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image,
            instance_seg_result=instance_seg_image,
            source_image=original_image,
            with_lane_fit=with_lane_fit,
            data_source=data_source,
            with_2d_lane_fit=with_2d_lane_fit,
            cam_geom=self.cg,
        )
        return postprocess_result

    def detect(self, img_array, original_image):
        binary_seg_image, instance_seg_image = self._predict(img_array)
        postprocess_result = self._postprocess(
            binary_seg_image,
            instance_seg_image,
            original_image,
            with_lane_fit=True,
            data_source="INHAND",
            with_2d_lane_fit=False,
        )
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
    def __init__(
        self,
        cam_geom=CameraGeometry(camera_name=CameraName.CARLA, field_of_view_deg=45),
        model_path="./weights/tusimple_lanenet_zyb0213/best_model_miou0.6358.ckpt-270",
        debug=False,
    ):
        super().__init__(cam_geom, model_path, debug)
        self.estimated_pitch_deg = 0
        self.estimated_yaw_deg = 0
        self.mean_residuals_thresh = 15
        #self.update_cam_geometry()
        self.pitch_yaw_history = []
        self.calibration_success = False

    def detect4calibration(self, img_array, original_image):
        # Fit lane lines in 2D image
        binary_seg_image, instance_seg_image = self._predict(img_array)
        postprocess_result = self._postprocess(
            binary_seg_image,
            instance_seg_image,
            original_image,
            with_lane_fit=False,
            data_source="TODO",
            with_2d_lane_fit=True,
        )

        return postprocess_result

    def detect_vanishing_point_from_file(self, filename):
        # Detect vanishing point in 3D space
        if self.cg.camera_name == "accord_camera":
            rotate_180 = True
        else:
            rotate_180 = False
        img_array, original_image = self.read_imagefile_to_array(filename, rotate_180)
        return self.detect_vanishing_point(img_array, original_image)

    def detect_vanishing_point(self, img_array, original_image):
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

        color_map = lanenet_postprocess.COLOR_MAP

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
            lane_points = np.vstack(
                resized_lane_coords
            ).T  # Convert to Nx2 array [x, y]

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
                y_end = int(self.cg.image_height * 0.9)  # original image height
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
                cv2.circle(
                    original_image, (int(vp[2][0]), int(vp[2][1])), 5, (255, 0, 0), -1
                )

        filtered_vp_mean, filtered_vp_list = filter_vp_list(vp_list)
        if self.debug:
            print("filtered_vp_mean:")
            print(filtered_vp_mean)
            print("filtered_vp_list:")
            print(filtered_vp_list)

        if filtered_vp_mean is not None and self.debug:
            # 在原图上绘制过滤后的消失点
            cv2.circle(
                original_image,
                (int(filtered_vp_mean[0]), int(filtered_vp_mean[1])),
                10,
                (0, 0, 255),
                -1,
            )
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
            mean_pitch = np.mean(py[:, 0])
            mean_yaw = np.mean(py[:, 1])
            self.estimated_pitch_deg = np.rad2deg(mean_pitch)
            self.estimated_yaw_deg = np.rad2deg(mean_yaw)
            self.update_cam_geometry()
            self.calibration_success = True
            self.pitch_yaw_history = []
            print("yaw, pitch = ", self.estimated_yaw_deg, self.estimated_pitch_deg)
            self.cg.precompute_bidirectional_mapping()
            self.cg.save_forward_map_to_file(self.cg.get_forward_map_filename())

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
            camera_name=self.cg.camera_name,
            height=self.cg.height,
            roll_deg=self.cg.roll_deg,
            image_width=self.cg.image_width,
            image_height=self.cg.image_height,
            field_of_view_deg=self.cg.field_of_view_deg,
            pitch_deg=self.estimated_pitch_deg,
            yaw_deg=self.estimated_yaw_deg,
        )
        #self.cut_v, self.grid = self.cg.precompute_grid()


def test_virtual_camera():
    test_image_path = "./data/carla_vp_calib01.png"    
    #test_image_path = "./data/carla/calibration_video_imgs/frame_00011033ms.jpg"
    carla_cam_geom = CameraGeometry(
        camera_name=CameraName.CARLA,
        height=1.3,
        roll_deg=0,
        pitch_deg=-5.0, # -5.0
        yaw_deg=-2.0, # -2.0
        image_width=1024,
        image_height=512,
        field_of_view_deg=45,
    )
    print("Precompute bidirectional mapping...")
    carla_cam_geom.precompute_bidirectional_mapping()

    print("is_forward_map_precomputed: {}".format(carla_cam_geom.is_forward_map_precomputed()))
    
    calib_lane_detector = CalibLaneDetector(carla_cam_geom, debug=True)
    # Use the calibrated camera geometry to detect the lane lines in the image
    print("Use the calibrated camera geometry to detect the lane lines in the image...")
    calib_lane_detector.detect_from_file(test_image_path)    


def test_detect_video():
    #video_path = "./data/carla/calibration_video.mp4"
    video_path = "./data/route28/road28_66_20250306_11_12_47_Pro.mp4"
    interval = 1.0 # seconds
    rotate_180 = True

    # <video_name>_output.mp4
    output_video_path = video_path.replace(".mp4", "_output.mp4")

    if False:
        carla_cam_geom = CameraGeometry(
            camera_name=CameraName.CARLA,
            height=1.3, # meters
            roll_deg=0,
            image_width=1024,
            image_height=512,
            field_of_view_deg=45,
        )
        cam_geom = carla_cam_geom
    else:
        accord_cam_geom = CameraGeometry(
            camera_name=CameraName.ACCORD_LENOVO,
            height=1.2, # meters
            roll_deg=0,
            image_width=1920,
            image_height=1080,
        )
        cam_geom = accord_cam_geom

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Calculate frame interval
    frame_interval = int(fps * interval)

    calib_lane_detector = CalibLaneDetector(cam_geom, debug=True)
    
    frame_number = 0
    saved_count = 0
    
    # create a new output video file
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_fps = int(1.0 / interval)

    try:
        os.remove(output_video_path)
    except:
        pass
    out_video = cv2.VideoWriter(output_video_path, fourcc, output_fps, 
                                (cam_geom.image_width, cam_geom.image_height),
                                isColor=True)
    
    # 添加写入检查
    if not out_video.isOpened():
        raise RuntimeError(f"无法创建视频文件，请检查编码器 {fourcc} 是否支持")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame at specified interval
        if frame_number % frame_interval == 0:            
            print(f"Processing frame {frame_number}...")
            src_image = None
            raw_image = frame.copy()
            resized_image, original_image = calib_lane_detector.preprocess_image(frame, rotate_180)
            if not calib_lane_detector.calibration_success:
                vp, postprocess_result = calib_lane_detector.detect_vanishing_point(resized_image, original_image)
                src_image = postprocess_result["source_image"]
                # display the frame number
                cv2.putText(src_image, f"F#: {frame_number:06d}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if vp is not None:
                    # draw the vanishing point on the original image
                    cv2.circle(src_image, (int(vp[0]), int(vp[1])), 5, (0, 255, 0), -1)
                    pitch, yaw = calib_lane_detector.get_pitch_yaw_from_vp(vp[0], vp[1]) # in radian
                    calib_lane_detector.add_to_pitch_yaw_history(pitch, yaw)
                    
                    if not calib_lane_detector.calibration_success:
                        # draw a RED dot on the left top corner of the original image
                        cv2.circle(src_image, (50, 50), 30, (0, 0, 255), -1)
                        # display the length of the pitch and yaw history
                        cv2.putText(src_image, f"P: {np.rad2deg(pitch):.2f} deg / Y: {np.rad2deg(yaw):.2f} deg, / C: {len(calib_lane_detector.pitch_yaw_history)}", 
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # draw a GREEN dot on the left top corner of the original image
                        cv2.circle(src_image, (50, 50), 30, (0, 255, 0), -1)
                        # display the length of the pitch and yaw history
                        cv2.putText(src_image, f"P: {calib_lane_detector.estimated_pitch_deg:.2f} deg / Y: {calib_lane_detector.estimated_yaw_deg:.2f} deg, / C: {len(calib_lane_detector.pitch_yaw_history)}", 
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(f"./output/route28-vp_calib_{frame_number:06d}.jpg", src_image)
                    cv2.imwrite(f"./output/route28-raw_image_{frame_number:06d}.jpg", raw_image)
                else:
                    # draw a RED dot on the left top corner of the original image
                    cv2.circle(src_image, (50, 50), 30, (0, 0, 255), -1)
                    # display the length of the pitch and yaw history
                    cv2.putText(src_image, f"C: {len(calib_lane_detector.pitch_yaw_history)}", 
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                postprocess_result = calib_lane_detector.detect(resized_image, original_image)
                src_image = postprocess_result["source_image"]
                # display the frame number
                cv2.putText(src_image, f"Frame#: {frame_number:06d}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # draw a GREEN dot on the left top corner of the original image
                cv2.circle(src_image, (50, 50), 30, (0, 255, 0), -1)
                # display the length of the pitch and yaw history
                cv2.putText(src_image, f"P: {calib_lane_detector.estimated_pitch_deg:.2f} deg / Y: {calib_lane_detector.estimated_yaw_deg:.2f} deg, / \
                                    C: {len(calib_lane_detector.pitch_yaw_history)}", 
                                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 在 out_video.write(src_image) 前添加：
            if src_image.dtype != np.uint8:
                src_image = src_image.astype(np.uint8)

            if len(src_image.shape) == 2:  # 灰度图转BGR
                src_image = cv2.cvtColor(src_image, cv2.COLOR_GRAY2BGR)
            elif src_image.shape[2] == 4:  # 带alpha通道
                src_image = cv2.cvtColor(src_image, cv2.COLOR_BGRA2BGR)

            # 确保尺寸匹配
            assert src_image.shape[1] == cam_geom.image_width, "图像宽度不匹配"
            assert src_image.shape[0] == cam_geom.image_height, "图像高度不匹配"

            out_video.write(src_image)
            saved_count += 1
        frame_number += 1
    
    # Release resources
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    print(f"\nProcessing complete. Saved {saved_count} frames.")

def test_camera_calibration():
    # test_image_path = "./data/carla_vp_calib01.png"
    test_image_path = "./data/route28/vp_calib4.jpg"

    if False:
        carla_cam_geom = CameraGeometry(
            camera_name=CameraName.CARLA,
            height=1.3, # meters
            roll_deg=0,
            image_width=1024,
            image_height=512,
            field_of_view_deg=45,
        )
        cam_geom = carla_cam_geom
    else:
        accord_cam_geom = CameraGeometry(
            camera_name=CameraName.ACCORD_LENOVO,
            height=1.2, # meters
            roll_deg=0,
            image_width=1920,
            image_height=1080,
        )
        cam_geom = accord_cam_geom


    calib_lane_detector = CalibLaneDetector(cam_geom, debug=True)
    filtered_vp_mean, postprocess_result = calib_lane_detector.detect_vanishing_point_from_file(
        test_image_path
    )

    # 计算pitch和yaw
    pitch, yaw = calib_lane_detector.get_pitch_yaw_from_vp(
        filtered_vp_mean[0], filtered_vp_mean[1]
    )

    trafo_cam_to_road = calib_lane_detector.cg.trafo_cam_to_road
    print("Before calibration, trafo_cam_to_road:")
    print(trafo_cam_to_road)

    for i in range(51):
        calib_lane_detector.add_to_pitch_yaw_history(pitch, yaw)

    calibed_cam_geom = calib_lane_detector.cg
    trafo_cam_to_road = calibed_cam_geom.trafo_cam_to_road
    print("After calibration, trafo_cam_to_road:")
    print(trafo_cam_to_road)

    ln1_pt1_roadXYZ = calibed_cam_geom.uv_to_roadXYZ_roadframe_iso8855(583, 170)
    ln1_pt2_roadXYZ = calibed_cam_geom.uv_to_roadXYZ_roadframe_iso8855(981, 460)

    ln2_pt1_roadXYZ = calibed_cam_geom.uv_to_roadXYZ_roadframe_iso8855(522, 170)
    ln2_pt2_roadXYZ = calibed_cam_geom.uv_to_roadXYZ_roadframe_iso8855(141, 460)

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
    calibed_cam_geom.precompute_bidirectional_mapping()
    time2 = time.time()

    print("precompute_bidirectional_mapping time: {}".format(time2 - time1))

    carlar_cam_map_file = "./data/carla_cam_map.txt"
    calibed_cam_geom.save_forward_map_to_file(carlar_cam_map_file)

    calibed_cam_geom.load_forward_map_from_file(carlar_cam_map_file)

    ln1_pt1_roadXYZ_fast = np.round(
        calibed_cam_geom.uv_to_roadxy_iso8855_fast(583, 170), 2
    )
    ln1_pt2_roadXYZ_fast = np.round(
        calibed_cam_geom.uv_to_roadxy_iso8855_fast(981, 460), 2
    )
    ln2_pt1_roadXYZ_fast = np.round(
        calibed_cam_geom.uv_to_roadxy_iso8855_fast(522, 170), 2
    )
    ln2_pt2_roadXYZ_fast = np.round(
        calibed_cam_geom.uv_to_roadxy_iso8855_fast(141, 460), 2
    )

    print("Please check the following results:")
    print("ln1_pt1_roadXYZ_fast:")
    print(ln1_pt1_roadXYZ_fast)
    print("ln1_pt2_roadXYZ_fast:")
    print(ln1_pt2_roadXYZ_fast)
    print("ln2_pt1_roadXYZ_fast:")
    print(ln2_pt1_roadXYZ_fast)
    print("ln2_pt2_roadXYZ_fast:")
    print(ln2_pt2_roadXYZ_fast)

    uv_coords = calibed_cam_geom.roadxy_iso8855_to_uv_fast(
        ln1_pt1_roadXYZ_fast[0], ln1_pt1_roadXYZ_fast[1]
    )
    print("uv_coords: u = {}, v = {}".format(uv_coords[0], uv_coords[1]))

    uv_coords = [[583, 170], [981, 460], [522, 170], [141, 460]]
    road_coords = calibed_cam_geom.uv_coords_to_roadxy_iso8855_fast(uv_coords)
    print("road_coords:")
    print(road_coords)

    # Use the calibrated camera geometry to detect the lane lines in the image
    print("Use the calibrated camera geometry to detect the lane lines in the image...")
    calib_lane_detector.detect_from_file(test_image_path)




if __name__ == "__main__":
    #test_virtual_camera()
    #test_camera_calibration()
    test_detect_video()
