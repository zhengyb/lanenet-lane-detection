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
from tools.lane_detector import CalibLaneDetector
from yolox.tracking_utils.timer import Timer
from yolox.exp import get_exp
from yolox.data.datasets.bdd_classes import is_bdd_collision_class
from yolox.tracker.byte_tracker import BYTETracker, STrack
from tools.demo_track import make_parser, main_init, track_on_image


CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix="lanenet_test")
LANENET_WIDTH = 512
LANENET_HEIGHT = 256
PITCH_YAW_HISTORY_SIZE = 100
CUT_V_OFFSET = 20

# filter the lanes those are not straight enough
STRAIGHT_FIT_PARAM_THRESHOLD = [0.001, 10] # 0.003 is better than 0.01
#STRAIGHT_FIT_PARAM_THRESHOLD = [0.01, 10]

DEF_LEFT_LANE_FIT_PARAM = [0.0, 0.0, 2.0]
DEF_RIGHT_LANE_FIT_PARAM = [0.0, 0.0, -2.0]

def is_in_mylane(obj_bmwh, left_lane_fit_param, right_lane_fit_param, cam_geom):
    """ 
    input:
        obj_bmwh: [x, y, w, h], x, y is the bottom center of the object
        left_lane_fit_param: [a, b, c]
        right_lane_fit_param: [a, b, c]
        cam_geom: CameraGeometry
    output:
        in_mylane: bool
        x: ISO8855 road x coordinate
        y: ISO8855 road y coordinate
    """
    Y_LEFT_MIN = 20.0
    Y_RIGHT_MAX = -20.0
    # TODO: cut the object by the cut_v
    u, v = int(obj_bmwh[0]), int(obj_bmwh[1])
    x, y = cam_geom.uv_to_roadxy_iso8855_fast(u, v)
    y_left = left_lane_fit_param[0] * x**2 + left_lane_fit_param[1] * x + left_lane_fit_param[2]
    y_right = right_lane_fit_param[0] * x**2 + right_lane_fit_param[1] * x + right_lane_fit_param[2]

    y_left = min(y_left, Y_LEFT_MIN)
    y_right = max(y_right, Y_RIGHT_MAX)
    # In a ISO8855 coordinate system, Y is positive to the left of vehicle
    if y > y_right and y < y_left:
        in_mylane = True
    else:   
        in_mylane = False
    return in_mylane, x, y


def obj_dist_n_speed(obj_bmwh, this_frame_number, obj_prev_bmwh, prev_frame_number, cam_geom, fps=30):
    """
    input:
        obj_bmwh: [x, y, w, h], x, y is the bottom center of the object
        this_frame_number: current frame number
        obj_prev_bmwh: [x, y, w, h], x, y is the bottom center of the object
        prev_frame_number: previous frame number
        cam_geom: CameraGeometry
        fps: frame per second
    output:
        dist: distance to the object, in meters
        speed: speed of the object, in meters/second
        ttc: time to collision, in seconds
    """
    assert this_frame_number > prev_frame_number
    assert obj_bmwh is not None and obj_prev_bmwh is not None

    u, v = int(obj_bmwh[0]), int(obj_bmwh[1])
    u_prev, v_prev = int(obj_prev_bmwh[0]), int(obj_prev_bmwh[1])
    x, y = cam_geom.uv_to_roadxy_iso8855_fast(u, v)
    x_prev, y_prev = cam_geom.uv_to_roadxy_iso8855_fast(u_prev, v_prev)
    # TODO: Only use the X distance, ignore the Y distance and Speed
    delta_dist = x - x_prev
    delta_time = float(this_frame_number - prev_frame_number) / fps
    if abs(delta_dist) < 0.01:
        speed = 0.0
    elif abs(delta_time) < 0.00001:
        speed = np.nan
    else:
        # the object is moving towards the vehicle if speed > 0;
        # the object is moving away from the vehicle if speed < 0;
        speed = - delta_dist / delta_time # in meters/second
    if speed < 0.01:
        ttc = np.nan
    else:
        ttc = x / speed # in seconds

    return x, speed, ttc


def draw_main_obj(src_image, u, v, dist, speed, ttc):
    # 1. draw a red dot at the (u, v)
    cv2.circle(src_image, (int(u), int(v)), 5, (0, 0, 255), -1)
    # 2. draw a red dot at the bottom center of the object
    width = src_image.shape[1]
    height = src_image.shape[0]
    cv2.circle(src_image, (int(width/2), int(height)), 5, (0, 0, 255), -1)
    # 3. draw a red line from the (u, v) to the bottom center of the object
    cv2.line(src_image, (int(u), int(v)), (int(width/2), int(height)), (0, 0, 255), 1)
    # 4. draw the distance and speed
    x = int((u + width/2) / 2)
    y = int((v + height) / 2)
    cv2.putText(src_image, f"D={dist:.2f}m, Spd={speed:.2f}m/s, TTC={ttc:.2f}s", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    pass


def test_detect_video(force_calib=False):
    bt_args = make_parser().parse_args()
    bt_exp = get_exp(bt_args.exp_file, bt_args.name)

    bt_predictor, bt_vis_folder, bt_current_time, bt_args = main_init(bt_exp, bt_args)
    timer = Timer()
    tracker = BYTETracker(bt_args, frame_rate=30)
    tracker_results = []

    #video_path = "./data/carla/calibration_video.mp4"
    video_path = "./data/route28/road28_66_20250306_11_12_47_Pro.mp4"
    interval = 0.5 # seconds
    rotate_180 = True

    # <video_name>_output.mp4
    output_video_path = video_path.replace(".mp4", "_output.mp4")
    roadframe_video_path = video_path.replace(".mp4", "_roadframe.mp4")

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = int(1.0 / interval)

    try:
        os.remove(output_video_path)
        os.remove(roadframe_video_path)
    except:
        pass
    out_video = cv2.VideoWriter(output_video_path, fourcc, output_fps, 
                                (cam_geom.image_width, cam_geom.image_height),
                                isColor=True)
    
    roadframe_video = cv2.VideoWriter(roadframe_video_path, fourcc, output_fps, 
                                (1000, 600),
                                isColor=True)
    
    
    # 添加写入检查
    if not out_video.isOpened():
        raise RuntimeError(f"无法创建视频文件，请检查编码器 {fourcc} 是否支持")

    stop_frame_number = 10* 60 * fps
    start_frame_number = 6 * 60 * fps

    if force_calib:
        calib_lane_detector.calibration_success = False
        calib_lane_detector.estimated_pitch_deg = 0.0
        calib_lane_detector.estimated_yaw_deg = 0.0
        calib_lane_detector.pitch_yaw_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # debug:
        if frame_number > stop_frame_number:
            break
        
        # skip the first 6 minutes
        if frame_number < start_frame_number:
            frame_number += 1
            continue
        # Process frame at specified interval
        if frame_number % frame_interval == 0:            
            print(f"Processing frame {frame_number}...")
            # Lane detection
            print("Lane detection...")
            src_image = None
            raw_image = frame.copy()
            if rotate_180:
                raw_image = cv2.rotate(raw_image, cv2.ROTATE_180)
            resized_image, original_image = calib_lane_detector.preprocess_image(raw_image, rotate_180=False)
            if not calib_lane_detector.calibration_success:
                # TODO: in the product case, only used the images while the vehicle is moving faster than 30 km/h to calibrate the camera
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
                lanedet_postproc_result = calib_lane_detector.detect(resized_image, original_image)
                src_image = lanedet_postproc_result["source_image"] 
                # display the frame number
                cv2.putText(src_image, f"Frame#: {frame_number:06d}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                # draw a line at cut_v
                cv2.line(src_image, (0, calib_lane_detector.cg.cut_v+CUT_V_OFFSET), (cam_geom.image_width, calib_lane_detector.cg.cut_v+CUT_V_OFFSET), (0, 0, 255), 1)
                # draw a line at 0.8 * cam_geom.image_height
                cv2.line(src_image, (0, int(0.8 * cam_geom.image_height)), (cam_geom.image_width, int(0.8 * cam_geom.image_height)), (0, 0, 255), 1)
                if calib_lane_detector.estimated_pitch_deg != 0.0 and calib_lane_detector.estimated_yaw_deg != 0.0:
                    # draw a GREEN dot on the left top corner of the original image
                    cv2.circle(src_image, (50, 50), 30, (0, 255, 0), -1)
                    # display the length of the pitch and yaw history
                    cv2.putText(src_image, f"P: {calib_lane_detector.estimated_pitch_deg:.2f} deg / Y: {calib_lane_detector.estimated_yaw_deg:.2f} deg, / C: {len(calib_lane_detector.pitch_yaw_history)}", 
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                roadframe_lane_image = lanedet_postproc_result["ipm_image"]
                if roadframe_lane_image is not None:
                    # resize to 500x300
                    roadframe_lane_image = cv2.resize(roadframe_lane_image, (500, 300))
                    #roadframe_video.write(roadframe_lane_image)
                    # cover the source image with the roadframe_lane_image at the top right corner
                    src_image[0:300, -500:] = roadframe_lane_image

                left_lane_fit_param = lanedet_postproc_result["left_lane_fit_param"]
                if left_lane_fit_param is None:
                    # TODO: is this right for the FCW detection?
                    left_lane_fit_param = DEF_LEFT_LANE_FIT_PARAM
                right_lane_fit_param = lanedet_postproc_result["right_lane_fit_param"]
                if right_lane_fit_param is None:
                    right_lane_fit_param = DEF_RIGHT_LANE_FIT_PARAM
                if True:
                    # FCW detection
                    print("FCW detection...")
                    bt_result_img, tracker_results =  track_on_image(bt_predictor, tracker, raw_image, frame_number, timer, 
                                                    bt_exp, bt_args, tracker_results, src_image)
                    if tracker_results is not None:
                        # Find the main object
                        main_obj_idx = None
                        nearest_obj_x = 10000 # meters
                        for obj_idx, obj in enumerate(tracker_results):
                            if not is_bdd_collision_class(obj.cls_id):
                                continue
                            if obj.bmwh[1] < calib_lane_detector.cg.cut_v:
                                continue
                            in_mylane, x, y = is_in_mylane(obj.bmwh, left_lane_fit_param, right_lane_fit_param, cam_geom)
                            if not in_mylane:
                                continue
                            if x < nearest_obj_x:
                                nearest_obj_x = x
                                main_obj_idx = obj_idx

                        if main_obj_idx is not None:
                            # finded the main object
                            print("Main object found, track_id: {}".format(tracker_results[main_obj_idx].track_id))
                            this_bmwh = tracker_results[main_obj_idx].bmwh
                            prev_bmwh = STrack.tlwh_to_bmwh(tracker_results[main_obj_idx].prev_tlwh)
                            this_frame_id = tracker_results[main_obj_idx].frame_id
                            prev_frame_id = tracker_results[main_obj_idx].prev_frame_id
                            x_dist, speed, ttc = obj_dist_n_speed(this_bmwh, this_frame_id, prev_bmwh, prev_frame_id, 
                                                            cam_geom, fps)
                            draw_main_obj(bt_result_img, this_bmwh[0], this_bmwh[1], x_dist, speed, ttc)

                            src_image = bt_result_img

                
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
    roadframe_video.release()
    cv2.destroyAllWindows()

    print(f"\nProcessing complete. Saved {saved_count} frames.")





if __name__ == "__main__":
    #test_virtual_camera()
    #test_camera_calibration("./output/route28-raw_hwy/")
    test_detect_video(force_calib=False)
    #test_detector("./data/route28/road28_66_20250306_11_12_47_Pro_imgs")
