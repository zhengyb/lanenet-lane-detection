#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""

import os.path as ops
import math

import cv2
import numpy as np
import loguru
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


from tools.utils import make_instance_seg_img_visuable, minmax_scale, CameraName
from local_utils.config_utils import parse_config_utils
from tools.camera_geometry import CameraGeometry
LOG = loguru.logger


COLOR_MAP = [
    np.array([255, 0, 0]), # 红色
    np.array([0, 255, 0]), # 绿色
    np.array([0, 0, 255]), # 蓝色
    np.array([125, 125, 0]), # 深黄色
    np.array([0, 125, 125]), # 深青色
    np.array([125, 0, 125]), # 深紫色
    np.array([50, 100, 50]), # 深绿色
    np.array([50, 50, 100]), # 深蓝色
    np.array([100, 50, 50]), # 深红色
    np.array([100, 50, 100]), # 深紫色
    np.array([100, 100, 50]), # 深黄色
    np.array([50, 100, 100]), # 深青色
    np.array([255, 255, 0]), # 亮黄色
    np.array([255, 0, 255]), # 亮紫色
    np.array([0, 255, 255]), # 亮青色
]


def figure_to_cv2_image(fig):
    """
    将matplotlib figure转换为OpenCV图像格式（BGR）
    :param fig: matplotlib figure对象
    :return: OpenCV图像（numpy数组）
    """
    # 将figure渲染到canvas
    fig.canvas.draw()
    
    # 获取RGB像素数据
    img_rgb = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]  # 去除alpha通道
    
    # 转换RGB到BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 关闭figure释放内存
    plt.close(fig)
    
    return img_bgr

def _morphological_process(image, kernel_size=5, iterations=2):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError(
            "Binary segmentation result image should be a single channel image"
        )

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_ELLIPSE,  # 椭圆结构元素
        ksize=(kernel_size, kernel_size),
    )  # 结构元素的大小

    # close operation fille hole
    closing = cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,  # 闭运算
        kernel,
        iterations=iterations,
    )  # 迭代次数

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(
        gray_image, connectivity=8, ltype=cv2.CV_32S
    )


class _LaneFeat(object):
    """ """

    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError("Class id must be integer")

        self._class_id = value


class _LaneNetCluster(object):
    """
    Instance segmentation result cluster
    """

    def __init__(self, cfg):
        """ """
        self._color_map = COLOR_MAP
        self._cfg = cfg

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """

        # print the first 5 rows and 5 columns of embedding_image_feats

        db = DBSCAN(
            eps=self._cfg.POSTPROCESS.CLUSTERING.DBSCAN_EPS,
            min_samples=self._cfg.POSTPROCESS.CLUSTERING.DBSCAN_MIN_SAMPLES,
        )
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            LOG.error(err)
            ret = {
                "origin_features": None,
                "cluster_nums": 0,
                "db_labels": None,
                "unique_labels": None,
                "cluster_center": None,
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            "origin_features": features,
            "cluster_nums": num_clusters,
            "db_labels": db_labels,
            "unique_labels": unique_labels,
            "cluster_center": cluster_centers,
        }

        print("Unique labels:")
        print(unique_labels)
        #print("Cluster centers:")
        #print(cluster_centers)

        if False:
            # print the first 5 features of each lable
            for label in unique_labels:
                if label == -1:
                    continue
                print("Label: {}".format(label))
                print("Features:")
                idx = np.where(db_labels == label)
                # print the first 5 features of each label
                print(embedding_image_feats[idx][0:50])

        return ret
    
    @staticmethod
    def make_embedding_feats_visuable(instance_seg_ret, idx):
        # debug: save an image to show the lane embedding feats
        # 将车道线的特征向量转换为图像
        lane_embedding_feats_image = np.zeros(
            (
                instance_seg_ret.shape[0],
                instance_seg_ret.shape[1],
                instance_seg_ret.shape[2],
            ),
            dtype=np.float32,
        )
        print("lane_embedding_feats_image.shape:")
        print(lane_embedding_feats_image.shape)
        lane_embedding_feats_image[idx] = instance_seg_ret[idx]

        return make_instance_seg_img_visuable(lane_embedding_feats_image)


    # @staticmethod
    def _get_lane_embedding_feats(self, binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        # 获取二值分割结果中车道线的像素坐标,idx 返回两个数组：(row_indices, col_indices)
        idx = np.where(binary_seg_ret == 255)
        # 获取车道线的特征向量
        lane_embedding_feats = instance_seg_ret[idx]
        # 将车道线的像素坐标转换为车道线的坐标,transpose 转置得到 Nx2 的坐标矩阵,
        # 每行是一个点的 [x,y] 坐标
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        # debug: save an image to show the lane embedding feats
        # 将车道线的特征向量转换为图像
        lane_embedding_feats_image = self.make_embedding_feats_visuable(instance_seg_ret, idx)

        ret = {
            "lane_embedding_feats": lane_embedding_feats,
            "lane_coordinates": lane_coordinate,
            "lane_embedding_feats_image": lane_embedding_feats_image,
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result, final_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result, instance_seg_ret=instance_seg_result
        )

        if len(get_lane_embedding_feats_result["lane_embedding_feats"]) == 0:
            final_result["lane_embedding_feats_image"] = None
            return None, None
        
        final_result["lane_embedding_feats_image"] = get_lane_embedding_feats_result["lane_embedding_feats_image"]
        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result[
                "lane_embedding_feats"
            ]
        )

        mask = np.zeros(
            shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3],
            dtype=np.uint8,
        )
        db_labels = dbscan_cluster_result["db_labels"]
        unique_labels = dbscan_cluster_result[
            "unique_labels"
        ]  # array([-1,  0,  1,  2,  3,  4]
        coord = get_lane_embedding_feats_result["lane_coordinates"]

        if db_labels is None:
            return None, None

        try:
            lane_coords = []
            for index, label in enumerate(unique_labels.tolist()):
                if label == -1:
                    continue
                # debug: filter not read lanes
                #if label != 1:
                #    continue
                idx = np.where(db_labels == label)
                pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
                mask[pix_coord_idx] = self._color_map[label]
                lane_coords.append(coord[idx])
        except Exception as e:
            print("Error:")
            print(e)

        print("mask.shape:")
        print(mask.shape)
        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """

    def __init__(self, cfg, ipm_remap_file_path="./data/tusimple_ipm_remap.yml", data_source="tusimple"):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        #assert ops.exists(ipm_remap_file_path), "{:s} not exist".format(
        #    ipm_remap_file_path
        #)

        self._cfg = cfg
        self._cluster = _LaneNetCluster(cfg=cfg)
        self._ipm_remap_file_path = ipm_remap_file_path

        remap_file_load_ret = self._load_remap_matrix()
        self._remap_to_ipm_x = remap_file_load_ret["remap_to_ipm_x"]
        self._remap_to_ipm_y = remap_file_load_ret["remap_to_ipm_y"]

        self._color_map = COLOR_MAP

    def _load_remap_matrix(self):
        """

        :return:
        """
        if ops.exists(self._ipm_remap_file_path):
            fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

            remap_to_ipm_x = fs.getNode("remap_ipm_x").mat()
            remap_to_ipm_y = fs.getNode("remap_ipm_y").mat()
            ret = {
                "remap_to_ipm_x": remap_to_ipm_x,
                "remap_to_ipm_y": remap_to_ipm_y,
            }

            fs.release()

            return ret            

        return {
            "remap_to_ipm_x": None,
            "remap_to_ipm_y": None,
        }


    def save_postprocess_result(self, original_image, result, result_file_path):

        empty_image = np.zeros((256, 512, 3), dtype=np.uint8)

        ipm_image = result["ipm_image"]
        mask_image = result["mask_image"]
        lane_embedding_feats_image = result["lane_embedding_feats_image"]
        connect_image = result["connect_components_analysis_ret"]
        morphological_image = result["morphological_ret"]
        instance_seg_image2 = result["instance_seg_result"]
        binary_seg_image2 = result["binary_seg_result"]

        if ipm_image is None:
            LOG.warning("Failed to get ipm image")
            ipm_image = empty_image
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

        axs[0, 1].imshow(ipm_image[:, :, (2, 1, 0)])
        axs[0, 1].set_title("ipm_image")
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


    def postprocess(
        self,
        binary_seg_result,
        instance_seg_result=None,
        # min_area_threshold=100,
        source_image=None,
        with_lane_fit=True,
        data_source="tusimple",
        with_2d_lane_fit=False,
        cam_geom=CameraGeometry(camera_name=CameraName.INVALID, field_of_view_deg=45),
        cut_v_start=0,
        cut_v_end=0,
    ):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param with_lane_fit:
        :param data_source:
        :param cut_v_start: in pixels, original image height
        :param cut_v_end: in pixels, original image height
        :return:
        """

        result = {
            "mask_image": None,
            "fit_params": None,
            "source_image": None,
        }

        debug_image_dir = "/app/test/"

        ORIGINAL_IMAGE_HEIGHT = source_image.shape[0]
        ORIGINAL_IMAGE_WIDTH = source_image.shape[1]
        #print("ORIGINAL_IMAGE_HEIGHT:")
        #print(ORIGINAL_IMAGE_HEIGHT)
        #print("ORIGINAL_IMAGE_WIDTH:")
        #print(ORIGINAL_IMAGE_WIDTH)

        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)
        resized_height = binary_seg_result.shape[0]

        result["binary_seg_result"] = binary_seg_result
        result["instance_seg_result"] = make_instance_seg_img_visuable(instance_seg_result)

        if cut_v_start == 0:
            cut_v_start = int(ORIGINAL_IMAGE_HEIGHT * 0.5)
        if cut_v_end == 0:
            cut_v_end = int(ORIGINAL_IMAGE_HEIGHT * 0.85)
        resized_height_factor = ORIGINAL_IMAGE_HEIGHT / 256
        cut_v_start_resized = int(cut_v_start * resized_height_factor)
        cut_v_end_resized = int(cut_v_end * resized_height_factor)

        binary_seg_result[cut_v_start_resized:cut_v_end_resized, :] = 0

        if self._cfg.POSTPROCESS.MORPHOLOGICAL_PROCESS.ENABLE:
            # apply image morphology operation to fill in the hold and reduce the small area
            # 形态学操作，填充孔洞并减少小面积
            morphological_ret = _morphological_process(
                binary_seg_result,
                kernel_size=self._cfg.POSTPROCESS.MORPHOLOGICAL_PROCESS.KERNEL_SIZE,
                iterations=self._cfg.POSTPROCESS.MORPHOLOGICAL_PROCESS.ITERATIONS,
            )
            cv2.imwrite(debug_image_dir + "5-morphological_ret.jpg", morphological_ret)
        else:
            morphological_ret = binary_seg_result
            
        result["morphological_ret"] = np.copy(morphological_ret)

        if self._cfg.POSTPROCESS.CONNECT_COMPONENTS_ANALYSIS.ENABLE:
            # 连通区域分析
            connect_components_analysis_ret = _connect_components_analysis(
                image=morphological_ret
            )

            num_labels = connect_components_analysis_ret[0]
            labels = connect_components_analysis_ret[1]
            stats = connect_components_analysis_ret[2]
            # 删除面积小于min_area_threshold的连通区域
            for index, stat in enumerate(stats):
                if (
                    stat[4]
                    <= self._cfg.POSTPROCESS.CONNECT_COMPONENTS_ANALYSIS.MIN_AREA_THRESHOLD
                ):
                    idx = np.where(labels == index)
                    morphological_ret[idx] = 0

            cv2.imwrite(
                debug_image_dir + "6-connect_components_analysis_ret.jpg",
                morphological_ret,
            )
        else:
            morphological_ret = morphological_ret

        result["connect_components_analysis_ret"] = np.copy(morphological_ret)

        # apply embedding features cluster
        # mask_image: 不同车道线用不同颜色标记的掩码图像
        # lane_coords: 每条车道线的像素坐标列表, resized coordinates
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,  # 二值分割结果
            instance_seg_result=instance_seg_result,  # 实例分割结果
            final_result=result
        )
        if mask_image is None:
            result["mask_image"] = None
            result["fit_params"] = None
            result["source_image"] = None
            return result
        
        result["mask_image"] = mask_image
        print("mask_image.shape:")
        print(mask_image.shape)

        print("lane num: {}".format(len(lane_coords)))

        result["lane_coords"] = lane_coords

        if (not with_lane_fit and not with_2d_lane_fit) \
            or (with_lane_fit and data_source == "tusimple" and self._remap_to_ipm_x is None) \
            or (with_lane_fit and data_source != "tusimple" and cam_geom.camera_name == CameraName.INVALID):
            print("not with_lane_fit")
            result["ipm_image"] = None
            tmp_mask = cv2.resize(
                mask_image,
                dsize=(source_image.shape[1], source_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            # source_image = cv2.addWeighted(source_image, 0.6, tmp_mask, 0.4, 0.0, dst=source_image)
            source_image = cv2.addWeighted(
                source_image,  # src1: 原始图像
                0.6,  # alpha: 原始图像的权重
                tmp_mask,  # src2: 车道线掩码图像
                0.4,  # beta: 掩码图像的权重
                0.0,  # gamma: 额外的亮度调节值
                dst=source_image,  # dst: 输出图像存储位置
            )
            result["source_image"] = source_image
            return result
        
        elif with_2d_lane_fit:
            print("with_2d_lane_fit")

            # lane line fit
            resized_lane_coords = []
            fit_params = []
            src_lane_pts = []  # lane pts every single lane

            for lane_index, coords in enumerate(lane_coords):  
                
                # 创建一个与原始图像大小相同的掩码图像
                tmp_mask = np.zeros(shape=(ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH), dtype=np.uint8)
                # 修改开始：添加坐标裁剪
                resized_coords = tuple((np.int_(coords[:, 1] * ORIGINAL_IMAGE_HEIGHT / 256), 
                                            np.int_(coords[:, 0] * ORIGINAL_IMAGE_WIDTH / 512)))
                resized_lane_coords.append(resized_coords)
                tmp_mask[resized_coords] = 255
                
                nonzero_y = np.array(tmp_mask.nonzero()[0])
                nonzero_x = np.array(tmp_mask.nonzero()[1])

                # 使用二次多项式对车道线点进行拟合, 在2D空间中
                fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
                fit_params.append(fit_param)

                # 在原始图像tmp_lane_mask上绘制拟合的车道线
                lane_color = self._color_map[lane_index].tolist()

                start_plot_y = int(ORIGINAL_IMAGE_HEIGHT / 3)
                end_plot_y = ORIGINAL_IMAGE_HEIGHT - 1
                step = int(math.floor((end_plot_y - start_plot_y) / 10))

                plot_y = np.linspace(start_plot_y, end_plot_y, step).astype(int)
                fit_x = fit_param[0] * plot_y**2 + fit_param[1] * plot_y + fit_param[2]
    
                # Draw using numpy indexing like mask_image
                lane_color = self._color_map[lane_index].tolist()
                for plot_y, fit_x in zip(plot_y, fit_x):
                    cv2.circle(
                            source_image,
                            (int(fit_x), int(plot_y)),
                            5,
                            lane_color,
                            -1,
                        )    #cv2.polylines(tmp_lane_mask, [points], isClosed=False, color=self._color_map[lane_index].tolist(), thickness=5)
                
            
            
            result["resized_lane_coords"] = resized_lane_coords
            result["source_image"] = source_image
            result["mask_image"] = mask_image
            result["fit_params"] = fit_params
            result["ipm_image"] = None

        else:             
            print("with_3d_lane_fit")

            # 3D lane fit
            fit_params = []
            src_lane_pts = []  # lane pts every single lane    
            lane_colors_index = []            
            if data_source == "tusimple":
                # get BEV image with lane fit
                ipm_image = cv2.remap(
                    source_image,
                    self._remap_to_ipm_x,
                    self._remap_to_ipm_y,
                    interpolation=cv2.INTER_LINEAR,
                )
                result["ipm_image"] = ipm_image

                # lane line fit
                for lane_index, coords in enumerate(lane_coords):
                    # 添加调试信息
                    print(f"Lane {lane_index} coordinates before resize:")
                    print(f"Shape: {coords.shape}")
                    print(f"Sample points: {coords[:5]}")  # 打印前5个点
                    tmp_mask = np.zeros(shape=(ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH), dtype=np.uint8)
                    resized_coords = tuple(
                        (
                            np.int_(coords[:, 1] * ORIGINAL_IMAGE_HEIGHT / 256),
                            np.int_(coords[:, 0] * ORIGINAL_IMAGE_WIDTH / 512),
                        )
                    )

                    tmp_mask[resized_coords] = 255
                    # 添加调试信息
                    print(f"Lane {lane_index} coordinates after resize:")
                    print(f"Points in mask: {np.sum(tmp_mask == 255)}")

                    # 将普通视角的图像转换为鸟瞰图（IPM, Inverse Perspective Mapping）
                    tmp_ipm_mask = cv2.remap(
                        tmp_mask,
                        self._remap_to_ipm_x,
                        self._remap_to_ipm_y,
                        interpolation=cv2.INTER_NEAREST,
                        )
                    # 添加调试信息
                    print(f"Lane {lane_index} after IPM:")
                    print(f"Points in IPM mask: {np.sum(tmp_ipm_mask == 255)}")

                    # Filter the background points
                    nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
                    nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

                    # 使用二次多项式对车道线点进行拟合, in the BEV space
                    fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
                    fit_params.append(fit_param)

                    # get the fit lane points in the BEV space
                    [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
                    plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
                    fit_x = fit_param[0] * plot_y**2 + fit_param[1] * plot_y + fit_param[2]
                    # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

                    # 这个过程的目的是将鸟瞰图中拟合出的车道线点重新映射回原始图像视角，
                    # 这样就可以在原始图像上正确显示检测到的车道线。这种转换是必要的，
                    # 因为我们需要在原始视角下展示结果，而不是鸟瞰图视角。
                    lane_pts = []
                    for index in range(0, plot_y.shape[0], 5):
                        src_x = self._remap_to_ipm_x[
                            int(plot_y[index]),
                            int(np.clip(fit_x[index], 0, ipm_image_width - 1)),
                        ]
                        if src_x <= 0:
                            continue
                        src_y = self._remap_to_ipm_y[
                            int(plot_y[index]),
                            int(np.clip(fit_x[index], 0, ipm_image_width - 1)),
                        ]
                        src_y = src_y if src_y > 0 else 0

                        # the fit lane points in the original image
                        lane_pts.append([src_x, src_y])

                        # draw the lane on the ipm image
                        lane_color = self._color_map[lane_index].tolist()
                        lane_colors_index.append(lane_index)
                        cv2.circle(
                            ipm_image,
                            (int(fit_x[index]), int(plot_y[index])),
                            5,
                            lane_color,
                            -1,
                        )

                    src_lane_pts.append(lane_pts)
            elif cam_geom.camera_name != CameraName.INVALID and cam_geom.camera_name != CameraName.TUSIMPLE: 
                # TODO: InHand data source
                print("cam_geom.camera_name: {}".format(cam_geom.camera_name))
                fig =plt.figure(figsize=(10, 6), dpi=100)
                # 生成高密度采样点（1000个点保证曲线连续）
                MAX_X = 100
                poly_x = np.linspace(0, MAX_X, 1000)
                for lane_index, coords in enumerate(lane_coords):
                    # get the min and max of the coords
                    coords = np.array(coords)         
                    # Resize the lane points to the original image size
                    scaled_u = np.round(coords[:, 0] * ORIGINAL_IMAGE_WIDTH / 512).astype(int)
                    scaled_v = np.round(coords[:, 1] * ORIGINAL_IMAGE_HEIGHT / 256).astype(int)

                    # 添加边界约束
                    scaled_u = np.clip(scaled_u, 0, ORIGINAL_IMAGE_WIDTH - 1)
                    scaled_v = np.clip(scaled_v, 0, ORIGINAL_IMAGE_HEIGHT - 1)

                    original_coords = np.column_stack((scaled_u, scaled_v)) 
                    # debug: VP_v = 147
                    # only keep the coords with v > 147
                    # original_coords = original_coords[original_coords[:, 1] > (147+10)]

                    #start_end_coords = np.array([original_coords[0], original_coords[1], original_coords[-1]])
                    #print("start_end_coords:")
                    #print(start_end_coords)
                   
                    # 将颜色值从0-255转换为0-1的浮点数，并转换BGR到RGB
                    lane_color = np.array(self._color_map[lane_index][::-1]) / 255.0  # [::-1]将BGR转为RGB
                    # Step 1: Map the lane points to the iso8855 coordinate system
                    iso8855_roadXY_coords = cam_geom.uv_coords_to_roadxy_iso8855_fast(original_coords)
                    iso8855_roadXY_coords = np.array(iso8855_roadXY_coords)

                    # get the min and max of the X
                    min_x = np.min(iso8855_roadXY_coords[:, 0])
                    max_x = np.max(iso8855_roadXY_coords[:, 0])
                    print("min_x: {}".format(min_x))
                    print("max_x: {}".format(max_x))

                    #print("iso8855_roadXY_coords:")
                    #print(iso8855_roadXY_coords)
                    # Step 2: Fit the lane points in the iso8855 coordinate system
                    # 添加数据验证
                    valid_mask = ~np.isnan(iso8855_roadXY_coords).any(axis=1)
                    iso8855_roadXY_coords = iso8855_roadXY_coords[valid_mask]

                    # only use the coords with x < 100
                    # iso8855_roadXY_coords = iso8855_roadXY_coords[iso8855_roadXY_coords[:, 0] < 100]

                    # draw the iso8855_roadXY_coords coords
                    # plt.scatter(iso8855_roadXY_coords[:, 1], iso8855_roadXY_coords[:, 0], color=lane_color, marker='*', s=100)

                    if len(iso8855_roadXY_coords) < 3:  # 二次多项式至少需要3个点
                        print(f"跳过车道{lane_index}，有效点数不足: {len(iso8855_roadXY_coords)}")
                        continue

                    # 检查x值是否全相同
                    if np.all(iso8855_roadXY_coords[:, 0] == iso8855_roadXY_coords[0, 0]):
                        print(f"跳过车道{lane_index}，x值无变化")
                        continue                    
                    # X为自变量，Y为因变量
                    if True:
                        # y = ax^2 + bx + c
                        fit_param = np.polyfit(iso8855_roadXY_coords[:, 0], iso8855_roadXY_coords[:, 1], 2)
                        if abs(fit_param[0]) > 0.003:
                            print("Fliter fit_param of lane {}: {}".format(lane_index, fit_param))
                            continue
                    else:
                        # y = bx + c
                        fit_param = np.polyfit(iso8855_roadXY_coords[:, 0], iso8855_roadXY_coords[:, 1], 1)
                        fit_param = np.array([0, fit_param[0], fit_param[1]])

                    fit_params.append(fit_param)
                    print("fit_param of lane {}: {}".format(lane_index, fit_param))

                    fit_y = fit_param[0] * poly_x**2 + fit_param[1] * poly_x + fit_param[2]
                    # 绘制连续曲线
                    plt.plot(fit_y, poly_x, 
                            color=tuple(lane_color),  # 转换为元组格式
                            linewidth=2.5,
                            linestyle='-',
                            label=f'Lane {lane_index+1}')
                    
                    fit_y_min_x = fit_param[0] * min_x**2 + fit_param[1] * min_x + fit_param[2]
                    fit_y_max_x = fit_param[0] * max_x**2 + fit_param[1] * max_x + fit_param[2]
                    # draw the min and max points
                    plt.scatter(fit_y_min_x, min_x, color=lane_color, marker='o', s=100)
                    plt.scatter(fit_y_max_x, max_x, color=lane_color, marker='o', s=100)

                    # Step 3: Map the fit lane points back to the original image
                    fit_uv_coords = cam_geom.road_coords_iso8855_to_uv_coords_fast(np.stack((poly_x, fit_y), axis=1))
                    src_lane_pts.append(fit_uv_coords)
                    lane_colors_index.append(lane_index)
                    pass

                # 获取当前坐标轴并反转X轴
                ax = plt.gca()
                ax.invert_xaxis()  # 新增这行
                # 坐标轴设置
                plt.xlim(10, -10)
                plt.ylim(-10, MAX_X)
                plt.xlabel('Y (m)')
                plt.ylabel('X (m)')
                plt.title('ISO8855 Lane Curves')
                plt.grid(True)
                plt.legend()
                plt.savefig("./output/iso8855_lane_curves.jpg", bbox_inches='tight')

                ipm_image = figure_to_cv2_image(fig)
            else:
                result["source_image"] = source_image
                result["mask_image"] = mask_image
                result["fit_params"] = fit_params
                print("data_source: {}".format(data_source))
                raise ValueError("data_source: {} is not supported".format(data_source))
            
            if True:
                # tusimple test data sample point along y axis every 10 pixels
                source_image_width = source_image.shape[1]
                for index, single_lane_pts in enumerate(src_lane_pts):
                    single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
                    single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
                    start_plot_y = int(ORIGINAL_IMAGE_HEIGHT / 3)
                    end_plot_y = ORIGINAL_IMAGE_HEIGHT
                    step = int(math.floor((end_plot_y - start_plot_y) / 10))
                    for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                        diff = single_lane_pt_y - plot_y
                        fake_diff_bigger_than_zero = diff.copy()
                        fake_diff_smaller_than_zero = diff.copy()
                        fake_diff_bigger_than_zero[np.where(diff <= 0)] = float("inf")
                        fake_diff_smaller_than_zero[np.where(diff > 0)] = float("-inf")
                        idx_low = np.argmax(fake_diff_smaller_than_zero)
                        idx_high = np.argmin(fake_diff_bigger_than_zero)

                        previous_src_pt_x = single_lane_pt_x[idx_low]
                        previous_src_pt_y = single_lane_pt_y[idx_low]
                        last_src_pt_x = single_lane_pt_x[idx_high]
                        last_src_pt_y = single_lane_pt_y[idx_high]

                        if (
                            previous_src_pt_y < start_plot_y
                            or last_src_pt_y < start_plot_y
                            or fake_diff_smaller_than_zero[idx_low] == float("-inf")
                            or fake_diff_bigger_than_zero[idx_high] == float("inf")
                        ):
                            continue

                        interpolation_src_pt_x = (
                            abs(previous_src_pt_y - plot_y) * previous_src_pt_x
                            + abs(last_src_pt_y - plot_y) * last_src_pt_x
                        ) / (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                        interpolation_src_pt_y = (
                            abs(previous_src_pt_y - plot_y) * previous_src_pt_y
                            + abs(last_src_pt_y - plot_y) * last_src_pt_y
                        ) / (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

                        if (
                            interpolation_src_pt_x > source_image_width
                            or interpolation_src_pt_x < 10
                        ):
                            continue

                        lane_color = self._color_map[lane_colors_index[index]].tolist()
                        cv2.circle(
                            source_image,
                            (int(interpolation_src_pt_x), int(interpolation_src_pt_y)),
                            5,
                            lane_color,
                            -1,
                        )
        
            result["source_image"] = source_image
            result["mask_image"] = mask_image
            result["fit_params"] = fit_params
            result["ipm_image"] = ipm_image

        return result
