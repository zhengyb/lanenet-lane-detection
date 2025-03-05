import cv2
import numpy as np
from tools.utils import CameraName
from scipy.interpolate import LinearNDInterpolator


def get_intrinsic_matrix4carla(image_width, image_height, field_of_view_deg=45):
    # For our Carla camera alpha_u = alpha_v = alpha
    # alpha can be computed given the cameras field of view via
    if field_of_view_deg is None:
        raise ValueError("field_of_view_deg must be provided for Carla camera")
    field_of_view_rad = field_of_view_deg * np.pi / 180
    alpha = (image_width / 2.0) / np.tan(field_of_view_rad / 2.0)
    Cu = image_width / 2.0
    Cv = image_height / 2.0
    return np.array([[alpha, 0, Cu], [0, alpha, Cv], [0, 0, 1.0]])


def get_intrinsic_matrix(camera_name):
    # return the intrinsic matrix of the camera, a 3x3 numpy array
    # TODO: implement this
    raise NotImplementedError("Not implemented")


def project_polyline(polyline_world, trafo_world_to_cam, K):
    x, y, z = polyline_world[:, 0], polyline_world[:, 1], polyline_world[:, 2]
    homvec = np.stack((x, y, z, np.ones_like(x)))
    proj_mat = K @ trafo_world_to_cam[:3, :]
    pl_uv_cam = (proj_mat @ homvec).T
    u = pl_uv_cam[:, 0] / pl_uv_cam[:, 2]
    v = pl_uv_cam[:, 1] / pl_uv_cam[:, 2]
    return np.stack((u, v)).T


class CameraGeometry(object):
    def __init__(
        self,
        camera_name,
        height=1.3,
        yaw_deg=0,
        pitch_deg=-5,
        roll_deg=0,
        image_width=1920,
        image_height=1080,
        field_of_view_deg=None,
        debug=True,
    ):
        # scalar constants
        self.camera_name = camera_name
        self.height = height
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.yaw_deg = yaw_deg
        self.image_width = image_width
        self.image_height = image_height
        self.field_of_view_deg = field_of_view_deg  # 45
        # camera intriniscs and extrinsics
        if self.camera_name == CameraName.CARLA or self.camera_name == CameraName.BEV or self.camera_name == CameraName.INVALID:
            self.intrinsic_matrix = get_intrinsic_matrix4carla(
                self.image_width, self.image_height, self.field_of_view_deg
            )
        else:
            self.intrinsic_matrix = get_intrinsic_matrix(self.camera_name)
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        ## Note that "rotation_cam_to_road" has the math symbol R_{rc} in the book
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation_road_to_cam = np.array(
            [
                [cr * cy + sp * sr + sy, cr * sp * sy - cy * sr, -cp * sy],
                [cp * sr, cp * cr, sp],
                [cr * sy - cy * sp * sr, -cr * cy * sp - sr * sy, cp * cy],
            ]
        )
        self.rotation_cam_to_road = (
            rotation_road_to_cam.T
        )  # for rotation matrices, taking the transpose is the same as inversion
        self.translation_cam_to_road = np.array([0, -self.height, 0])
        self.trafo_cam_to_road = np.eye(4)
        self.trafo_cam_to_road[0:3, 0:3] = self.rotation_cam_to_road
        self.trafo_cam_to_road[0:3, 3] = self.translation_cam_to_road
        # compute vector nc. Note that R_{rc}^T = R_{cr}
        self.road_normal_camframe = self.rotation_cam_to_road.T @ np.array([0, 1, 0])
        self.uv_to_roadXYZ_iso8855_tbl = []
        self.roadXYZ_iso8855_to_uv_tbl = []
        self.cut_v = 0
        self.forward_map_x = None
        self.forward_map_y = None
        self.debug = debug

    def camframe_to_roadframe(self, vec_in_cam_frame):
        return (
            self.rotation_cam_to_road @ vec_in_cam_frame + self.translation_cam_to_road
        )

    def roadframe_to_camframe(self, vec_in_road_frame):
        return self.rotation_cam_to_road.T @ (
            vec_in_road_frame - self.translation_cam_to_road
        )

    def uv_to_roadXYZ_camframe(self, u, v):
        # NOTE: The results depend very much on the pitch angle (0.5 degree error yields bad result)
        # Here is a paper on vehicle pitch estimation:
        # https://refubium.fu-berlin.de/handle/fub188/26792
        uv_hom = np.array([u, v, 1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe.dot(Kinv_uv_hom)
        return self.height * Kinv_uv_hom / denominator

    def uv_to_roadXYZ_roadframe(self, u, v):
        r_camframe = self.uv_to_roadXYZ_camframe(u, v)
        return self.camframe_to_roadframe(r_camframe)

    def uv_to_roadXYZ_roadframe_iso8855(self, u, v):
        X, Y, Z = self.uv_to_roadXYZ_roadframe(u, v)
        return np.array(
            [Z, -X, -Y]
        )  # read book section on coordinate systems to understand this


    def save_forward_map_to_file(self, filename):
        # Only save the point where X or Y is not 0
        with open(filename, 'w') as f:
            f.write("#V, U, X, Y\n")
            for v in range(self.image_height):
                for u in range(self.image_width):
                    X, Y = self.forward_map_x[v, u], self.forward_map_y[v, u]
                    if X != 0 or Y != 0:
                        f.write(f"{v}, {u}, {X}, {Y}\n")

        print(f"Forward map saved to {filename}")

    def load_forward_map_from_file(self, filename):
        cut_v = 0
        forward_map_x = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        forward_map_y = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        road_points = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("#V, U, X, Y"):
                    continue
                v, u, X, Y = map(float, line.split(","))
                v = int(v)
                u = int(u)
                forward_map_x[v, u] = X
                forward_map_y[v, u] = Y
                if cut_v == 0:
                    cut_v = v
                road_points.append((X, Y, u, v))

        road_points = np.array(road_points)
        self.forward_map_x = forward_map_x
        self.forward_map_y = forward_map_y

        # Get cut_v and road_points
        self.cut_v = cut_v
        # get slice of road_points where v == cut_v
        cut_v_points = road_points[road_points[:, 3] == cut_v]
        # find the point which Y is closest to 0
        closest_point = cut_v_points[np.argmin(np.abs(cut_v_points[:, 1]))]
        self.cut_dist = np.linalg.norm(closest_point[:2])

        self._precompute_inverse_mapping(road_points)
        if self.debug:
            print("Center point of the CUT_V: ", closest_point)
            print("Cut distance: ", self.cut_dist)
            print("Load forward map from file: {}".format(filename))

                

    def precompute_bidirectional_mapping(self, dist=300):
        """预计算双向映射表"""
        # 正向映射：uv -> road (ISO 8855)
        self.forward_map_x = np.zeros(
            (self.image_height, self.image_width), dtype=np.float32
        )
        self.forward_map_y = np.zeros(
            (self.image_height, self.image_width), dtype=np.float32
        )

        # 填充映射表
        cut_v = int(self.compute_minimum_v(dist=dist) + 1)
        self.cut_v = cut_v
        self.cut_dist = dist
        if self.debug:
            print("Cut distance: {}, Cut V: {}".format(self.cut_dist, self.cut_v))

        # 正向映射填充        
        road_points = []
        for v in range(cut_v, self.image_height):
            for u in range(self.image_width):
                # 正向映射：uv -> road
                X, Y, Z = self.uv_to_roadXYZ_roadframe_iso8855(u, v)
                self.forward_map_x[v, u] = X
                self.forward_map_y[v, u] = Y

                # 收集逆向映射数据
                road_points.append((X, Y, u, v))

        self._precompute_inverse_mapping(road_points)

    def _precompute_inverse_mapping(self, road_points):
        # road_points: list of (X, Y, u, v)
        # 逆向映射：road (ISO 8855) -> uv
        # 需要定义道路坐标的离散化范围
        self.road_x_min, self.road_x_max = -10, 100  # 根据实际情况调整
        self.road_y_min, self.road_y_max = -10, 10
        self.road_resolution = 0.1  # 米/像素

        # 创建逆向映射网格
        road_grid_w = int((self.road_x_max - self.road_x_min) / self.road_resolution)
        road_grid_h = int((self.road_y_max - self.road_y_min) / self.road_resolution)
        self.inverse_map_u = np.zeros((road_grid_h, road_grid_w), dtype=np.float32)
        self.inverse_map_v = np.zeros((road_grid_h, road_grid_w), dtype=np.float32)

        # 构建逆向查找表（使用网格插值）
        road_coords = np.array([(p[0], p[1]) for p in road_points])  # X, Y
        uv_values = np.array([(p[2], p[3]) for p in road_points])  # u, v

        # 创建逆向插值器
        self.inverse_interpolator_u = LinearNDInterpolator(road_coords, uv_values[:, 0])
        self.inverse_interpolator_v = LinearNDInterpolator(road_coords, uv_values[:, 1])

    def uv_to_roadxy_iso8855_fast(self, u, v):
        """使用预计算的映射表实现uv到road坐标的映射"""
        if self.forward_map_x is None or self.forward_map_y is None:
            raise ValueError("Forward map is not precomputed")
        return self.forward_map_x[v, u], self.forward_map_y[v, u]

    def is_forward_map_precomputed(self):
        return self.forward_map_x is not None and self.forward_map_y is not None

    def uv_coords_to_roadxy_iso8855_fast(self, uv_coords):
        """
        使用预计算的映射表实现uv到road坐标的映射    
        input: uv_coords: Nx2 数组，包含[u, v]坐标
        output: road_coords: Nx2 数组，包含[X, Y]坐标
        """
        if not self.is_forward_map_precomputed():
            raise ValueError("Forward map is not precomputed")
        
        uv_coords = np.array(uv_coords)
        u, v = uv_coords[:, 0], uv_coords[:, 1]
        road_coords = np.stack((self.forward_map_x[v, u], self.forward_map_y[v, u]), axis=1)
        return road_coords

    def road_coords_iso8855_to_uv_coords_fast(self, road_coords):
        """
            使用插值器实现road到uv的映射
            input: road_coords: Nx2 数组，包含[X, Y]坐标
            output: uv_coords: Nx2 数组，包含[u, v]坐标
        """  

        if self.inverse_interpolator_u is None or self.inverse_interpolator_v is None:
            raise ValueError("Inverse map is not precomputed")
        u = self.inverse_interpolator_u(road_coords)
        v = self.inverse_interpolator_v(road_coords)
        return np.stack((u, v), axis=1).astype(np.int32)

    def roadxy_iso8855_to_uv_fast(self, x, y):
        """使用插值器实现road到uv的映射"""
        if self.inverse_interpolator_u is None or self.inverse_interpolator_v is None:
            raise ValueError("Inverse map is not precomputed")
        road_coords = np.array(np.array([x, y]))
        uv_coords = self.road_coords_iso8855_to_uv_coords_fast(road_coords)
        return uv_coords[0][0], uv_coords[0][1]

    def precompute_grid(self, dist=300):
        cut_v = int(self.compute_minimum_v(dist=dist) + 1)
        xy = []
        for v in range(cut_v, self.image_height):
            for u in range(self.image_width):
                X, Y, Z = self.uv_to_roadXYZ_roadframe_iso8855(u, v)
                xy.append(np.array([X, Y]))
        xy = np.array(xy)
        return cut_v, xy

    def compute_minimum_v(self, dist):
        """
        Find cut_v such that pixels with v<cut_v are irrelevant for polynomial fitting.
        Everything that is further than `dist` along the road is considered irrelevant.
        """
        trafo_road_to_cam = np.linalg.inv(self.trafo_cam_to_road)
        point_far_away_on_road = trafo_road_to_cam @ np.array([0, 0, dist, 1])
        uv_vec = self.intrinsic_matrix @ point_far_away_on_road[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v
    
