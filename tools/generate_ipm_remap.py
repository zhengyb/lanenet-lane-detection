#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate IPM (Inverse Perspective Mapping) remap files
"""
import os
import numpy as np
import cv2
import glob
import argparse
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONT_VIEW_IMAGE_RESIZE_WIDTH = 1920
FRONT_VIEW_IMAGE_RESIZE_HEIGHT = 1080

BEV_IMAGE_WIDTH = 640
BEV_IMAGE_HEIGHT = 640

class IPMRemapGenerator:
    def __init__(self, args):
        """
        Initialize IPM remap generator
        """
        self.checkerboard_size = args.checkerboard_size
        self.square_size = args.square_size
        self.calib_images_dir = args.calib_images_dir
        self.test_image_path = args.test_image_path
        self.output_dir = args.output_dir
        self.visualize = args.visualize

        # Create output directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)

    def calibrate_camera(self):
        """
        Calibrate camera using checkerboard images
        """
        logger.info("Starting camera calibration...")
        
        # Prepare object points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp = objp * self.square_size  # Scale by square size
        
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        # Get all calibration images
        images = glob.glob(os.path.join(self.calib_images_dir, '*.jpg'))
        if not images:
            raise ValueError(f"No calibration images found in {self.calib_images_dir}")
        

        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                logger.warning(f"Failed to read image: {fname}")
                continue
            
            # Resize image to IMAGE_RESIZE_WIDTH and IMAGE_RESIZE_HEIGHT
            img = cv2.resize(img, (FRONT_VIEW_IMAGE_RESIZE_WIDTH, FRONT_VIEW_IMAGE_RESIZE_HEIGHT))
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                
                if self.visualize:
                    cv2.drawChessboardCorners(img, self.checkerboard_size, corners, ret)
                    cv2.imshow('Corners', img)
                    cv2.waitKey(500)
        
        if self.visualize:
            cv2.destroyAllWindows()
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        logger.info("Camera calibration completed")
        return mtx, dist

    def get_perspective_transform(self, ):
        """
        Calculate perspective transform matrix
        """
        height, width = BEV_IMAGE_HEIGHT, BEV_IMAGE_WIDTH
        
        # Define source points (adjust these based on your camera setup)
        # coordinates in a 640*640 image
        # src_points = np.float32([
        #     [235, 471],    # Top left
        #     [415, 474],    # Top right
        #     [499, 587],      # Bottom right
        #     [141, 583],    # Bottom left
        # ])
        # coordinates in a 512*256 image
        #src_points = np.float32([
        #    [188, 188.4],    # Top left
        #    [332, 189.6],    # Top right
        #    [399, 234.8],      # Bottom right
        #    [112, 233.2],    # Bottom left
        #])
        # coordinates in a 1920*1080 image
        src_points = np.float32([
            [705, 795],    # Top left
            [1245, 800],    # Top right
            [1497, 990.6],      # Bottom right
            [423, 983.8],    # Bottom left
        ])
        
        # Define destination points
        # coordinates in a 640*640 image
        dst_points = np.float32([
            [width * 0.2, height * 0.6],                # Top left
            [width * 0.8, height * 0.6],                # Top right
            [width * 0.8, height * 0.9],            # Bottom right
            [width * 0.2, height * 0.9],           # Bottom left
        ])
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return M, src_points, dst_points

    def create_remap_matrix(self, M):
        """
        Generate remap matrices for IPM
        """
        height, width = BEV_IMAGE_HEIGHT, BEV_IMAGE_WIDTH
        
        M_inv = np.linalg.inv(M)

        # Create coordinate grid for the BEV image
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        # Output the first 10 points of the grid
        print(x[0:10, 0:10])
        print(y[0:10, 0:10])

        points = np.float32([x.flatten(), y.flatten()]).T
        
        # Output the first 10 points of the points
        print(points[0:10])
        
        # Apply perspective transform
        points_homogeneous = np.array([points])

        # Transform the points from the BEV image to the front view image
        transformed_points = cv2.perspectiveTransform(points_homogeneous, M_inv)
        
        # output the shape of the transformed_points
        print(transformed_points.shape)
        # Output the first 10 points of the transformed_points
        print(transformed_points[0:10])
        # Reshape to image size
        remap_x = transformed_points[0, :, 0]
        remap_y = transformed_points[0, :, 1]

        # reshape to image size
        remap_x = remap_x.reshape((width, height)).astype(np.float32)
        remap_y = remap_y.reshape((width, height)).astype(np.float32)

        return remap_x, remap_y


    def save_remap_matrix_to_yml(self, remap_x, remap_y, file_path):
        """
        Save the remap matrices to the yml file
        """
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_WRITE)
        fs.write('remap_ipm_x', remap_x)
        fs.write('remap_ipm_y', remap_y)
        fs.release()

    def save_M_matrix_to_yaml(self, M, file_path):
        """
        Save the M matrix to the yaml file
        """
        with open(file_path, 'w') as f:
            f.write(f"M:\n")
            f.write(f"  rows: {M.shape[0]}\n")
            f.write(f"  cols: {M.shape[1]}\n")
            f.write(f"  dt: f\n")
            f.write(f"  data: {M.flatten().tolist()}\n")

    def load_M_matrix_from_yaml(self, file_path):
        """
        Load the M matrix from the yaml file
        """
        with open(file_path, 'r') as f:
            M_yaml = yaml.load(f, Loader=yaml.FullLoader)

        M = np.array(M_yaml['M']['data']).reshape(M_yaml['M']['rows'], M_yaml['M']['cols'])
        return M

    def generate_and_save(self):
        """
        Generate and save IPM remap files
        """
        # Read test image to get size
        test_image = cv2.imread(self.test_image_path)
        if test_image is None:
            raise ValueError(f"Failed to read test image: {self.test_image_path}")
        
        # Get perspective transform
        M, src_points, dst_points = self.get_perspective_transform()
        
        # Print the M matrix
        print("M:")
        print(M)
        # save the M to the yaml file
        self.save_M_matrix_to_yaml(M, os.path.join(self.output_dir, 'M.yaml'))

        # load the M from the yaml file
        M_loaded = self.load_M_matrix_from_yaml(os.path.join(self.output_dir, 'M.yaml'))
        # Print the M matrix
        print("M_loaded:")
        print(M_loaded)

        # Create remap matrices
        logger.info("Generating remap matrices...")
        remap_x, remap_y = self.create_remap_matrix(M)
        
        self.save_remap_matrix_to_yml(remap_x, remap_y, os.path.join(self.output_dir, 'lab_lenovo_ipm_remap.yml'))

        # test the perspective transform
        #test_points = src_points
        #print("test_points:")
        #print(test_points)
        #test_points_homogeneous = np.array([test_points])
        #transformed_points = cv2.perspectiveTransform(test_points_homogeneous, M)
        #print("transformed_points:")
        #print(transformed_points)

        # Resize image to IMAGE_RESIZE_WIDTH and IMAGE_RESIZE_HEIGHT
        test_image = cv2.resize(test_image, (FRONT_VIEW_IMAGE_RESIZE_WIDTH, FRONT_VIEW_IMAGE_RESIZE_HEIGHT))
        
        #image_size = (test_image.shape[0], test_image.shape[1])

        # Test and visualize results
        ipm_image = cv2.remap(test_image, remap_x, remap_y, cv2.INTER_LINEAR)
        #ipm_image = cv2.warpPerspective(test_image, M_loaded, (BEV_IMAGE_WIDTH, BEV_IMAGE_HEIGHT))
            
        # Draw source points on original image
        for pt in src_points:
            cv2.circle(test_image, tuple(pt.astype(int)), 5, (0, 0, 255), -1)

        # Save visualization
        cv2.imwrite(os.path.join(self.output_dir, 'original_with_points.jpg'), test_image)
        cv2.imwrite(os.path.join(self.output_dir, 'ipm_result.jpg'), ipm_image)
        
        if self.visualize:
            cv2.imshow('Original Image', test_image)
            cv2.imshow('IPM Result', ipm_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            

        logger.info("IPM remap generation completed successfully!")

def main():
    """

    python generate_ipm_remap.py \
    --checkerboard_size "(6,9)" \
    --square_size 0.025 \
    --calib_images_dir ./calibration_images \
    --test_image_path ./test_images/test.jpg \
    --output_dir ./output \
    --visualize

    """
    parser = argparse.ArgumentParser(description='Generate IPM remap files')
    parser.add_argument('--checkerboard_size', type=tuple, default=(6, 9),
                        help='Size of the checkerboard (rows, cols)')
    parser.add_argument('--square_size', type=float, default=0.025,
                        help='Size of checkerboard squares in meters')
    parser.add_argument('--calib_images_dir', type=str, default='./data/calibration_images/',
                        help='Directory containing calibration images')
    parser.add_argument('--test_image_path', type=str, default='./data/ipm_test_image.jpg',
                        help='Path to test image')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for remap files')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the calibration and results')
    
    args = parser.parse_args()
    
    try:
        generator = IPMRemapGenerator(args)
        generator.generate_and_save()
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
