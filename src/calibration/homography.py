"""
Homography module for pixel-to-pitch mapping.
"""
from typing import Optional, List, Tuple
import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("Homography")

class HomographyEstimator:
    def __init__(self, config):
        """
        Initialize Homography Estimator.
        """
        self.config = config.calibration
        self.pitch_length = self.config.pitch_length
        self.pitch_width = self.config.pitch_width
        self.homography_matrix = None
        
    def update(self, src_points: np.ndarray, dst_points: np.ndarray):
        """
        Update the homography matrix using known point correspondences.
        
        Args:
            src_points: Points in image (pixels), shape (N, 2)
            dst_points: Points in real world (meters), shape (N, 2)
        """
        if len(src_points) < 4:
            logger.warning("Need at least 4 points to compute homography.")
            return

        h, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        
        if h is not None:
            self.homography_matrix = h
            logger.info("Homography matrix updated successfully.")
        else:
            logger.error("Failed to compute homography matrix.")

    def pixel_to_pitch(self, points: np.ndarray) -> np.ndarray:
        """
        Map pixel coordinates to real-world pitch coordinates (meters).
        
        Args:
            points: (N, 2) array of [x, y] in pixels
            
        Returns:
            (N, 2) array of [x, y] in meters
        """
        if self.homography_matrix is None:
            return np.zeros_like(points) # No calibration
            
        # Reshape for perspectiveTransform: (1, N, 2)
        pts_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        
        dst = cv2.perspectiveTransform(pts_reshaped, self.homography_matrix)
        
        return dst.reshape(-1, 2)

    def pitch_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """
        Map real-world coordinates back to pixels (for visualization).
        """
        if self.homography_matrix is None:
            return np.zeros_like(points)
            
        inv_h = np.linalg.inv(self.homography_matrix)
        pts_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        dst = cv2.perspectiveTransform(pts_reshaped, inv_h)
        return dst.reshape(-1, 2)
