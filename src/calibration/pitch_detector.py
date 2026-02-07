"""
Pitch detection module for detecting field lines.
"""
from typing import List, Tuple
import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("PitchDetector")

class PitchDetector:
    def __init__(self, config):
        """
        Initialize Pitch Detector.
        """
        self.config = config.calibration
        
    def detect_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect white pitch lines in the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            np.ndarray: Array of lines [x1, y1, x2, y2]
        """
        # Convert to HSV to isolate green/white
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define ranges for white lines on green grass
        # This is tricky and might need tuning per video
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Edge detection
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=20
        )
        
        if lines is not None:
             # Squeeze to (N, 4)
            lines = lines.squeeze()
            if len(lines.shape) == 1: # Handle single line case
                lines = lines[np.newaxis, :]
            return lines
            
        return np.array([])

    def visualize(self, frame, lines):
        """Draw detected lines on frame."""
        vis_frame = frame.copy()
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return vis_frame
