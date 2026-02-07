"""
Pitch boundary filter - Excludes detections outside playing field.
Filters out ball boys, coaches, spectators, and bench players.
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from src.utils.logger import get_logger


logger = get_logger("PitchFilter")


class PitchBoundaryFilter:
    """
    Filters player detections to only those on the pitch.
    Uses pitch line detection or manual boundary definition.
    """
    
    def __init__(
        self,
        pitch_keypoints: Optional[np.ndarray] = None,
        use_auto_detection: bool = True
    ):
        """
        Initialize pitch boundary filter.
        
        Args:
            pitch_keypoints: Manual pitch boundary points [[x1,y1], [x2,y2], ...]
            use_auto_detection: Auto-detect pitch from green color
        """
        self.pitch_keypoints = pitch_keypoints
        self.use_auto_detection = use_auto_detection
        self.pitch_mask = None
        self.pitch_polygon = None
    
    def detect_pitch_boundary(self, frame: np.ndarray) -> np.ndarray:
        """
        Auto-detect pitch boundaries from green field.
        
        Args:
            frame: Input frame
        
        Returns:
            Polygon points defining pitch boundary
        """
        # Convert to HSV for green detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green color range for grass
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No pitch boundary detected, using full frame")
            h, w = frame.shape[:2]
            return np.array([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Get largest contour (pitch)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return polygon.reshape(-1, 2)
    
    def initialize_pitch_boundary(self, frame: np.ndarray):
        """
        Initialize pitch boundary on first frame.
        
        Args:
            frame: First frame of video
        """
        if self.pitch_keypoints is not None:
            # Use manual keypoints
            self.pitch_polygon = self.pitch_keypoints
        elif self.use_auto_detection:
            # Auto-detect from green field
            self.pitch_polygon = self.detect_pitch_boundary(frame)
        else:
            # Default: use full frame
            h, w = frame.shape[:2]
            self.pitch_polygon = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Create mask from polygon
        h, w = frame.shape[:2]
        self.pitch_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self.pitch_mask, [self.pitch_polygon.astype(np.int32)], 255)
        
        logger.info(f"Pitch boundary initialized with {len(self.pitch_polygon)} points")
    
    def is_on_pitch(self, bbox: np.ndarray) -> bool:
        """
        Check if bounding box is on the pitch.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            True if bbox center is on pitch
        """
        if self.pitch_mask is None:
            return True  # If not initialized, accept all
        
        # Get bbox center
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        
        # Check if center is within pitch mask
        h, w = self.pitch_mask.shape
        if 0 <= cx < w and 0 <= cy < h:
            return self.pitch_mask[cy, cx] > 0
        
        return False
    
    def filter_detections(self, detections):
        """
        Filter detections to only those on pitch.
        
        Args:
            detections: Detections object or numpy array of bboxes
        
        Returns:
            Filtered detections (only on-pitch)
        """
        if detections is None:
            return detections
        
        # Handle supervision Detections object
        if hasattr(detections, 'xyxy'):
            if len(detections) == 0:
                return detections
            
            if self.pitch_mask is None:
                logger.warning("Pitch boundary not initialized, returning all detections")
                return detections
            
            # Check each detection
            on_pitch_mask = np.array([self.is_on_pitch(bbox) for bbox in detections.xyxy])
            
            # Filter detections
            filtered = detections[on_pitch_mask]
            
            logger.debug(f"Filtered {len(detections)} → {len(filtered)} detections (on-pitch only)")
            
            return filtered
        
        # Handle numpy array
        else:
            return detections  # Pass through if not Detections object

    
    def draw_pitch_boundary(self, frame: np.ndarray, color=(0, 255, 0), thickness=2):
        """
        Draw pitch boundary on frame for visualization.
        
        Args:
            frame: Frame to draw on
            color: Line color
            thickness: Line thickness
        
        Returns:
            Frame with pitch boundary drawn
        """
        if self.pitch_polygon is not None:
            cv2.polylines(
                frame,
                [self.pitch_polygon.astype(np.int32)],
                isClosed=True,
                color=color,
                thickness=thickness
            )
        
        return frame
