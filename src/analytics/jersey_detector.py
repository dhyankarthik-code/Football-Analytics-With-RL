"""
Jersey number detector using EasyOCR.
Identifies player jersey numbers for ball possession tracking.
"""
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not available. Jersey number detection disabled.")

import numpy as np
import cv2
from typing import Optional, Tuple
from src.utils.logger import get_logger


logger = get_logger("JerseyDetector")


class JerseyNumberDetector:
    """
    Detects jersey numbers using OCR on player crops.
    Identifies which player has ball possession.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize jersey number detector.
        
        Args:
            use_gpu: Use GPU for OCR (faster)
        """
        self.use_gpu = use_gpu
        self.reader = None
        self._initialize_reader()
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader."""
        if not EASYOCR_AVAILABLE:
            logger.warning("EasyOCR not installed. Jersey detection disabled.")
            self.reader = None
            return
        
        try:
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
            logger.info(f"EasyOCR initialized (GPU={'ON' if self.use_gpu else 'OFF'})")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None

    
    def detect_number(
        self,
        player_crop: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Optional[int]:
        """
        Detect jersey number from player crop.
        
        Args:
            player_crop: Cropped image of player
            confidence_threshold: Minimum confidence for OCR
        
        Returns:
            Jersey number (int) or None
        """
        if self.reader is None:
            return None
        
        if player_crop.size == 0:
            return None
        
        try:
            # Run OCR
            results = self.reader.readtext(player_crop)
            
            # Filter for numbers only
            for (bbox, text, conf) in results:
                # Check if text is numeric
                cleaned = ''.join(filter(str.isdigit, text))
                
                if cleaned and conf >= confidence_threshold:
                    number = int(cleaned)
                    # Valid jersey numbers: 1-99
                    if 1 <= number <= 99:
                        logger.debug(f"Detected jersey number: {number} (conf={conf:.2f})")
                        return number
            
            return None
        
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None
    
    def find_player_with_ball(
        self,
        frame: np.ndarray,
        player_bboxes: np.ndarray,
        ball_position: np.ndarray,
        proximity_threshold: float = 100.0
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find which player has the ball and detect their number.
        
        Args:
            frame: Full frame
            player_bboxes: Player bounding boxes [[x1,y1,x2,y2], ...]
            ball_position: Ball center position [x, y]
            proximity_threshold: Max distance for possession (pixels)
        
        Returns:
            (player_index, jersey_number) or (None, None)
        """
        if len(player_bboxes) == 0:
            return None, None
        
        # Calculate distances from ball to each player
        player_centers = []
        for bbox in player_bboxes:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            player_centers.append([cx, cy])
        
        player_centers = np.array(player_centers)
        
        # Distance to ball
        distances = np.linalg.norm(player_centers - ball_position, axis=1)
        
        # Find closest player
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] > proximity_threshold:
            # No player close enough
            return None, None
        
        # Extract player crop
        bbox = player_bboxes[closest_idx]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Clip to frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return closest_idx, None
        
        player_crop = frame[y1:y2, x1:x2]
        
        # Detect jersey number
        jersey_number = self.detect_number(player_crop)
        
        return closest_idx, jersey_number
    
    def draw_jersey_number(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        jersey_number: Optional[int],
        color=(0, 255, 0)
    ):
        """
        Draw jersey number on frame.
        
        Args:
            frame: Frame to draw on
            bbox: Player bounding box
            jersey_number: Jersey number to display
            color: Text color
        
        Returns:
            Frame with jersey number drawn
        """
        if jersey_number is None:
            return frame
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw number above bounding box
        text = f"#{jersey_number}"
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 1.0
        thickness = 2
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background rectangle
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            (0, 0, 0),
            -1
        )
        
        # Text
        cv2.putText(
            frame,
            text,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            color,
            thickness
        )
        
        return frame
