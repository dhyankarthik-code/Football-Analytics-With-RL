"""
Annotator module for video visualization using Supervision.
"""
from typing import List, Dict, Optional
import cv2
import numpy as np
import supervision as sv
from supervision import Detections

from src.utils.logger import get_logger

logger = get_logger("Annotator")

# BGR color tuples for OpenCV drawing
TEAM_COLORS_BGR = {
    0: (0, 0, 255),       # Red (BGR) - Team A
    1: (255, 100, 50),    # Blue/Cyan (BGR) - Team B
    "ref": (0, 255, 255), # Yellow (BGR) - Referee
    "ball": (0, 255, 0),  # Green (BGR) - Ball
    "default": (255, 255, 255)  # White fallback
}


class Annotator:
    def __init__(self, config):
        """
        Initialize Annotator.
        """
        self.config = config.visualization
        
        # Trace annotator for ball trail
        self.trace_annotator = sv.TraceAnnotator(
            thickness=3,
            trace_length=50,
            color=sv.Color(0, 255, 0)  # Green trace for ball
        )
        
        # Ball position history for custom trail rendering
        self.ball_positions = []

    def annotate(self, frame: np.ndarray, detections: Detections, team_ids: np.ndarray, 
                 player_numbers: Dict[int, int], event: Optional[str] = None) -> np.ndarray:
        """
        Draw visual elements on the frame with team-specific colors.
        """
        annotated_frame = frame.copy()
        
        if not self.config.draw_tracks:
            return annotated_frame

        # Process each detection individually for custom colors
        for i, (bbox, tracker_id, class_id, conf) in enumerate(zip(
            detections.xyxy, 
            detections.tracker_id if detections.tracker_id is not None else [None] * len(detections),
            detections.class_id,
            detections.confidence
        )):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Determine Color and Label based on detection type
            if class_id == 32:  # Ball (COCO class 32 = sports ball)
                color = TEAM_COLORS_BGR["ball"]
                label = "Ball"
                thickness = 3
                
                # Track ball position for smoother trail
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.ball_positions.append((cx, cy))
                if len(self.ball_positions) > 50:
                    self.ball_positions.pop(0)
                    
            elif class_id == 0:  # Person
                # Get team color based on team_ids
                if i < len(team_ids) and team_ids[i] is not None:
                    tid = int(team_ids[i])
                    color = TEAM_COLORS_BGR.get(tid, TEAM_COLORS_BGR["default"])
                else:
                    color = TEAM_COLORS_BGR["default"]
                
                # Build label
                p_num = player_numbers.get(tracker_id, None) if tracker_id else None
                if p_num:
                    label = f"#{p_num}"
                elif tracker_id is not None:
                    label = f"ID: {tracker_id}"
                else:
                    label = ""
                thickness = 2
            else:
                color = TEAM_COLORS_BGR["default"]
                label = f"ID: {tracker_id}" if tracker_id else ""
                thickness = 2

            # Draw bounding box with team color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background and text
            if label and self.config.draw_ids:
                # Calculate text size
                font_scale = 0.6
                font_thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w + 10, y1),
                    color,
                    -1  # Filled
                )
                
                # Draw label text (white on colored background)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness
                )

        # Draw ball trail
        if len(self.ball_positions) > 1:
            for j in range(1, len(self.ball_positions)):
                # Fade effect: older positions are more transparent
                alpha = j / len(self.ball_positions)
                thickness = int(2 + 3 * alpha)  # Thicker towards current position
                color_intensity = int(255 * alpha)
                trail_color = (0, color_intensity, 0)  # Green fading
                
                cv2.line(
                    annotated_frame,
                    self.ball_positions[j - 1],
                    self.ball_positions[j],
                    trail_color,
                    thickness
                )
            
        # Draw Event Alert
        if event:
            cv2.putText(
                annotated_frame, 
                f"EVENT: {event}", 
                (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (0, 0, 255), 
                3
            )
            
        return annotated_frame
