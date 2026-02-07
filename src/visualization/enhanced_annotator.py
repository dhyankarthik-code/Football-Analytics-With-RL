"""
Enhanced annotator with team-colored bounding boxes and analytics overlays.
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple



class EnhancedAnnotator:
    """
    Advanced annotator with team colors, speed, trajectory, and jersey numbers.
    """
    
    # Team color mapping
    TEAM_COLORS = {
        0: (0, 0, 255),    # Team 1: Red
        1: (255, 0, 0),    # Team 2: Blue
        2: (255, 255, 0),  # Referee: Yellow
        99: (128, 128, 128)  # Unknown: Gray
    }
    
    BALL_COLOR = (0, 255, 0)  # Green
    
    def __init__(self):
        """Initialize enhanced annotator."""
        pass
    
    def draw_team_bbox(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        team_id: int,
        tracker_id: Optional[int] = None,
        thickness: int = 2
    ):
        """
        Draw bounding box with team color.
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            team_id: Team ID (0, 1, 2 for referee)
            tracker_id: Optional tracker ID to display
            thickness: Line thickness
        
        Returns:
            Frame with bbox drawn
        """
        color = self.TEAM_COLORS.get(team_id, self.TEAM_COLORS[99])
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw tracker ID if provided
        if tracker_id is not None:
            cv2.putText(
                frame,
                f"ID:{tracker_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return frame
    
    def draw_ball_bbox(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        confidence: Optional[float] = None,
        thickness: int = 3
    ):
        """
        Draw ball bounding box.
        
        Args:
            frame: Frame to draw on
            bbox: Ball bbox [x1, y1, x2, y2]
            confidence: Optional RL confidence score
            thickness: Line thickness
        
        Returns:
            Frame with ball bbox drawn
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.BALL_COLOR, thickness)
        
        if confidence is not None:
            cv2.putText(
                frame,
                f"Ball {confidence:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.BALL_COLOR,
                2
            )
        
        return frame
    
    def draw_speed(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        speed: float,
        color=(255, 255, 255)
    ):
        """
        Draw speed label.
        
        Args:
            frame: Frame to draw on
            bbox: Object bbox
            speed: Speed in km/h
            color: Text color
        
        Returns:
            Frame with speed drawn
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        text = f"{speed:.1f} km/h"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Draw below bbox
        cv2.putText(
            frame,
            text,
            (x1, y2 + 20),
            font,
            font_scale,
            color,
            thickness
        )
        
        return frame
    
    def draw_possession(
        self,
        frame: np.ndarray,
        player_bbox: np.ndarray,
        jersey_number: Optional[int],
        team_id: int
    ):
        """
        Highlight player with ball possession.
        
        Args:
            frame: Frame to draw on
            player_bbox: Player bbox
            jersey_number: Jersey number
            team_id: Team ID
        
        Returns:
            Frame with possession indicator
        """
        x1, y1, x2, y2 = map(int, player_bbox)
        team_color = self.TEAM_COLORS.get(team_id, self.TEAM_COLORS[99])
        
        # Draw thick border
        cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 5)
        cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 3)
        
        # Draw jersey number with background
        if jersey_number is not None:
            text = f"#{jersey_number} HAS BALL"
            font = cv2.FONT_HERSHEY_BOLD
            font_scale = 0.8
            thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Background
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 15),
                (x1 + text_w + 10, y1),
                (0, 0, 0),
                -1
            )
            
            # Text
            cv2.putText(
                frame,
                text,
                (x1 + 5, y1 - 8),
                font,
                font_scale,
                (0, 255, 0),
                thickness
            )
        
        return frame
    
    def draw_analytics_overlay(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        gpu_util: Optional[float] = None
    ):
        """
        Draw analytics overlay (frame info, FPS, GPU usage).
        
        Args:
            frame: Frame to draw on
            frame_idx: Current frame index
            fps: Processing FPS
            gpu_util: GPU utilization (0-100%)
        
        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        
        cv2.putText(frame, f"Frame: {frame_idx}", (20, y_offset), font, font_scale, color, 2)
        y_offset += 25
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), font, font_scale, color, 2)
        y_offset += 25
        
        if gpu_util is not None:
            cv2.putText(frame, f"GPU: {gpu_util:.0f}%", (20, y_offset), font, font_scale, color, 2)
        
        return frame
