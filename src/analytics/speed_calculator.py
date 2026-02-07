"""
Speed calculator for ball and players.
Calculates real-world speed in km/h from pixel movement.
"""
import numpy as np
from typing import Dict, Optional
from src.utils.logger import get_logger

logger = get_logger("SpeedCalculator")


class SpeedCalculator:
    """
    Calculates speed for ball and players in km/h.
    Uses FPS and pitch dimensions for real-world conversion.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        pitch_length_meters: float = 105.0,
        pitch_width_meters: float = 68.0,
        pitch_length_pixels: Optional[float] = None,
        pitch_width_pixels: Optional[float] = None
    ):
        """
        Initialize speed calculator.
        
        Args:
            fps: Video frames per second
            pitch_length_meters: Real pitch length (default FIFA standard)
            pitch_width_meters: Real pitch width
            pitch_length_pixels: Pitch length in pixels (auto-detect if None)
            pitch_width_pixels: Pitch width in pixels (auto-detect if None)
        """
        self.fps = fps
        self.pitch_length_m = pitch_length_meters
        self.pitch_width_m = pitch_width_meters
        self.pitch_length_px = pitch_length_pixels
        self.pitch_width_px = pitch_width_pixels
        
        # Calibration factor (meters per pixel)
        self.meters_per_pixel_x = None
        self.meters_per_pixel_y = None
        
        if pitch_length_pixels and pitch_width_pixels:
            self.calibrate(pitch_length_pixels, pitch_width_pixels)
        
        # Tracking history
        self.position_history = {}  # track_id -> [(frame, pos), ...]
    
    def calibrate(self, pitch_length_px: float, pitch_width_px: float):
        """
        Calibrate pixel-to-meter conversion.
        
        Args:
            pitch_length_px: Detected pitch length in pixels
            pitch_width_px: Detected pitch width in pixels
        """
        self.pitch_length_px = pitch_length_px
        self.pitch_width_px = pitch_width_px
        
        self.meters_per_pixel_x = self.pitch_length_m / pitch_length_px
        self.meters_per_pixel_y = self.pitch_width_m / pitch_width_px
        
        logger.info(f"Calibrated: {self.meters_per_pixel_x:.4f} m/px (x), {self.meters_per_pixel_y:.4f} m/px (y)")
    
    def auto_calibrate_from_frame(self, frame_width: int, frame_height: int):
        """
        Auto-calibrate assuming pitch fills ~80% of frame.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        # Rough estimate
        pitch_length_px = frame_width * 0.8
        pitch_width_px = frame_height * 0.6
        
        self.calibrate(pitch_length_px, pitch_width_px)
    
    def update_position(self, track_id: int, frame_idx: int, position: np.ndarray):
        """
        Update position history for tracking object.
        
        Args:
            track_id: Track ID
            frame_idx: Current frame index
            position: Position [x, y] in pixels
        """
        if track_id not in self.position_history:
            self.position_history[track_id] = []
        
        self.position_history[track_id].append((frame_idx, position))
        
        # Keep only last 10 positions (for smoothing)
        if len(self.position_history[track_id]) > 10:
            self.position_history[track_id].pop(0)
    
    def calculate_speed(
        self,
        track_id: int,
        current_position: np.ndarray,
        frame_idx: int,
        smoothing_window: int = 3
    ) -> Optional[float]:
        """
        Calculate speed in km/h.
        
        Args:
            track_id: Track ID
            current_position: Current position [x, y] in pixels
            frame_idx: Current frame index
            smoothing_window: Number of frames for smoothing
        
        Returns:
            Speed in km/h, or None if insufficient data
        """
        # Update position
        self.update_position(track_id, frame_idx, current_position)
        
        # Need calibration
        if self.meters_per_pixel_x is None:
            logger.warning("Speed calculator not calibrated")
            return None
        
        # Need at least 2 positions
        history = self.position_history.get(track_id, [])
        if len(history) < 2:
            return None
        
        # Get last N positions for smoothing
        recent = history[-smoothing_window:] if len(history) >= smoothing_window else history
        
        # Calculate average displacement
        total_distance_px = 0.0
        total_frames = 0
        
        for i in range(1, len(recent)):
            prev_frame, prev_pos = recent[i-1]
            curr_frame, curr_pos = recent[i]
            
            # Distance in pixels
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            distance_px = np.sqrt(dx**2 + dy**2)
            
            total_distance_px += distance_px
            total_frames += (curr_frame - prev_frame)
        
        if total_frames == 0:
            return None
        
        # Convert to meters
        # Use average of x and y calibration
        avg_meters_per_pixel = (self.meters_per_pixel_x + self.meters_per_pixel_y) / 2
        distance_meters = total_distance_px * avg_meters_per_pixel
        
        # Time in seconds
        time_seconds = total_frames / self.fps
        
        # Speed in m/s
        speed_ms = distance_meters / time_seconds if time_seconds > 0 else 0
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh
    
    def get_instantaneous_speed(
        self,
        position_current: np.ndarray,
        position_previous: np.ndarray
    ) -> float:
        """
        Calculate instantaneous speed between two frames.
        
        Args:
            position_current: Current position [x, y]
            position_previous: Previous position [x, y]
        
        Returns:
            Speed in km/h
        """
        if self.meters_per_pixel_x is None:
            return 0.0
        
        # Distance in pixels
        dx = position_current[0] - position_previous[0]
        dy = position_current[1] - position_previous[1]
        distance_px = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters
        avg_meters_per_pixel = (self.meters_per_pixel_x + self.meters_per_pixel_y) / 2
        distance_meters = distance_px * avg_meters_per_pixel
        
        # Time for 1 frame
        time_seconds = 1.0 / self.fps
        
        # Speed
        speed_ms = distance_meters / time_seconds
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh
    
    def reset(self):
        """Reset position history."""
        self.position_history.clear()
