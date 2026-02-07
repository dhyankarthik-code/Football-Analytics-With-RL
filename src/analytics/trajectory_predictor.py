"""
Ball trajectory predictor using physics-based and ML models.
Predicts ball path for next N frames and visualizes on frame.
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from collections import deque
from src.utils.logger import get_logger

logger = get_logger("TrajectoryPredictor")


class TrajectoryPredictor:
    """
    Predicts ball trajectory using velocity + acceleration estimation.
    Draws predicted path on frame.
    """
    
    def __init__(
        self,
        prediction_horizon: int = 10,
        history_length: int = 5,
        gravity_factor: float = 0.5
    ):
        """
        Initialize trajectory predictor.
        
        Args:
            prediction_horizon: Number of frames to predict ahead
            history_length: Number of past positions to use
            gravity_factor: Gravity effect (for aerial balls)
        """
        self.prediction_horizon = prediction_horizon
        self.history_length = history_length
        self.gravity_factor = gravity_factor
        
        # Position history (deque for efficient pop/append)
        self.position_history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length)
    
    def update(self, position: np.ndarray):
        """
        Update with new ball position.
        
        Args:
            position: Ball center position [x, y]
        """
        # Calculate velocity if we have previous position
        if len(self.position_history) > 0:
            velocity = position - self.position_history[-1]
            self.velocity_history.append(velocity)
        
        self.position_history.append(position)
    
    def predict_trajectory(self) -> Optional[np.ndarray]:
        """
        Predict ball trajectory.
        
        Returns:
            Array of predicted positions [[x1, y1], [x2, y2], ...]
            or None if insufficient data
        """
        if len(self.position_history) < 2:
            return None
        
        # Estimate current velocity (average of recent velocities)
        if len(self.velocity_history) > 0:
            velocity = np.mean(self.velocity_history, axis=0)
        else:
            return None
        
        # Estimate acceleration (change in velocity)
        if len(self.velocity_history) >= 2:
            accel = self.velocity_history[-1] - self.velocity_history[-2]
        else:
            accel = np.zeros(2)
        
        # Predict future positions
        trajectory = []
        current_pos = self.position_history[-1].copy()
        current_vel = velocity.copy()
        
        for t in range(1, self.prediction_horizon + 1):
            # Update velocity with acceleration
            current_vel = current_vel + accel
            
            # Apply gravity (downward acceleration for aerial balls)
            # Assume y-axis points downward
            current_vel[1] += self.gravity_factor * t
            
            # Update position
            current_pos = current_pos + current_vel
            
            trajectory.append(current_pos.copy())
        
        return np.array(trajectory)
    
    def draw_trajectory(
        self,
        frame: np.ndarray,
        color=(0, 255, 255),
        thickness=2,
        show_confidence=True
    ):
        """
        Draw predicted trajectory on frame.
        
        Args:
            frame: Frame to draw on
            color: Trajectory line color
            thickness: Line thickness
            show_confidence: Fade color with distance
        
        Returns:
            Frame with trajectory drawn
        """
        trajectory = self.predict_trajectory()
        
        if trajectory is None or len(trajectory) == 0:
            return frame
        
        # Draw lines between predicted points
        for i in range(len(trajectory) - 1):
            pt1 = tuple(trajectory[i].astype(int))
            pt2 = tuple(trajectory[i + 1].astype(int))
            
            # Fade color based on distance in prediction
            if show_confidence:
                alpha = 1.0 - (i / len(trajectory))
                current_color = tuple(int(c * alpha) for c in color)
            else:
                current_color = color
            
            cv2.line(frame, pt1, pt2, current_color, thickness)
        
        # Draw circles at predicted positions
        for i, pos in enumerate(trajectory[::2]):  # Every other point
            alpha = 1.0 - (i / (len(trajectory) / 2))
            radius = max(3, int(5 * alpha))
            cv2.circle(frame, tuple(pos.astype(int)), radius, color, -1)
        
        return frame
    
    def draw_history(
        self,
        frame: np.ndarray,
        color=(255, 0, 255),
        thickness=2
    ):
        """
        Draw historical trajectory (where ball has been).
        
        Args:
            frame: Frame to draw on
            color: History line color
            thickness: Line thickness
        
        Returns:
            Frame with history drawn
        """
        if len(self.position_history) < 2:
            return frame
        
        positions = list(self.position_history)
        
        for i in range(1, len(positions)):
            pt1 = tuple(positions[i-1].astype(int))
            pt2 = tuple(positions[i].astype(int))
            cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame
    
    def reset(self):
        """Reset trajectory history."""
        self.position_history.clear()
        self.velocity_history.clear()
