"""
Reward calculator for RL-based football analytics.
Implements composite reward function with IoU, smoothness, and track continuity.
"""
import numpy as np
from typing import Dict, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger("RewardCalculator")


def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]
    
    Returns:
        IoU score (0.0 to 1.0)
    """
    # Intersection coordinates
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    
    # Intersection area
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    
    # Union area
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


class RewardCalculator:
    """
    Composite reward function for football tracking RL.
    """
    
    def __init__(
        self,
        w_iou: float = 1.0,
        w_diff_iou: float = 0.5,
        w_smooth: float = 0.2,
        w_track: float = 0.5,
        step_penalty: float = 0.01,
        alpha_smooth: float = 0.1
    ):
        """
        Initialize reward calculator.
        
        Args:
            w_iou: Weight for IoU reward
            w_diff_iou: Weight for differential IoU (improvement)
            w_smooth: Weight for smoothness penalty
            w_track: Weight for track continuity
            step_penalty: Per-step cost to encourage efficiency
            alpha_smooth: Smoothness penalty coefficient
        """
        self.w_iou = w_iou
        self.w_diff_iou = w_diff_iou
        self.w_smooth = w_smooth
        self.w_track = w_track
        self.step_penalty = step_penalty
        self.alpha_smooth = alpha_smooth
        
        # Track previous IoU for differential reward
        self.prev_iou = 0.0
    
    def reward_iou(self, pred_box: np.ndarray, gt_box: np.ndarray) -> float:
        """
        IoU-based reward with threshold bonuses.
        
        Args:
            pred_box: Predicted bounding box [x1, y1, x2, y2]
            gt_box: Ground truth bounding box [x1, y1, x2, y2]
        
        Returns:
            Reward value
        """
        iou = calculate_iou(pred_box, gt_box)
        
        # Threshold-based reward
        if iou > 0.7:
            return 1.0
        elif iou > 0.5:
            return 0.5
        elif iou < 0.3:
            return -1.0
        else:
            return 0.0
    
    def reward_differential_iou(
        self, 
        pred_box: np.ndarray, 
        gt_box: np.ndarray
    ) -> float:
        """
        Reward based on IoU improvement from previous step.
        
        Args:
            pred_box: Current predicted box
            gt_box: Ground truth box
        
        Returns:
            Differential IoU reward
        """
        current_iou = calculate_iou(pred_box, gt_box)
        diff = current_iou - self.prev_iou
        self.prev_iou = current_iou
        return diff
    
    def reward_smoothness(
        self, 
        box_current: np.ndarray, 
        box_previous: np.ndarray
    ) -> float:
        """
        Penalize jerky bounding box movements.
        
        Args:
            box_current: Current box [x1, y1, x2, y2]
            box_previous: Previous box [x1, y1, x2, y2]
        
        Returns:
            Smoothness penalty (negative value)
        """
        diff = np.linalg.norm(box_current - box_previous)
        return -self.alpha_smooth * diff
    
    def reward_track_continuity(
        self, 
        track_id_current: Optional[int], 
        track_id_previous: Optional[int]
    ) -> float:
        """
        Bonus for maintaining consistent track ID.
        
        Args:
            track_id_current: Current track ID
            track_id_previous: Previous track ID
        
        Returns:
            Continuity reward
        """
        if track_id_current is None or track_id_previous is None:
            return 0.0
        
        if track_id_current == track_id_previous:
            return 0.5  # Continuity maintained
        else:
            return -1.0  # ID switch penalty
    
    def compute(
        self,
        pred_box: np.ndarray,
        gt_box: np.ndarray,
        prev_box: Optional[np.ndarray] = None,
        track_id_current: Optional[int] = None,
        track_id_previous: Optional[int] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute composite reward.
        
        Args:
            pred_box: Predicted bounding box
            gt_box: Ground truth bounding box
            prev_box: Previous bounding box (for smoothness)
            track_id_current: Current track ID
            track_id_previous: Previous track ID
        
        Returns:
            (total_reward, reward_components_dict)
        """
        # IoU reward
        r_iou = self.reward_iou(pred_box, gt_box)
        
        # Differential IoU
        r_diff_iou = self.reward_differential_iou(pred_box, gt_box)
        
        # Smoothness (if previous box available)
        r_smooth = 0.0
        if prev_box is not None:
            r_smooth = self.reward_smoothness(pred_box, prev_box)
        
        # Track continuity
        r_track = 0.0
        if track_id_current is not None and track_id_previous is not None:
            r_track = self.reward_track_continuity(track_id_current, track_id_previous)
        
        # Composite reward
        total_reward = (
            self.w_iou * r_iou +
            self.w_diff_iou * r_diff_iou +
            self.w_smooth * r_smooth +
            self.w_track * r_track -
            self.step_penalty
        )
        
        # Component breakdown for logging
        components = {
            "iou": r_iou,
            "diff_iou": r_diff_iou,
            "smooth": r_smooth,
            "track": r_track,
            "step_penalty": -self.step_penalty,
            "total": total_reward
        }
        
        return total_reward, components
    
    def reset(self):
        """Reset internal state (e.g., previous IoU)."""
        self.prev_iou = 0.0
