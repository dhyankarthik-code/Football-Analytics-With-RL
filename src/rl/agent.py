"""
RL Agent wrapper for trained models.
Provides interface for inference with trained PPO/SAC agents.
"""
import numpy as np
from typing import Optional, Dict
from stable_baselines3 import PPO, SAC
from src.utils.logger import get_logger

logger = get_logger("RLAgent")


class RLAgent:
    """
    Wrapper for trained RL agents.
    """
    
    def __init__(
        self,
        model_path: str,
        algorithm: str = "PPO"
    ):
        """
        Initialize RL agent.
        
        Args:
            model_path: Path to trained model
            algorithm: Algorithm type ("PPO" or "SAC")
        """
        self.model_path = model_path
        self.algorithm = algorithm
        
        # Load model
        if algorithm == "PPO":
            self.model = PPO.load(model_path)
        elif algorithm == "SAC":
            self.model = SAC.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        logger.info(f"Loaded {algorithm} model from {model_path}")
    
    def predict(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Predict action for given observation.
        
        Args:
            observation: Environment observation
            deterministic: Use deterministic policy (no exploration)
        
        Returns:
            Action array
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def refine_ball_box(
        self,
        ball_box: np.ndarray,
        observation: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Refine ball bounding box using RL agent.
        
        Args:
            ball_box: Current ball box [x1, y1, x2, y2]
            observation: Environment observation
        
        Returns:
            Refined ball box
        """
        action = self.predict(observation, deterministic=True)
        
        # Apply action adjustments
        dx, dy, dw, dh = action
        box_w = ball_box[2] - ball_box[0]
        box_h = ball_box[3] - ball_box[1]
        
        refined_box = ball_box.copy()
        refined_box[0] += dx * box_w
        refined_box[1] += dy * box_h
        refined_box[2] += dw * box_w
        refined_box[3] += dh * box_h
        
        return refined_box
