"""
Iterative RL Agent - Refines bounding box until confidence threshold is met.
Implements multi-step refinement with early stopping based on confidence.
"""
import numpy as np
from typing import Optional, Dict, Tuple
from stable_baselines3 import PPO, SAC
from src.utils.logger import get_logger

logger = get_logger("IterativeRLAgent")


class IterativeRLAgent:
    """
    RL Agent that iteratively refines bounding boxes until confidence threshold is met.
    
    Features:
    - Multi-step refinement (up to max_iterations)
    - Early stopping when confidence > threshold
    - Tracks refinement history
    """
    
    def __init__(
        self,
        model_path: str,
        algorithm: str = "PPO",
        max_iterations: int = 5,
        confidence_threshold: float = 0.85,
        min_improvement: float = 0.01
    ):
        """
        Initialize iterative RL agent.
        
        Args:
            model_path: Path to trained model
            algorithm: Algorithm type ("PPO" or "SAC")
            max_iterations: Maximum refinement iterations per frame
            confidence_threshold: Stop when confidence exceeds this
            min_improvement: Stop if improvement < this threshold
        """
        self.model_path = model_path
        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.min_improvement = min_improvement
        
        # Load model
        if algorithm == "PPO":
            self.model = PPO.load(model_path)
        elif algorithm == "SAC":
            self.model = SAC.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        logger.info(f"Loaded {algorithm} model with iterative refinement (max_iter={max_iterations}, conf_thresh={confidence_threshold})")
    
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
    
    def estimate_confidence(
        self,
        ball_box: np.ndarray,
        observation: Dict[str, np.ndarray],
        prev_iou: Optional[float] = None
    ) -> float:
        """
        Estimate confidence in current bounding box.
        
        Uses multiple heuristics:
        1. Visual feature consistency
        2. Velocity smoothness
        3. Box size stability
        4. IoU improvement (if available)
        
        Args:
            ball_box: Current ball box
            observation: Current observation
            prev_iou: Previous IoU score (if available)
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_scores = []
        
        # 1. Visual feature magnitude (higher = more confident)
        visual_features = observation.get("visual_features", np.zeros(256))
        feature_magnitude = np.linalg.norm(visual_features)
        feature_confidence = min(feature_magnitude / 10.0, 1.0)  # Normalize
        confidence_scores.append(feature_confidence)
        
        # 2. Velocity smoothness (lower velocity = more stable = higher confidence)
        velocity = observation.get("ball_velocity", np.zeros(2))
        velocity_magnitude = np.linalg.norm(velocity)
        velocity_confidence = max(1.0 - velocity_magnitude * 5.0, 0.0)
        confidence_scores.append(velocity_confidence)
        
        # 3. Box size stability (compare to previous box)
        prev_box = observation.get("prev_ball_box", np.zeros(4))
        if np.any(prev_box):
            curr_area = (ball_box[2] - ball_box[0]) * (ball_box[3] - ball_box[1])
            prev_area = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
            area_ratio = min(curr_area, prev_area) / max(curr_area, prev_area)
            confidence_scores.append(area_ratio)
        
        # 4. IoU improvement (if ground truth available during training)
        if prev_iou is not None:
            # Higher IoU = higher confidence
            confidence_scores.append(prev_iou)
        
        # Aggregate confidence (weighted average)
        if confidence_scores:
            confidence = np.mean(confidence_scores)
        else:
            confidence = 0.5  # Default medium confidence
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def refine_ball_box_iterative(
        self,
        ball_box: np.ndarray,
        observation: Dict[str, np.ndarray],
        frame: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Iteratively refine ball bounding box until confidence threshold met.
        
        Args:
            ball_box: Initial ball box [x1, y1, x2, y2]
            observation: Environment observation
            frame: Optional frame for feature extraction
            verbose: Print iteration details
        
        Returns:
            (refined_box, refinement_info)
        """
        current_box = ball_box.copy()
        refinement_history = []
        
        for iteration in range(self.max_iterations):
            # Estimate confidence
            confidence = self.estimate_confidence(current_box, observation)
            
            # Store iteration info
            iter_info = {
                "iteration": iteration,
                "box": current_box.copy(),
                "confidence": confidence
            }
            
            if verbose:
                logger.info(f"Iteration {iteration}: Confidence = {confidence:.3f}")
            
            # Early stopping: confidence threshold met
            if confidence >= self.confidence_threshold:
                iter_info["stopped_reason"] = "confidence_threshold"
                refinement_history.append(iter_info)
                if verbose:
                    logger.info(f"✓ Confidence threshold met ({confidence:.3f} >= {self.confidence_threshold})")
                break
            
            # Predict refinement action
            action = self.predict(observation, deterministic=True)
            
            # Apply action to refine box
            box_w = current_box[2] - current_box[0]
            box_h = current_box[3] - current_box[1]
            
            prev_box = current_box.copy()
            current_box[0] += action[0] * box_w
            current_box[1] += action[1] * box_h
            current_box[2] += action[2] * box_w
            current_box[3] += action[3] * box_h
            
            # Calculate improvement
            improvement = np.linalg.norm(current_box - prev_box)
            iter_info["improvement"] = improvement
            iter_info["action"] = action
            
            refinement_history.append(iter_info)
            
            # Early stopping: minimal improvement
            if improvement < self.min_improvement:
                if verbose:
                    logger.info(f"✓ Minimal improvement ({improvement:.4f} < {self.min_improvement})")
                refinement_history[-1]["stopped_reason"] = "minimal_improvement"
                break
            
            # Update observation for next iteration
            observation["prev_ball_box"] = observation["ball_box"].copy()
            observation["ball_box"] = current_box / np.array([
                frame.shape[1] if frame is not None else 1280,
                frame.shape[0] if frame is not None else 720,
                frame.shape[1] if frame is not None else 1280,
                frame.shape[0] if frame is not None else 720
            ])
        
        # If max iterations reached
        if iteration == self.max_iterations - 1:
            refinement_history[-1]["stopped_reason"] = "max_iterations"
            if verbose:
                logger.info(f"⚠ Max iterations reached ({self.max_iterations})")
        
        # Refinement summary
        refinement_info = {
            "iterations": len(refinement_history),
            "final_confidence": confidence,
            "history": refinement_history,
            "total_improvement": np.linalg.norm(current_box - ball_box)
        }
        
        return current_box, refinement_info
    
    def refine_ball_box(
        self,
        ball_box: np.ndarray,
        observation: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Single-step refinement (backward compatibility).
        
        Args:
            ball_box: Current ball box [x1, y1, x2, y2]
            observation: Environment observation
        
        Returns:
            Refined ball box
        """
        refined_box, _ = self.refine_ball_box_iterative(ball_box, observation, verbose=False)
        return refined_box
