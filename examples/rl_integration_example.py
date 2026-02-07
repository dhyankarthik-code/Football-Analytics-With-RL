"""
Example: Integrating RL agent with existing pipeline.
Shows how to use trained RL agent for ball tracking refinement.
"""
import cv2
import numpy as np
from src.utils.config import load_config
from src.detection import ObjectDetector
from src.tracking import Tracker
from src.rl.agent import RLAgent
from src.rl.environment import FootballTrackingEnv
from src.utils.logger import get_logger

logger = get_logger("RLIntegrationExample")


def process_video_with_rl(
    video_path: str,
    model_path: str,
    config_path: str = "configs/config.yaml",
    output_path: str = "output/rl_enhanced.mp4"
):
    """
    Process video with RL-enhanced ball tracking.
    
    Args:
        video_path: Input video path
        model_path: Path to trained RL model
        config_path: Config file path
        output_path: Output video path
    """
    # Load config
    config = load_config(config_path)
    
    # Initialize components
    detector = ObjectDetector(config)
    tracker = Tracker(config)
    rl_agent = RLAgent(model_path, algorithm="PPO")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # State tracking
    prev_ball_box = None
    
    logger.info(f"Processing video with RL enhancement...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Standard detection and tracking
        detections = detector.detect(frame)
        tracked_detections = tracker.update(detections)
        
        # Find ball
        ball_mask = tracked_detections.class_id == 32
        if np.any(ball_mask):
            ball_detection = tracked_detections[ball_mask][0]
            ball_box = ball_detection.xyxy[0].copy()
            
            # Create observation for RL agent
            # (simplified - in practice, use environment's _get_observation)
            ball_box_norm = ball_box / np.array([width, height, width, height])
            
            velocity = np.zeros(2, dtype=np.float32)
            if prev_ball_box is not None:
                cx_curr = (ball_box[0] + ball_box[2]) / 2
                cy_curr = (ball_box[1] + ball_box[3]) / 2
                cx_prev = (prev_ball_box[0] + prev_ball_box[2]) / 2
                cy_prev = (prev_ball_box[1] + prev_ball_box[3]) / 2
                velocity = np.array([
                    (cx_curr - cx_prev) / width,
                    (cy_curr - cy_prev) / height
                ], dtype=np.float32)
            
            # Extract visual features (simplified)
            x1, y1, x2, y2 = map(int, ball_box)
            crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (32, 32))
                visual_features = crop_resized.flatten().astype(np.float32) / 255.0
                visual_features = visual_features[:256] if len(visual_features) > 256 else np.pad(visual_features, (0, 256 - len(visual_features)))
            else:
                visual_features = np.zeros(256, dtype=np.float32)
            
            prev_box_norm = prev_ball_box / np.array([width, height, width, height]) if prev_ball_box is not None else np.zeros(4, dtype=np.float32)
            
            observation = {
                "ball_box": ball_box_norm.astype(np.float32),
                "ball_velocity": velocity,
                "visual_features": visual_features,
                "prev_ball_box": prev_box_norm.astype(np.float32)
            }
            
            # RL refinement
            refined_box = rl_agent.refine_ball_box(ball_box, observation)
            
            # Draw both boxes for comparison
            # Original (red)
            x1, y1, x2, y2 = map(int, ball_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Original", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # RL-refined (green)
            rx1, ry1, rx2, ry2 = map(int, refined_box)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
            cv2.putText(frame, "RL-Enhanced", (rx1, ry1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            prev_ball_box = refined_box
        
        # Write frame
        out.write(frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info(f"Processed {frame_idx} frames")
    
    cap.release()
    out.release()
    
    logger.info(f"RL-enhanced video saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    process_video_with_rl(
        video_path="input/sample_match.mp4",
        model_path="models/rl/ppo_ball_tracking_final.zip",
        output_path="output/rl_enhanced_result.mp4"
    )
