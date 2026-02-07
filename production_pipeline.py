"""
Production pipeline with all enhanced features.
Integrates RL, analytics, and visualization with GPU optimization.
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import time

from src.utils.config import load_config
from src.detection import ObjectDetector
from src.tracking import Tracker
from src.rl.iterative_agent import IterativeRLAgent
from src.analytics import (
    PitchBoundaryFilter,
    SpeedCalculator,
    TrajectoryPredictor,
    JerseyNumberDetector
)
from src.visualization.enhanced_annotator import EnhancedAnnotator
from src.utils.logger import get_logger

logger = get_logger("ProductionPipeline")


class ProductionPipeline:
    """
    Production-ready pipeline with all enhancements:
    - Team-colored bounding boxes
    - Sharp ball tracking (iterative RL)
    - Speed calculation
    - Trajectory prediction
    - Jersey number detection
    - GPU-optimized processing
    """
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        rl_model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize production pipeline.
        
        Args:
            config_path: Config file path
            rl_model_path: Path to trained RL model (optional)
            use_gpu: Force GPU usage
        """
        # GPU check
        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA not available! Falling back to CPU")
            self.use_gpu = False
            self.device = 'cpu'
        else:
            self.use_gpu = use_gpu
            self.device = 'cuda' if use_gpu else 'cpu'

        
        logger.info(f"Initializing pipeline on {self.device.upper()}")
        
        # Load config
        self.config = load_config(config_path)
        
        # Initialize components
        self.detector = ObjectDetector(self.config)
        self.tracker = Tracker(self.config)
        
        # Analytics
        self.pitch_filter = PitchBoundaryFilter(use_auto_detection=True)
        self.speed_calculator = None  # Initialized when FPS known
        self.trajectory_predictor = TrajectoryPredictor(
            prediction_horizon=10,
            history_length=5
        )
        self.jersey_detector = JerseyNumberDetector(use_gpu=use_gpu)
        
        # RL agent (if model provided)
        self.rl_agent = None
        if rl_model_path and Path(rl_model_path).exists():
            self.rl_agent = IterativeRLAgent(
                rl_model_path,
                algorithm="PPO",
                max_iterations=5,
                confidence_threshold=0.85
            )
            logger.info("RL agent loaded for iterative refinement")
        
        # Visualization
        self.annotator = EnhancedAnnotator()
        
        # State
        self.frame_idx = 0
        self.pitch_initialized = False
    
    def process_video(
        self,
        video_path: str,
        output_path: str,
        show_intermediate: bool = False
    ):
        """
        Process full video with all enhancements.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            show_intermediate: Show iterative RL steps
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing {total_frames} frames at {fps} FPS")
        
        # Initialize speed calculator with FPS
        self.speed_calculator = SpeedCalculator(fps=fps)
        self.speed_calculator.auto_calibrate_from_frame(width, height)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing stats
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Write output
            out.write(processed_frame)
            
            self.frame_idx += 1
            
            if self.frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                processing_fps = self.frame_idx / elapsed
                logger.info(f"Frame {self.frame_idx}/{total_frames} | FPS: {processing_fps:.1f}")
        
        cap.release()
        out.release()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = total_frames / total_time
        logger.info(f"Processing complete! Avg FPS: {avg_fps:.2f}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame with all enhancements.
        
        Args:
            frame: Input frame
        
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Initialize pitch filter on first frame
        if not self.pitch_initialized:
            self.pitch_filter.initialize_pitch_boundary(frame)
            self.pitch_initialized = True
        
        # 1. Detection
        detections = self.detector.detect(frame)
        
        # 2. Filter to pitch only (exclude ball boys, spectators)
        detections = self.pitch_filter.filter_detections(detections)
        
        # 3. Tracking
        tracked_detections = self.tracker.update(detections)
        
        # 4. Extract ball and players
        ball_mask = tracked_detections.class_id == 32
        player_mask = tracked_detections.class_id == 0
        
        ball_box = None
        ball_position = None
        rl_confidence = None
        
        # Process ball
        if np.any(ball_mask):
            ball_detection = tracked_detections[ball_mask][0]
            ball_box = ball_detection.xyxy[0].copy()
            
            # RL refinement (if available)
            if self.rl_agent:
                # Create observation (simplified)
                observation = self._create_observation(ball_box, frame)
                
                # Iterative refinement
                refined_box, info = self.rl_agent.refine_ball_box_iterative(
                    ball_box,
                    observation,
                    frame=frame,
                    verbose=False
                )
                
                ball_box = refined_box
                rl_confidence = info["final_confidence"]
            
            # Ball position for trajectory
            ball_position = np.array([
                (ball_box[0] + ball_box[2]) / 2,
                (ball_box[1] + ball_box[3]) / 2
            ])
            
            # Update trajectory
            self.trajectory_predictor.update(ball_position)
        
        # 5. Start annotating frame
        annotated = frame.copy()
        
        # Draw pitch boundary
        annotated = self.pitch_filter.draw_pitch_boundary(annotated)
        
        # Draw trajectory
        if ball_position is not None:
            annotated = self.trajectory_predictor.draw_history(annotated)
            annotated = self.trajectory_predictor.draw_trajectory(annotated)
        
        # 6. Process players
        player_with_ball = None
        jersey_number = None
        
        if np.any(player_mask) and ball_position is not None:
            player_detections = tracked_detections[player_mask]
            player_bboxes = player_detections.xyxy
            
            # Find player with ball
            player_idx, jersey_number = self.jersey_detector.find_player_with_ball(
                frame,
                player_bboxes,
                ball_position,
                proximity_threshold=100.0
            )
            
            if player_idx is not None:
                player_with_ball = player_bboxes[player_idx]
            
            # Draw players with team colors
            for i, bbox in enumerate(player_bboxes):
                # Get team ID (from your existing team assignment logic)
                # For now, use placeholder
                team_id = 0 if i % 2 == 0 else 1
                
                tracker_id = player_detections.tracker_id[i] if player_detections.tracker_id is not None else None
                
                # Draw bbox
                annotated = self.annotator.draw_team_bbox(
                    annotated,
                    bbox,
                    team_id,
                    tracker_id
                )
                
                # Calculate and draw speed
                if tracker_id is not None:
                    position = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    speed = self.speed_calculator.calculate_speed(
                        tracker_id,
                        position,
                        self.frame_idx
                    )
                    
                    if speed is not None:
                        annotated = self.annotator.draw_speed(annotated, bbox, speed)
        
        # 7. Draw ball
        if ball_box is not None:
            annotated = self.annotator.draw_ball_bbox(
                annotated,
                ball_box,
                confidence=rl_confidence
            )
            
            # Ball speed
            ball_speed = self.speed_calculator.calculate_speed(
                9999,  # Ball tracker ID
                ball_position,
                self.frame_idx
            )
            
            if ball_speed is not None:
                annotated = self.annotator.draw_speed(
                    annotated,
                    ball_box,
                    ball_speed,
                    color=(0, 255, 0)
                )
        
        # 8. Highlight player with ball
        if player_with_ball is not None:
            team_id = 0  # Placeholder
            annotated = self.annotator.draw_possession(
                annotated,
                player_with_ball,
                jersey_number,
                team_id
            )
        
        # 9. Draw analytics overlay
        processing_fps = 30.0  # Placeholder
        gpu_util = self._get_gpu_util()
        
        annotated = self.annotator.draw_analytics_overlay(
            annotated,
            self.frame_idx,
            processing_fps,
            gpu_util
        )
        
        return annotated
    
    def _create_observation(self, ball_box, frame):
        """Create RL observation (simplified)."""
        h, w = frame.shape[:2]
        
        ball_box_norm = ball_box / np.array([w, h, w, h])
        
        # Extract visual features (simplified)
        x1, y1, x2, y2 = map(int, ball_box)
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        
        if crop.size > 0:
            crop_resized = cv2.resize(crop, (32, 32))
            visual_features = crop_resized.flatten().astype(np.float32) / 255.0
            visual_features = visual_features[:256] if len(visual_features) > 256 else np.pad(visual_features, (0, 256 - len(visual_features)))
        else:
            visual_features = np.zeros(256, dtype=np.float32)
        
        return {
            "ball_box": ball_box_norm.astype(np.float32),
            "ball_velocity": np.zeros(2, dtype=np.float32),
            "visual_features": visual_features,
            "prev_ball_box": np.zeros(4, dtype=np.float32)
        }
    
    def _get_gpu_util(self) -> Optional[float]:
        """Get GPU utilization."""
        if not self.use_gpu:
            return None
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except:
            return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--rl_model", default=None, help="RL model path")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    
    args = parser.parse_args()
    
    pipeline = ProductionPipeline(
        rl_model_path=args.rl_model,
        use_gpu=not args.cpu
    )
    
    pipeline.process_video(args.video, args.output)
