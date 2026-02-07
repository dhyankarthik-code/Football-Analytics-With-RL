"""
Custom Gymnasium environment for football ball tracking with RL.
Wraps YOLO detector and ByteTrack tracker for ball position refinement.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from supervision import Detections

from src.detection import ObjectDetector
from src.tracking import Tracker
from src.utils.logger import get_logger
from src.utils.video_io import VideoReader
from .reward import RewardCalculator

logger = get_logger("FootballTrackingEnv")


class FootballTrackingEnv(gym.Env):
    """
    Custom Gymnasium environment for RL-based ball tracking enhancement.
    
    State: Ball bounding box + velocity + visual features
    Action: Continuous adjustments [Δx, Δy, Δw, Δh] to refine ball bbox
    Reward: IoU with ground truth + smoothness + track continuity
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config,
        video_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        max_steps: int = 500,
        render_mode: Optional[str] = None
    ):
        """
        Initialize environment.
        
        Args:
            config: Configuration object
            video_path: Path to video file for training
            ground_truth_path: Path to ground truth annotations (JSON/CSV)
            max_steps: Maximum steps per episode
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.config = config
        self.video_path = video_path
        self.ground_truth_path = ground_truth_path
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Initialize detector and tracker
        self.detector = ObjectDetector(config)
        self.tracker = Tracker(config)
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Video reader (initialized in reset)
        self.video_reader = None
        self.current_frame = None
        self.frame_idx = 0
        
        # Ground truth annotations (loaded from file)
        self.ground_truth = {}  # frame_idx -> ball_box
        if ground_truth_path:
            self._load_ground_truth(ground_truth_path)
        
        # State tracking
        self.ball_box = None  # Current ball bounding box
        self.prev_ball_box = None
        self.ball_track_id = None
        self.prev_ball_track_id = None
        self.step_count = 0
        
        # Define action space: continuous adjustments to bbox
        # [Δx, Δy, Δw, Δh] normalized between -0.1 and 0.1 (max 10% adjustment)
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(4,),
            dtype=np.float32
        )
        
        # Define observation space
        # State includes:
        # - Ball bbox (4): [x1, y1, x2, y2] normalized
        # - Ball velocity (2): [vx, vy]
        # - Visual features (256): CNN features from crop
        # - Previous bbox (4): for temporal context
        self.observation_space = spaces.Dict({
            "ball_box": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            "ball_velocity": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "visual_features": spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32),
            "prev_ball_box": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
        })
    
    def _load_ground_truth(self, path: str):
        """
        Load ground truth annotations from file.
        
        Expected format (JSON):
        {
            "0": {"ball": [x1, y1, x2, y2]},
            "1": {"ball": [x1, y1, x2, y2]},
            ...
        }
        """
        import json
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for frame_idx_str, annotations in data.items():
                    frame_idx = int(frame_idx_str)
                    if "ball" in annotations:
                        self.ground_truth[frame_idx] = np.array(annotations["ball"], dtype=np.float32)
            logger.info(f"Loaded ground truth for {len(self.ground_truth)} frames")
        except Exception as e:
            logger.warning(f"Could not load ground truth: {e}")
    
    def _normalize_box(self, box: np.ndarray, width: int, height: int) -> np.ndarray:
        """Normalize bounding box to [0, 1] range."""
        return np.array([
            box[0] / width,
            box[1] / height,
            box[2] / width,
            box[3] / height
        ], dtype=np.float32)
    
    def _denormalize_box(self, box: np.ndarray, width: int, height: int) -> np.ndarray:
        """Denormalize bounding box from [0, 1] to pixel coordinates."""
        return np.array([
            box[0] * width,
            box[1] * height,
            box[2] * width,
            box[3] * height
        ], dtype=np.float32)
    
    def _extract_visual_features(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Extract visual features from ball crop.
        
        For now, uses simple histogram features. Can be replaced with
        CNN features from a pre-trained backbone.
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        
        # Clip to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(256, dtype=np.float32)
        
        crop = frame[y1:y2, x1:x2]
        
        # Resize to fixed size
        crop_resized = cv2.resize(crop, (32, 32))
        
        # Flatten and normalize
        features = crop_resized.flatten().astype(np.float32) / 255.0
        
        # Pad/truncate to 256 dimensions
        if len(features) > 256:
            features = features[:256]
        elif len(features) < 256:
            features = np.pad(features, (0, 256 - len(features)))
        
        return features
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Construct observation from current state."""
        h, w = self.current_frame.shape[:2]
        
        # Normalize boxes
        ball_box_norm = self._normalize_box(self.ball_box, w, h)
        prev_box_norm = self._normalize_box(self.prev_ball_box, w, h) if self.prev_ball_box is not None else np.zeros(4, dtype=np.float32)
        
        # Calculate velocity
        velocity = np.zeros(2, dtype=np.float32)
        if self.prev_ball_box is not None:
            cx_curr = (self.ball_box[0] + self.ball_box[2]) / 2
            cy_curr = (self.ball_box[1] + self.ball_box[3]) / 2
            cx_prev = (self.prev_ball_box[0] + self.prev_ball_box[2]) / 2
            cy_prev = (self.prev_ball_box[1] + self.prev_ball_box[3]) / 2
            velocity = np.array([
                (cx_curr - cx_prev) / w,
                (cy_curr - cy_prev) / h
            ], dtype=np.float32)
        
        # Extract visual features
        visual_features = self._extract_visual_features(self.current_frame, self.ball_box)
        
        return {
            "ball_box": ball_box_norm,
            "ball_velocity": velocity,
            "visual_features": visual_features,
            "prev_ball_box": prev_box_norm,
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize video reader
        if self.video_path and self.video_reader is None:
            self.video_reader = VideoReader(self.video_path)
        
        # Reset state
        self.frame_idx = 0
        self.step_count = 0
        self.ball_box = None
        self.prev_ball_box = None
        self.ball_track_id = None
        self.prev_ball_track_id = None
        self.reward_calculator.reset()
        
        # Get first frame
        if self.video_reader:
            for idx, frame in self.video_reader:
                self.current_frame = frame
                self.frame_idx = idx
                break
        else:
            # Dummy frame for testing
            self.current_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Run initial detection to get ball position
        detections = self.detector.detect(self.current_frame)
        tracked_detections = self.tracker.update(detections)
        
        # Find ball (class_id == 32)
        ball_mask = tracked_detections.class_id == 32
        if np.any(ball_mask):
            ball_detection = tracked_detections[ball_mask][0]
            self.ball_box = ball_detection.xyxy[0].copy()
            if tracked_detections.tracker_id is not None:
                self.ball_track_id = tracked_detections.tracker_id[ball_mask][0]
        else:
            # No ball detected, use center of frame as fallback
            h, w = self.current_frame.shape[:2]
            self.ball_box = np.array([w//2 - 20, h//2 - 20, w//2 + 20, h//2 + 20], dtype=np.float32)
        
        observation = self._get_observation()
        info = {"frame_idx": self.frame_idx}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [Δx, Δy, Δw, Δh] adjustments to ball bbox
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Store previous state
        self.prev_ball_box = self.ball_box.copy()
        self.prev_ball_track_id = self.ball_track_id
        
        # Apply action to refine ball box
        h, w = self.current_frame.shape[:2]
        dx, dy, dw, dh = action
        
        # Apply adjustments (as percentage of current box size)
        box_w = self.ball_box[2] - self.ball_box[0]
        box_h = self.ball_box[3] - self.ball_box[1]
        
        self.ball_box[0] += dx * box_w
        self.ball_box[1] += dy * box_h
        self.ball_box[2] += dw * box_w
        self.ball_box[3] += dh * box_h
        
        # Clip to frame bounds
        self.ball_box = np.clip(self.ball_box, [0, 0, 0, 0], [w, h, w, h])
        
        # Get ground truth for current frame
        gt_box = self.ground_truth.get(self.frame_idx)
        
        # Calculate reward
        if gt_box is not None:
            reward, reward_components = self.reward_calculator.compute(
                pred_box=self.ball_box,
                gt_box=gt_box,
                prev_box=self.prev_ball_box,
                track_id_current=self.ball_track_id,
                track_id_previous=self.prev_ball_track_id
            )
        else:
            # No ground truth available, use heuristic reward
            reward = 0.0
            reward_components = {}
        
        # Get next frame
        terminated = False
        truncated = False
        
        if self.video_reader:
            try:
                self.frame_idx, self.current_frame = next(iter(self.video_reader))
                
                # Run detection on new frame
                detections = self.detector.detect(self.current_frame)
                tracked_detections = self.tracker.update(detections)
                
                # Update ball position from tracker
                ball_mask = tracked_detections.class_id == 32
                if np.any(ball_mask):
                    ball_detection = tracked_detections[ball_mask][0]
                    # Use tracker output as new "base" position
                    # (RL agent will refine this in next step)
                    self.ball_box = ball_detection.xyxy[0].copy()
                    if tracked_detections.tracker_id is not None:
                        self.ball_track_id = tracked_detections.tracker_id[ball_mask][0]
                
            except StopIteration:
                terminated = True
        
        # Check truncation
        if self.step_count >= self.max_steps:
            truncated = True
        
        observation = self._get_observation()
        info = {
            "frame_idx": self.frame_idx,
            "reward_components": reward_components,
            "ball_box": self.ball_box.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            frame_vis = self.current_frame.copy()
            
            # Draw current ball box
            x1, y1, x2, y2 = map(int, self.ball_box)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_vis, "Ball (RL)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw ground truth if available
            gt_box = self.ground_truth.get(self.frame_idx)
            if gt_box is not None:
                gx1, gy1, gx2, gy2 = map(int, gt_box)
                cv2.rectangle(frame_vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
                cv2.putText(frame_vis, "GT", (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.imshow("Football Tracking RL", frame_vis)
            cv2.waitKey(1)
        
        elif self.render_mode == "rgb_array":
            return self.current_frame
    
    def close(self):
        """Clean up resources."""
        if self.video_reader:
            self.video_reader = None
        cv2.destroyAllWindows()
