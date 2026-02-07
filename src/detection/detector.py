"""
Object detector module using Ultralytics YOLOv8.
"""
from typing import List, Optional, Dict, Union
import numpy as np
import torch
from ultralytics import YOLO
from supervision import Detections

from src.utils.logger import get_logger
from src.utils.config import get_device

logger = get_logger("ObjectDetector")

class ObjectDetector:
    def __init__(self, config):
        """
        Initialize the YOLOv8 object detector.
        
        Args:
            config: Configuration object containing detection parameters
        """
        self.config = config.detection
        self.device = get_device(config)
        
        logger.info(f"Loading YOLO model: {self.config.model_path} on {self.device}")
        
        # Load model (will auto-download if not present)
        try:
            self.model = YOLO(self.config.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        # Warmup
        logger.info("Warming up model...")
        # Placeholder warmup - optional but good practice

    def detect(self, frame: np.ndarray) -> Detections:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            supervision.Detections: Detections object containing boxes, confidence, class_ids
        """
        # Inference
        results = self.model.predict(
            frame, 
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.classes,
            device=self.device,
            verbose=False,
            imgsz=self.config.img_size,
            half=(self.device != 'cpu') # Use FP16 if on GPU
        )
        
        # Convert to Supervision format for easier handling downstream
        result = results[0] # Single frame
        detections = Detections.from_ultralytics(result)
        
        # Log stats periodically (optional, maybe too noisy for every frame)
        # logger.debug(f"Detected {len(detections)} objects")
        
        return detections

    def get_classes(self) -> Dict[int, str]:
        """Return a mapping of class ID to class name."""
        return self.model.names
