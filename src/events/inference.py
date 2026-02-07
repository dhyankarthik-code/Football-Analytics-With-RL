"""
Real-time inference engine for event detection using TimeSformer.
"""
from collections import deque
from typing import Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from src.utils.logger import get_logger
from src.utils.config import get_device
from .models import EventDetectorModel

logger = get_logger("EventInference")

class EventInference:
    def __init__(self, config):
        """
        Initialize Inference Engine.
        """
        self.config = config.events
        self.model = None
        
        if not self.config.enabled or not self.config.use_ml_pipeline:
            return

        self.device = get_device(config)
        self.sequence_length = 8  # TimeSformer pretrained expects 8 frames
        self.buffer = deque(maxlen=self.sequence_length)
        self.img_size = 224 # Standard for TimeSformer
        
        self.classes = {0: "No-Foul", 1: "Foul", 2: "Penalty"}
        
        # Load Model
        try:
            self.model = EventDetectorModel(num_classes=len(self.classes), pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load Event Model: {e}")
            self.model = None

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Add frame to buffer and run inference if buffer is full.
        Returns detected event label or None.
        """
        if self.model is None or not self.config.enabled:
            return None
            
        # Preprocess
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.buffer.append(rgb)
        
        if len(self.buffer) < self.sequence_length:
            return None
            
        # Inference every N frames to save compute (or sliding window stride)
        # For now, run every frame (stride 1) if hardware allows, else strict stride
        
        # Prepare Tensor: HuggingFace TimeSformer expects (B, T, C, H, W)
        clip = np.array(self.buffer) # (T, H, W, C)
        tensor = torch.FloatTensor(clip).permute(0, 3, 1, 2) # (T, C, H, W)
        tensor = tensor.unsqueeze(0).to(self.device) # (1, T, C, H, W)
        
        # Normalize
        tensor = tensor / 255.0
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            
            # Get max prob
            conf, cls_idx = torch.max(probs, 1)
            conf = conf.item()
            cls_idx = cls_idx.item()
            
            if conf > self.config.threshold and cls_idx != 0: # 0 is No-Foul
                return self.classes[cls_idx]
                
        return None
