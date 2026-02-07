"""
Contact detector module based on spatial proximity.
"""
from typing import List, Tuple
import numpy as np
import supervision as sv
from supervision import Detections

from src.utils.logger import get_logger

logger = get_logger("ContactDetector")

class ContactDetector:
    def __init__(self, config):
        """
        Initialize Contact Detector.
        """
        self.config = config.events
        # Distance threshold in pixels (approx 1 meter depending on resolution)
        self.proximity_thresh = 50.0 

    def detect(self, detections: Detections) -> List[Tuple[int, int]]:
        """
        Detect pairs of players in close proximity (potential contact).
        
        Args:
            detections: Supervision Detections
            
        Returns:
            List of tuples (track_id_1, track_id_2) representing interacting pairs
        """
        if len(detections) < 2:
            return []
            
        contacts = []
        
        # Get centers
        centers = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        ids = detections.tracker_id
        
        if ids is None:
            return []
            
        # Brute force (N^2 but N is small ~22)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dist = np.linalg.norm(centers[i] - centers[j])
                
                if dist < self.proximity_thresh:
                    contacts.append((ids[i], ids[j]))
                    
        return contacts
