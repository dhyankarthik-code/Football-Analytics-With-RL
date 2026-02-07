"""
Team classification module using color clustering.
"""
from typing import List, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans
from supervision import Detections

from src.utils.logger import get_logger

logger = get_logger("TeamClassifier")

class TeamClassifier:
    def __init__(self, config):
        """
        Initialize Team Classifier.
        Args:
            config: Configuration object
        """
        self.config = config.identification
        self.kmeans = None
        self.team_colors = {} # id -> color
        
    def fit(self, frame: np.ndarray, detections: Detections):
        """
        Train K-Means on the first batch of detections to define team colors.
        Assumes the first frame has a good mix of both teams.
        """
        player_crops = []
        
        for xyxy in detections.xyxy:
            crop = self._get_jersey_crop(frame, xyxy)
            if crop.size > 0:
                # Get dominant color of the crop
                avg_color = crop.mean(axis=0).mean(axis=0)
                player_crops.append(avg_color)
        
        if len(player_crops) < 2:
            logger.warning("Not enough players to cluster teams! Skipping fit.")
            return

        # Cluster into 2 teams
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.kmeans.fit(np.array(player_crops))
        logger.info(f"Team Classifier fitted. centroids: {self.kmeans.cluster_centers_}")

    def predict(self, frame: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Assign team IDs to detections.
        Returns:
            np.array of shape (N,) with team IDs (0 or 1)
        """
        if self.kmeans is None:
            self.fit(frame, detections)
            if self.kmeans is None:
                return np.zeros(len(detections)) # Fallback

        team_ids = []
        for xyxy in detections.xyxy:
            crop = self._get_jersey_crop(frame, xyxy)
            avg_color = crop.mean(axis=0).mean(axis=0).reshape(1, -1)
            team_id = self.kmeans.predict(avg_color)[0]
            team_ids.append(team_id)
            
        return np.array(team_ids)

    def _get_jersey_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract upper body crop to avoid shorts/socks colors."""
        x1, y1, x2, y2 = map(int, bbox)
        # Clip to frame bounds
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        
        # Take top 50% for jersey
        return crop[:crop.shape[0]//2, :]
