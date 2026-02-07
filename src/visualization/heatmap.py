"""
Heatmap visualization module.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

class HeatmapGenerator:
    def __init__(self, config):
        self.config = config.visualization
        self.pitch_dim = (1050, 680) # 10px per meter
        self.accumulated_heatmap = np.zeros((680, 1050), dtype=np.float32)

    def update(self, positions: np.ndarray):
        """
        Add positions to heatmap.
        Args:
           positions: (N, 2) array of coordinates in METERS
        """
        # Scale meters to pixels
        # x (0-105) -> 0-1050
        # y (0-68) -> 0-680
        pass
        
    def generate_overlay(self):
        """Return heatmap as image."""
        pass
