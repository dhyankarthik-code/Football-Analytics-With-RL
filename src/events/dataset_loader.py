"""
Dataset loader for video event detection training.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("DatasetLoader")

class VideoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, sequence_length=16, img_size=224):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
                               Expected columns: [video_path, label, start_frame, end_frame]
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied on a sample.
            sequence_length (int): Number of frames to extract.
            img_size (int): Height/Width to resize frames.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.img_size = img_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        label = int(self.annotations.iloc[idx, 1])
        start_frame = int(self.annotations.iloc[idx, 2])
        
        frames = self._load_video_clip(video_name, start_frame)
        
        # Format for model: (C, T, H, W)
        # Frames is currently (T, H, W, C)
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)
        
        # Normalize to 0-1 if not already or use processor
        frames = frames / 255.0
        
        return frames, label

    def _load_video_clip(self, video_path, start_frame):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        buffer = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                # Pad with last frame or zeros if end of video
                if buffer:
                    frame = buffer[-1]
                else:
                    frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            buffer.append(frame)
            
        cap.release()
        return np.array(buffer)
