"""
Video I/O utility for efficient video reading and writing.
"""
import cv2
import time
from typing import Generator, Tuple, Optional
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger("VideoIO")

class VideoReader:
    def __init__(self, source_path: str, stride: int = 1):
        """
        Initialize VideoReader.
        
        Args:
            source_path: Path to input video or rtsp stream
            stride: Frame processing stride (process every Nth frame)
        """
        self.source_path = source_path
        self.stride = stride
        self.cap = cv2.VideoCapture(source_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Opened video: {source_path} ({self.width}x{self.height} @ {self.fps:.2f} fps)")

    def __iter__(self) -> Generator[Tuple[int, 'np.ndarray'], None, None]:
        """Yields (frame_idx, frame) tuples."""
        frame_idx = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                if frame_idx % self.stride == 0:
                    yield frame_idx, frame
                
                frame_idx += 1
        finally:
            self.cap.release()
            
    def get_info(self):
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames
        }

class VideoWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: float):
        """
        Initialize VideoWriter.
        
        Args:
            output_path: Output file path (e.g., output.mp4)
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.output_path = output_path
        
        # Try MP4V codec which is widely supported
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Initialized video writer: {output_path}")

    def write(self, frame):
        """Write a frame to the video file."""
        self.writer.write(frame)

    def release(self):
        """Release the video writer."""
        self.writer.release()
        logger.info(f"Saved video to: {self.output_path}")
