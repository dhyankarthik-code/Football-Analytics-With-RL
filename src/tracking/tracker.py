"""
Object tracking module using ByteTrack (via Supervision).
"""
import supervision as sv
from supervision import Detections
from src.utils.logger import get_logger

logger = get_logger("Tracker")

class Tracker:
    def __init__(self, config):
        """
        Initialize the ByteTrack tracker.
        
        Args:
            config: Configuration object containing tracking parameters
        """
        self.config = config.tracking
        
        logger.info(f"Initializing {self.config.tracker_type} tracker")
        
        if self.config.tracker_type == 'bytetrack':
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=self.config.track_thresh,
                lost_track_buffer=self.config.track_buffer,
                minimum_matching_threshold=self.config.match_thresh,
                frame_rate=30 # Default, will be updated if video_io passes it
            )
        else:
            # Fallback or other trackers could be added here
            logger.warning(f"Unknown tracker type {self.config.tracker_type}, defaulting to ByteTrack")
            self.byte_tracker = sv.ByteTrack()

    def update(self, detections: Detections) -> Detections:
        """
        Update tracker with new detections.
        
        Args:
            detections: Supervision Detections object from detector
            
        Returns:
            Detections object with assigned tracker_id
        """
        # ByteTrack update
        tracked_detections = self.byte_tracker.update_with_detections(detections)
        
        # Log active tracks (optional debug)
        # if len(tracked_detections) > 0:
        #     logger.debug(f"Tracking {len(tracked_detections)} objects")
            
        return tracked_detections

    def reset(self):
        """Reset the tracker state."""
        self.byte_tracker.reset()
