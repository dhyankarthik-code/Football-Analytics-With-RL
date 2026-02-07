"""
Event logging module.
"""
import pandas as pd
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger("EventLogger")

class EventLogger:
    def __init__(self, config):
        self.config = config
        self.events = []
        
    def log(self, frame_idx: int, event_type: str, details: dict):
        """
        Log an event.
        
        Args:
            frame_idx: Frame number
            event_type: 'foul', 'goal', 'card', etc.
            details: Dictionary with extra info (involved_players, location)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "frame": frame_idx,
            "type": event_type,
            **details
        }
        self.events.append(event)
        logger.info(f"EVENT DETECTED [{frame_idx}]: {event_type} - {details}")
        
    def save(self, output_path: str):
        """Save events to CSV."""
        if not self.events:
            logger.info("No events to save.")
            return
            
        df = pd.DataFrame(self.events)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(self.events)} events to {output_path}")
