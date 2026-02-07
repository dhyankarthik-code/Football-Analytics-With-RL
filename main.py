"""
Football Analytics Pro - Main Pipeline
Orchestrates detection, tracking, identification, event detection, and visualization.
"""
import argparse
from tqdm import tqdm
import torch

from src.utils.logger import get_logger
from src.utils.config import load_config, get_device
from src.utils.video_io import VideoReader, VideoWriter

# Module Imports
from src.detection import ObjectDetector
from src.tracking import Tracker
from src.identification import TeamClassifier, OCRReader
from src.calibration import PitchDetector, HomographyEstimator
from src.events import EventInference, ContactDetector, EventLogger
from src.visualization import Annotator, StatsEngine

logger = get_logger("Pipeline")

def main():
    parser = argparse.ArgumentParser(description="Football Analytics Pro Pipeline")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--source', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output/result.mp4', help='Path to output video')
    args = parser.parse_args()

    # 1. Load Configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Config Error: {e}")
        return

    device = get_device(config)
    logger.info(f"Running on device: {device}")

    # 2. Initialize Modules
    logger.info("Initializing modules...")
    
    # Core Vision
    detector = ObjectDetector(config)
    tracker = Tracker(config)
    
    # Identification
    team_classifier = TeamClassifier(config)
    ocr_reader = OCRReader(config) if config.identification.ocr_enabled else None
    
    # Context
    pitch_detector = PitchDetector(config)
    homography_estimator = HomographyEstimator(config)
    
    # Analytics
    event_engine = EventInference(config)
    contact_detector = ContactDetector(config)
    event_logger = EventLogger(config)
    
    # Visualization
    annotator = Annotator(config)
    # stats_engine = StatsEngine(config) # Placeholder

    # 3. Video I/O
    video_reader = VideoReader(args.source)
    video_info = video_reader.get_info()
    
    video_writer = VideoWriter(
        args.output, 
        video_info['width'], 
        video_info['height'], 
        video_info['fps']
    )

    # State Containers
    player_numbers = {} # track_id -> number
    
    logger.info("Starting processing loop...")
    
    # 4. Processing Loop
    for frame_idx, frame in tqdm(video_reader, total=video_info['total_frames']):
        
        # --- Stage A: Detection & Tracking ---
        detections = detector.detect(frame)
        tracked_detections = tracker.update(detections)
        
        # --- Stage B: Identification ---
        # 1. Team Classification
        team_ids = team_classifier.predict(frame, tracked_detections)
        
        # 2. Jersey Number (OCR)
        # Sparse update: only check if we don't have it or periodically
        if ocr_reader is not None:
            sampling_rate = config.identification.sampling_rate
            if frame_idx % sampling_rate == 0:
                for i, (track_id, bbox) in enumerate(zip(tracked_detections.tracker_id, tracked_detections.xyxy)):
                    if track_id not in player_numbers:
                        number = ocr_reader.predict(frame, bbox)
                        if number is not None:
                            player_numbers[track_id] = number
                            logger.debug(f"Identified Player #{number} (Track {track_id})")

        # --- Stage C: Calibration (Sparse) ---
        if frame_idx % 30 == 0: # Every second
            lines = pitch_detector.detect_lines(frame)
            # Need reliable points matching logic here for homography
            # For now, pitch detector just finds lines for vis
            pass

        # --- Stage D: Event Analysis ---
        # 1. ML Pipeline (TimeSformer)
        event_label = event_engine.process_frame(frame)
        if event_label:
            event_logger.log(frame_idx, event_label, {"method": "ml"})
            
        # 2. Contact Rules
        contacts = contact_detector.detect(tracked_detections)
        if contacts:
            # Simple log, could filter by team
            # event_logger.log(frame_idx, "contact", {"ids": contacts})
            pass

        # --- Stage E: Visualization ---
        annotated_frame = annotator.annotate(
            frame, 
            tracked_detections, 
            team_ids, 
            player_numbers,
            event=event_label
        )
        
        if config.visualization.show_video:
             # Just resize for display if too big
             # cv2.imshow("Football Analytics Pro", cv2.resize(annotated_frame, (1280, 720)))
             # if cv2.waitKey(1) & 0xFF == ord('q'):
             #    break
             pass

        # Save Frame
        video_writer.write(annotated_frame)

    # Cleanup
    video_writer.release()
    event_logger.save(args.output.replace('.mp4', '_events.csv'))
    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
