<<<<<<< HEAD
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
=======
from typing import cast

import os
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Stream video instead of loading all frames to avoid OOM
    video_path = 'input_videos/sample_1min.avi'
    # Removed MAX_FRAMES limit for full video processing
    max_frames = None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read first frame of video: {video_path}")
    cap.release()

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks_for_video(video_path,
                                                resize_to=None,
                                                read_from_stub=False,
                                                stub_path=None,
                                                max_frames=max_frames)
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # # camera movement estimator - temporarily disabled
    # camera_movement_estimator = CameraMovementEstimator(first_frame)
    # camera_movement_per_frame = camera_movement_estimator.get_camera_movement_from_video(video_path,
    #                                                                                       read_from_stub=False,
    #                                                                                       stub_path=None,
    #                                                                                       max_frames=max_frames)
    # camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # # View Trasnformer - temporarily disabled
    # view_transformer = ViewTransformer()
    # view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.set_manual_team_colors((160, 180, 180), (100, 120, 115))  # Light team and Dark team
    team_assigner.set_debug(True)
    team_assigner.assign_team_color(first_frame, 
                                    tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Use a reference frame to determine team color to avoid loading the entire video into RAM
            team = team_assigner.get_player_team(first_frame,   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))

    # Debug: Print team assignment summary
    team_counts = {}
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = track.get('team', 0)
            team_counts[team] = team_counts.get(team, 0) + 1
    print("Team assignment summary:", team_counts)
    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        # Defensive access to ball bbox in case detection is missing in a frame
        ball_entry = tracks['ball'][frame_num].get(1)
        if not ball_entry or 'bbox' not in ball_entry:
            last_control = team_ball_control[-1] if team_ball_control else 0
            team_ball_control.append(last_control)
            continue
        ball_bbox = ball_entry['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            last_control = team_ball_control[-1] if team_ball_control else 0
            team_ball_control.append(last_control)
    team_ball_control= np.array(team_ball_control)


    # Draw and write output frame-by-frame to avoid high memory usage
    # FourCC helper to satisfy some linters/environments
    fourcc_func = getattr(cv2, 'VideoWriter_fourcc', None)
    if not callable(fourcc_func):
        raise RuntimeError('cv2.VideoWriter_fourcc not available in this OpenCV build.')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fourcc = cast(int, fourcc_func(*'XVID'))
    out = cv2.VideoWriter('output_videos/output_video.avi', fourcc, 24, (width, height))

    frame_idx = 0
    total_tracked_frames = len(tracks['players'])
    def slice_tracks_for_frame(all_tracks, idx):
        # Build a minimal view of tracks for a single frame index
        view = {}
        for obj, obj_tracks in all_tracks.items():
            if idx < len(obj_tracks):
                view[obj] = [obj_tracks[idx]]
            else:
                view[obj] = [{}]
        return view

    while frame_idx < total_tracked_frames:
        ret, frame = cap.read()
        if not ret:
            break
        idx = frame_idx
        annotated = tracker.annotate_frame(frame, tracks, idx, team_ball_control)
        # Overlay camera movement text for current frame - temporarily disabled
        movement_slice = [0, 0]  # camera_movement_per_frame[idx] if idx < len(camera_movement_per_frame) else [0, 0]
        # annotated = camera_movement_estimator.draw_camera_movement(
        #     [annotated],
        #     [movement_slice]
        # )[0]
        # Add team classification overlay
        player_detections = {pid: {'bbox': track['bbox']} for pid, track in tracks['players'][idx].items()}
        annotated = team_assigner.visualize_team_classification(annotated, player_detections)
        # Draw speed/distance if available using a sliced track view for this frame index
        mini_tracks = slice_tracks_for_frame(tracks, idx)
        annotated = speed_and_distance_estimator.draw_speed_and_distance([annotated], mini_tracks)[0]
        out.write(annotated)
        frame_idx += 1

    out.release()
    cap.release()

    print("Processing complete. Output saved to output_videos/output_video.avi")

if __name__ == '__main__':
    main()
>>>>>>> 187d1121d51dfdc2e5682cbdd5cccdd07690b538
