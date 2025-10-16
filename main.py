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