from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
import torch
from typing import List, Optional

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        # Enforce GPU-only execution for model inference
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU-only execution requested.")
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        processed_positions = []
        for entry in ball_positions:
            bbox = entry.get(1, {}).get('bbox') if isinstance(entry, dict) else None
            if bbox and len(bbox) == 4:
                processed_positions.append(bbox)
            else:
                processed_positions.append([np.nan, np.nan, np.nan, np.nan])

        df_ball_positions = pd.DataFrame(processed_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values forward/backward to cover leading/trailing gaps
        df_ball_positions = df_ball_positions.interpolate(limit_direction='both')
        df_ball_positions = df_ball_positions.bfill().ffill().fillna(0.0)

        filled_positions = df_ball_positions.to_numpy().tolist()
        return [{1: {"bbox": bbox}} for bbox in filled_positions]

    def detect_frames(self, frames: List[np.ndarray]):
        batch_size = 8
        detections = []
        for i in range(0, len(frames), batch_size):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA became unavailable during predict().")
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1, device='cuda')
            detections += detections_batch
        return detections

    def _ensure_track_slots(self, tracks, frame_num: int):
        for key in tracks.keys():
            while len(tracks[key]) <= frame_num:
                tracks[key].append({})

    def _process_detection(self, detection, tracks, frame_num: int):
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        detection_supervision = sv.Detections.from_ultralytics(detection)
        class_ids = getattr(detection_supervision, 'class_id', None)
        if class_ids is not None:
            for object_ind, class_id in enumerate(class_ids):
                if cls_names[class_id] == "goalkeeper":
                    class_ids[object_ind] = cls_names_inv["player"]

        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

        self._ensure_track_slots(tracks, frame_num)

        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv['player']:
                tracks["players"][frame_num][track_id] = {"bbox": bbox}

            if cls_id == cls_names_inv['referee']:
                tracks["referees"][frame_num][track_id] = {"bbox": bbox}

        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            if cls_id == cls_names_inv['ball']:
                tracks["ball"][frame_num][1] = {"bbox": bbox}

    def get_object_tracks_for_video(self, video_path: str, resize_to: Optional[tuple] = None,
                                    read_from_stub: bool = False, stub_path: Optional[str] = None,
                                    max_frames: Optional[int] = None, progress_interval: int = 200):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.tracker = sv.ByteTrack()  # reset tracker state for new video
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        batch_frames: List[np.ndarray] = []
        frame_counter = 0
        frames_read = 0
        progress_interval = max(progress_interval, 1)
        while True:
            if max_frames is not None and frames_read >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to)
            batch_frames.append(frame)
            frames_read += 1

            if len(batch_frames) == 16:
                detections = self.detect_frames(batch_frames)
                for detection in detections:
                    self._process_detection(detection, tracks, frame_counter)
                    frame_counter += 1
                    if frame_counter % progress_interval == 0:
                        print(f"[Tracker] processed {frame_counter} frames", flush=True)
                batch_frames.clear()

        if batch_frames:
            detections = self.detect_frames(batch_frames)
            for detection in detections:
                self._process_detection(detection, tracks, frame_counter)
                frame_counter += 1
                if frame_counter % progress_interval == 0:
                    print(f"[Tracker] processed {frame_counter} frames", flush=True)

        cap.release()

        if frame_counter and frame_counter % progress_interval != 0:
            print(f"[Tracker] processed {frame_counter} frames", flush=True)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Backwards compatibility: fall back to streaming method when frames is a list
        if isinstance(frames, list):
            return self.get_object_tracks_for_video_frames(frames, read_from_stub=read_from_stub, stub_path=stub_path)
        raise NotImplementedError("Use get_object_tracks_for_video or pass a list of frames.")

    def get_object_tracks_for_video_frames(self, frames: List[np.ndarray], read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        self.tracker = sv.ByteTrack()
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        detections = self.detect_frames(frames)
        for frame_num, detection in enumerate(detections):
            self._process_detection(detection, tracks, frame_num)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        valid_frames = team_ball_control_till_frame[team_ball_control_till_frame > 0]
        if valid_frames.size == 0:
            team_1 = 0.0
            team_2 = 0.0
        else:
            team_1_num_frames = np.count_nonzero(valid_frames == 1)
            team_2_num_frames = np.count_nonzero(valid_frames == 2)
            total_tracked = team_1_num_frames + team_2_num_frames
            if total_tracked == 0:
                team_1 = 0.0
                team_2 = 0.0
            else:
                team_1 = team_1_num_frames / total_tracked
                team_2 = team_2_num_frames / total_tracked

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    def annotate_frame(self, frame, tracks, frame_index, team_ball_control):
        """Annotate a single frame in-place and return it (memory-friendly path)."""
        frame = frame.copy()

        player_dict = tracks["players"][frame_index] if frame_index < len(tracks["players"]) else {}
        ball_dict = tracks["ball"][frame_index] if frame_index < len(tracks["ball"]) else {}
        referee_dict = tracks["referees"][frame_index] if frame_index < len(tracks["referees"]) else {}

        # Draw Players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get('has_ball', False):
                frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

        # Draw Referees
        for _, referee in referee_dict.items():
            frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

        # Draw Ball
        for _, ball in ball_dict.items():
            frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        # Draw Team Ball Control
        frame = self.draw_team_ball_control(frame, frame_index, team_ball_control)

        return frame