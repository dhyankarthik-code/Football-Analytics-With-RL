import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance
from typing import Optional

class CameraMovementEstimator():
    def __init__(self,frame):
        self.minimum_distance = 5

        self.win_size = (15, 15)
        self.max_level = 2
        self.term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.max_corners = 100
        self.quality_level = 0.3
        self.min_feature_distance = 3
        self.block_size = 7
        self.feature_mask = mask_features

    def _good_features_to_track(self, gray_frame):
        return cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=int(self.max_corners),
            qualityLevel=float(self.quality_level),
            minDistance=float(self.min_feature_distance),
            mask=self.feature_mask,
            blockSize=int(self.block_size)
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = self._good_features_to_track(old_gray)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            next_guess = old_features.copy()
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(
                old_gray,
                frame_gray,
                old_features,
                next_guess,
                winSize=self.win_size,
                maxLevel=self.max_level,
                criteria=self.term_criteria,
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = self._good_features_to_track(frame_gray)

            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement

    def get_camera_movement_from_video(self, video_path, read_from_stub=False, stub_path=None, max_frames: Optional[int] = None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for camera movement: {video_path}")

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Cannot read frames from video: {video_path}")

        camera_movement = [[0, 0]]
        frames_processed = 1

        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        old_features = self._good_features_to_track(old_gray)

        while True:
            if max_frames is not None and frames_processed >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if old_features is None or len(old_features) == 0:
                old_features = self._good_features_to_track(old_gray)
                camera_movement.append([0, 0])
                old_gray = frame_gray.copy()
                continue

            next_guess = old_features.copy()
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray,
                frame_gray,
                old_features,
                next_guess,
                winSize=self.win_size,
                maxLevel=self.max_level,
                criteria=self.term_criteria,
            )
            if new_features is None or status is None:
                camera_movement.append([0, 0])
                old_gray = frame_gray.copy()
                old_features = self._good_features_to_track(old_gray)
                continue

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for new, old, st in zip(new_features, old_features, status):
                if st == 0:
                    continue
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement.append([camera_movement_x, camera_movement_y])
                old_features = self._good_features_to_track(frame_gray)
            else:
                camera_movement.append([0, 0])

            old_gray = frame_gray.copy()
            frames_processed += 1

            if frames_processed % 200 == 0:
                print(f"[CameraMovementEstimator] processed {frames_processed} frames", flush=True)

        cap.release()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames