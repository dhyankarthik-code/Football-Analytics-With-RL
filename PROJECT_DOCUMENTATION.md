# Football Analytics with YOLO-B

Version: 1.0
Date: 2025-10-16
Repository: https://github.com/dhyankarthik-code/Football-Analytics-with-YOLO-B

---

## 1. Executive summary

This document describes "Football Analytics with YOLO-B", a project that applies object detection and computer vision techniques to football (soccer) video to extract actionable analytics such as player and ball tracking, speed and distance estimations, team assignment, camera movement estimation, and view transformation for top-down perspective. The project is implemented in Python, uses a YOLO model for detection, and contains modular components for tracking, assigning, and estimating metrics. This document covers the problem statement, solution overview, architecture, setup, how the system works, module-level explanations, limitations, and how to access the repository.

---

## 2. Problem statement (What we solve)

Football matches contain rich spatial-temporal data that can be extracted from broadcast or recorded video. Manual annotation and analysis is slow and expensive. Automated systems that can detect and track players and the ball, estimate movement metrics, and project play onto a consistent field plane enable performance analysis, scouting, automated highlights, and fan insights.

This project addresses the following problems:

- Detect players and ball in video frames robustly in different lighting and camera angles.
- Track detected objects across frames and assign consistent IDs (multi-object tracking).
- Distinguish teams and assign each player to a team.
- Estimate per-player speed and distance traveled using frame-to-frame motion and camera calibration.
- Estimate camera movement so world-space computations remain stable and compensations can be applied.
- Transform camera view into a bird's-eye (top-down) view for heatmaps and spatial analysis.

Target users: sports analysts, coaches, data scientists, and developers wanting an end-to-end pipeline to extract match-level and player-level metrics from match footage.

---

## 3. Solution overview (How we solve it)

High level architecture and components:

- Detection (YOLO-based): `yolo_inference.py` — Runs inference on video frames using a YOLO model (`models/best.pt`) to produce bounding boxes for players and ball.
- Tracking: `trackers/tracker.py` — Performs data association across frames to maintain persistent IDs for detected objects.
- Team assignment: `team_assigner/team_assigner.py` — Heuristics or clustering to separate players into teams based on jersey color and spatial relations.
- Player-ball assignment: `player_ball_assigner/player_ball_assigner.py` — Associates the ball with the closest player or identifies free-ball situations.
- Speed and distance estimation: `speed_and_distance_estimator/speed_and_distance_estimator.py` — Uses per-frame positions, frame timestamps, and an approximate scale (meters/pixel) from the view transformer to compute player speed and cumulative distance.
- Camera movement estimation: `camera_movement_estimator/camera_movement_estimator.py` — Estimates frame-to-frame camera motion to stabilize world coordinates and correctly compute player displacements.
- View transformer: `view_transformer/view_transformer.py` — Computes a homography from the camera image to a standardized pitch coordinate frame to enable bird's-eye metrics and heatmaps.
- Utilities: `utils/` — Contains helpers for video I/O, bounding box math, and other repeated logic.

Dataflow summary:

1. Read video frames from `input_videos/`.
2. Run YOLO detector to get bounding boxes for players and ball.
3. Track detections across frames to maintain IDs.
4. Assign players to teams using color clustering.
5. Estimate camera motion and remove camera-induced displacements.
6. Transform detections into pitch coordinates and compute speeds and distances.
7. Output annotated video to `output_videos/` and CSV/JSON analytics files.

---

## 4. Setup and installation (How to get it running)

Environment requirements (from `requirements.txt`):

- Python 3.8+ (tested on 3.9/3.10)
- torch (PyTorch) and torchvision compatible with your CUDA (or CPU-only)
- OpenCV (cv2)
- numpy, pandas, scikit-learn (for clustering/color analysis)
- yolov5 runtime or PyTorch model loading utilities

Minimal setup steps:

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place your YOLO model into `models/best.pt` (this repo's `.gitignore` excludes models/ by default; keep models out of git or use Git LFS).

3. Prepare input video(s) in `input_videos/`.

4. Run the main script for inference and analytics:

```powershell
python main.py --input input_videos/match1.mp4 --output output_videos/annotated_match1.mp4
```

5. Outputs: annotated video(s) in `output_videos/` and optionally CSV/JSON files with per-frame and per-player metrics.

Configuration options:

- Model path, confidence threshold, NMS IoU threshold.
- Camera calibration / homography points (for `view_transformer`) — stored in a settings file or passed as arguments.
- Frame sampling rate — for faster processing, you can skip frames and interpolate.

---

## 5. How the system works — internals and module explanations

5.1 Detection (`yolo_inference.py`)

- Loads a YOLO PyTorch model and runs inference on each frame.
- Outputs bounding boxes (x1,y1,x2,y2), confidence, and class id (player/ball).
- Performs basic filtering (confidence threshold, size threshold) to reduce false positives.

5.2 Tracking (`trackers/tracker.py`)

- A tracker (e.g., SORT, DeepSORT, or a custom centroid + Kalman filter tracker) is used.
- Detection boxes are associated to existing tracks using IoU and/or feature distance.
- Tracks are created, updated, and terminated based on detection continuity and age thresholds.

5.3 Team assignment (`team_assigner/team_assigner.py`)

- Extracts dominant colors from the player bounding boxes using HSV clustering.
- Performs k-means (k=2) clustering per frame or across a sliding window to separate jerseys.
- Assigns a team id to each track by majority vote across recent frames.

5.4 Player-ball assignment (`player_ball_assigner/player_ball_assigner.py`)

- Computes the distance between the ball's center and the nearest player's predicted position.
- Uses temporal smoothing to avoid noisy switches during occlusion.

5.5 Speed and distance estimation (`speed_and_distance_estimator/speed_and_distance_estimator.py`)

- Converts pixel displacements to meters using the homography scale or a known reference object.
- Uses the time difference between frames (fps) to estimate instantaneous speed and integrates over time for distance.

5.6 Camera movement estimation (`camera_movement_estimator/camera_movement_estimator.py`)

- Detects keypoints (ORB/FAST) and matches between consecutive frames.
- Estimates an affine or homography transform describing camera motion.
- Compensates player trajectories by removing camera motion, improving world-space estimates.

5.7 View transformer (`view_transformer/view_transformer.py`)

- A homography maps image coordinates to a standardized pitch coordinate system.
- Calibration is done by selecting 4 or more correspondences between pitch corners/lines and image points.
- After transformation, positions are in meters (or normalized pitch units) and can be used for heatmaps.

---

## 6. Outputs, analytics, and examples of usage

- Annotated video with bounding boxes and track IDs for players and ball.
- Per-player CSV with columns: track_id, frame, time, x_image, y_image, x_pitch, y_pitch, speed_m_s, cumulative_distance_m.
- Team-level summaries: possession estimates, heatmaps, average speeds.

Usage examples:

- Generate player trajectories and plot heatmaps using the transformed pitch coordinates.
- Calculate sprint distance (speed above a threshold) per player across the match.
- Build automated highlights based on ball possession changes or events (shots, passes) detected via heuristics.

---

## 7. Limitations, future work, and ethical considerations

Limitations:

- Detection errors happen with heavy occlusion, motion blur, or extreme camera zoom.
- Team assignment by color fails when teams have similar colors or lighting is poor.
- Camera calibration requires manual correspondences for accurate pitch mapping.
- Model file sizes are large — models should be stored externally (Git LFS or cloud).

Future work:

- Integrate pose estimation to improve player localization and action recognition.
- Use DeepSORT or appearance features to reduce ID switch errors.
- Improve ball detection with a specialized small-object detector and higher-resolution input.

Ethical considerations:

- Respect broadcast rights and privacy when processing video — ensure you have permission to analyze footage.
- Be transparent about analytics limitations when used for scouting or performance assessment.

---

## 8. Where the project is stored

Repository: https://github.com/dhyankarthik-code/Football-Analytics-with-YOLO-B

To see the generated documentation file, visit:

https://github.com/dhyankarthik-code/Football-Analytics-with-YOLO-B/blob/main/PROJECT_DOCUMENTATION.md

---

If you need this exported as a formatted 7-page PDF, I can render the Markdown to PDF and add it to the repo (requires pandoc or wkhtmltopdf / a LaTeX environment). Tell me if you want the PDF and I will generate and commit it.
