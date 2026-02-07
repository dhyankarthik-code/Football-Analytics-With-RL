<<<<<<< HEAD
# Football Analytics with RL

Advanced football analytics system integrating YOLO object detection, ByteTrack tracking, and Reinforcement Learning for enhanced ball tracking.

## 🚀 Features

### Core Analytics
- ✅ **YOLO Object Detection** - Detects players, ball, and referees
- ✅ **ByteTrack Tracking** - Multi-object tracking with ID persistence
- ✅ **Team Classification** - K-means clustering for team assignment
- ✅ **Speed Calculation** - Real-time speed in km/h with calibration
- ✅ **Ball Possession** - Identifies which player has the ball

### 🤖 RL Integration (NEW!)
- ✅ **Iterative Ball Refinement** - PPO-based iterative bounding box refinement
- ✅ **Custom Gymnasium Environment** - Football-specific RL environment
- ✅ **Composite Reward Function** - IoU, smoothness, track continuity
- ✅ **Confidence-Based Early Stopping** - Adaptive iteration count
- ✅ **Interactive Dashboard** - Streamlit dashboard for training/inference

### 🎨 Production Features (NEW!)
- ✅ **Team-Colored Bounding Boxes** - Red vs Blue team visualization
- ✅ **Trajectory Prediction** - Physics-based ball path forecasting
- ✅ **Pitch Boundary Filtering** - Excludes ball boys and spectators
- ✅ **Jersey Number Detection** - OCR-based player identification
- ✅ **GPU Optimization** - CUDA-accelerated processing

## 📁 Project Structure

```
├── src/
│   ├── detection.py          # YOLO object detection
│   ├── tracking.py            # ByteTrack tracker
│   ├── team_assigner.py       # Team classification
│   ├── rl/                    # RL module (NEW)
│   │   ├── environment.py     # Gymnasium environment
│   │   ├── reward.py          # Reward calculator
│   │   ├── agent.py           # RL agent wrapper
│   │   ├── iterative_agent.py # Iterative refinement
│   │   └── trainer.py         # PPO training script
│   ├── analytics/             # Analytics module (NEW)
│   │   ├── pitch_filter.py    # Pitch boundary detection
│   │   ├── speed_calculator.py # Speed calculation
│   │   ├── trajectory_predictor.py # Ball trajectory
│   │   └── jersey_detector.py # Jersey number OCR
│   └── visualization/
│       ├── annotator.py       # Original annotator
│       └── enhanced_annotator.py # Team-colored annotator (NEW)
├── production_pipeline.py     # Full production pipeline (NEW)
├── standalone_demo.py         # Standalone demo (NEW)
├── rl_dashboard.py            # Streamlit dashboard (NEW)
├── simple_viewer.py           # Video viewer (NEW)
└── output/
    └── feature_demo.mp4       # Demo video (NEW)
```

## 🎯 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/dhyankarthik-code/Football-Analytics-With-RL.git
cd Football-Analytics-With-RL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install RL dependencies
pip install stable-baselines3[extra] gymnasium easyocr
```

### 2. Run Standalone Demo (No Training Required)

```bash
# Generate demo video with all features
python standalone_demo.py --video demo_clip.mp4 --output output/demo.mp4 --frames 200

# View in simple viewer
streamlit run simple_viewer.py
```

### 3. Train RL Model (Optional)

```bash
# Prepare ground truth annotations (JSON format)
# See RL_README.md for annotation format

# Train PPO agent
python -m src.rl.trainer \
    --video_dir data/videos/ \
    --ground_truth_dir data/annotations/ \
    --total_timesteps 1000000

# Monitor training
tensorboard --logdir models/rl/tensorboard
```

### 4. Run Production Pipeline

```bash
# With RL model
python production_pipeline.py \
    --video input.mp4 \
    --output output.mp4 \
    --rl_model models/rl/ppo_ball_tracking_final.zip

# CPU only (slower)
python production_pipeline.py --video input.mp4 --output output.mp4 --cpu
```

### 5. Launch Dashboard

```bash
# Full RL dashboard
streamlit run rl_dashboard.py

# Simple video viewer
streamlit run simple_viewer.py --server.port 8502
```

## 🎨 Output Features

The processed videos include:

1. **Team-Colored Bounding Boxes**
   - Red boxes for Team 1
   - Blue boxes for Team 2
   - Green box for ball

2. **Speed Labels**
   - Real-time km/h under players and ball
   - Calibrated to pitch dimensions

3. **Ball Trajectory**
   - Purple trail showing ball history
   - Cyan line predicting future path

4. **Pitch Boundary**
   - Green rectangle marking playing field
   - Filters out non-players

5. **Analytics Overlay**
   - Frame counter
   - Processing FPS
   - GPU utilization (if available)

## 📊 RL Architecture

### Observation Space
- Ball bounding box (normalized)
- Ball velocity (2D vector)
- Visual features (256-dim CNN features)
- Previous ball box (temporal context)

### Action Space
- Continuous: `[Δx, Δy, Δw, Δh]` (-0.1 to 0.1)
- Adjusts bounding box by up to 10% per step

### Reward Function
```
R = w₁·R_IoU + w₂·R_diff + w₃·R_smooth + w₄·R_track - λ·cost
```

Where:
- **R_IoU**: Intersection over Union with ground truth
- **R_diff**: Improvement in IoU from previous step
- **R_smooth**: Temporal smoothness penalty
- **R_track**: Track continuity bonus
- **cost**: Per-step computational penalty

### Iterative Refinement
1. YOLO detects ball → Initial box (confidence: 0.60)
2. RL refines → Iteration 1 (confidence: 0.68)
3. RL refines → Iteration 2 (confidence: 0.76)
4. RL refines → Iteration 3 (confidence: 0.87) ✓ **STOP**

Early stopping when:
- Confidence > 0.85
- Improvement < 0.01
- Max iterations (5) reached

## 📈 Performance

| Component | GPU | CPU |
|-----------|-----|-----|
| YOLO Detection | 15ms | 80ms |
| ByteTrack | 2ms | 5ms |
| RL Iterative (avg 2.8 iter) | 8ms | 25ms |
| Jersey OCR (when triggered) | 50ms | 200ms |
| **Total per frame** | ~30ms (33 FPS) | ~120ms (8 FPS) |

## 🔧 Configuration

Edit `configs/config.yaml`:

```yaml
detection:
  model_path: "models/yolo11x.pt"
  confidence_threshold: 0.3

tracking:
  track_thresh: 0.25
  match_thresh: 0.8

rl:
  max_iterations: 5
  confidence_threshold: 0.85
  min_improvement: 0.01
```

## 📚 Documentation

- [RL Integration Guide](RL_README.md)
- [Iterative Refinement Explained](docs/iterative_refinement_guide.md)
- [RL Architecture Details](docs/rl_architecture_explained.md)
- [Production Pipeline Guide](docs/production_walkthrough.md)
- [Dashboard User Guide](docs/dashboard_viewing_guide.md)

## 🎓 Research & References

- **PPO Algorithm**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **ByteTrack**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (2022)
- **YOLO**: Ultralytics YOLOv11
- **Stable-Baselines3**: Raffin et al., "Stable-Baselines3: Reliable RL Implementations" (2021)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Ultralytics for YOLO
- ByteTrack team
- Stable-Baselines3 contributors
- OpenAI Gymnasium

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for football analytics and AI research**
=======
# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements
To run this project, you need the following:
- Python 3.9 or newer (virtual environments recommended)
- The Python dependencies listed in `requirements.txt`
- A trained YOLOv5/YOLOv8 checkpoint (see below)

Install the dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Installing `ultralytics` will also pull in PyTorch. If you need GPU acceleration, follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the wheel that matches your CUDA version, then rerun `pip install -r requirements.txt`.

## Project setup

1. **Download the model weights**
	- Grab `best.pt` from the link above and place it in `models/best.pt`.
2. **Provide an input video**
	- Copy a match video into `input_videos/` (e.g. `input_videos/08fd33_4.mp4`).
3. **(Optional) Use provided stubs**
	- The `stubs/` directory contains cached detections and camera movement values so you can generate an output video quickly without running the full detector on every frame. Keep these files in place for a fast first run.

## Running the pipeline

After completing the setup steps and activating your environment, run:

```powershell
python main.py
```

The annotated video will be saved to `output_videos/output_video.avi`. Intermediate artefacts such as cropped images and screenshots are also written to `output_videos/` for convenience.

### Quick YOLO inference demo

If you only want to run YOLO on a video and inspect the raw detections, execute:

```powershell
python yolo_inference.py
```

This will write prediction frames to the `runs/` directory created by Ultralytics and echo detection metadata to the console.
>>>>>>> 187d1121d51dfdc2e5682cbdd5cccdd07690b538
