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
