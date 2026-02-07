# Football Analytics Pro - Setup Guide

## 📋 Prerequisites

Before setting up Football Analytics Pro, ensure your system meets these requirements:

### Hardware
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **GPU**: NVIDIA GPU with CUDA support (RTX 2060 or better recommended)
  - For CPU-only: Processing will be significantly slower (~10-20x)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 5GB+ for models and dependencies

### Software
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: Version 3.8, 3.9, 3.10, or 3.11
- **Git**: For cloning the repository
- **CUDA Toolkit**: 11.8+ (if using GPU acceleration)

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Football-Analytics-with-YOLO-B
```

### Step 2: Create Virtual Environment

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Tip**: Always activate the virtual environment before running the pipeline

### Step 3: Install Dependencies

#### For GPU (CUDA 11.8+)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### For CPU-Only
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **Troubleshooting**: If PaddlePaddle installation fails, install CPU version:
> ```bash
> pip install paddlepaddle==2.6.0 -i https://mirror.baidu.com/pypi/simple
> ```

### Step 4: Download Model Weights

Download YOLOv8 models to the `models/` directory:

```bash
# YOLOv8x for object detection (required)
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt -P models/

# YOLOv8x-pose for keypoint detection (optional, for future foul detection)
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt -P models/
```

**Windows Alternative** (if wget not available):
1. Download manually from [Ultralytics Releases](https://github.com/ultralytics/assets/releases/tag/v8.1.0)
2. Place `yolov8x.pt` in the `models/` folder

### Step 5: Verify Installation

Test that all imports work correctly:

```bash
python -c "from src.utils.logger import get_logger; print('✅ Logger OK')"
python -c "import ultralytics; print('✅ YOLO OK')"
python -c "import paddleocr; print('✅ PaddleOCR OK')"
python -c "import cv2; print('✅ OpenCV OK')"
```

All checks should print `✅ [Module] OK` without errors.

## ⚙️ Configuration

### Basic Configuration

Edit `configs/config.yaml` to customize pipeline behavior:

```yaml
project:
  name: "FootballAnalytics_Pro"
  version: "2.0.0"
  output_dir: "output"          # Where results are saved
  log_level: "INFO"              # DEBUG, INFO, WARNING, ERROR

detection:
  model_path: "models/yolov8x.pt"
  confidence_threshold: 0.5      # Lower = more detections (but more false positives)
  iou_threshold: 0.45            # NMS threshold
  img_size: 640                  # Input resolution (higher = slower but better)

tracking:
  tracker_type: "deepsort"       # Options: deepsort, ocsort, byte
  max_age: 30                    # Max frames to keep lost tracks
  n_init: 3                      # Frames before confirming new track
  nn_budget: 100                 # Re-ID feature budget

identification:
  ocr_model_lang: "en"
  team_cluster_method: "kmeans"
  sampling_rate: 30              # Run OCR every N frames (higher = faster)

calibration:
  pitch_length: 105              # Standard pitch: 105m x 68m
  pitch_width: 68
```

### Advanced Tuning

**For High Accuracy** (slower):
- `detection.confidence_threshold: 0.4`
- `detection.img_size: 1280`
- `identification.sampling_rate: 10`

**For Speed** (lower accuracy):
- `detection.confidence_threshold: 0.6`
- `detection.img_size: 416`
- `identification.sampling_rate: 60`

## 🎬 Running the Pipeline

### Basic Usage

1. **Place input video** in `data/` directory:
   ```bash
   # Example
   cp /path/to/match.mp4 data/match.mp4
   ```

2. **Run pipeline**:
   ```bash
   python main.py
   ```

3. **View results** in `output/`:
   - Annotated video: `output/annotated_match.mp4`
   - Analytics JSON: `output/analytics.json`
   - Logs: Console output

### Custom Configuration

```bash
python main.py --config configs/custom_config.yaml
```

### Command Line Options

```bash
python main.py --help
```

Options:
- `--config PATH` - Path to YAML config file (default: `configs/config.yaml`)
- `--input PATH` - Override input video path (future feature)
- `--output PATH` - Override output directory (future feature)

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution**: Ensure you're running from the project root directory:
```bash
cd Football-Analytics-with-YOLO-B
python main.py
```

### Issue: CUDA out of memory

**Solutions**:
1. Reduce batch size or image resolution in config
2. Use smaller model: `yolov8l.pt` instead of `yolov8x.pt`
3. Enable CPU-only mode (see Installation Step 3)

### Issue: PaddleOCR crashes or slow

**Solutions**:
1. Increase `identification.sampling_rate` to run OCR less frequently
2. Install CPU-only PaddlePaddle if GPU conflicts occur:
   ```bash
   pip uninstall paddlepaddle-gpu
   pip install paddlepaddle==2.6.0
   ```

### Issue: "Cannot find yolov8x.pt"

**Solution**: Verify model file exists:
```bash
# Windows
dir models\yolov8x.pt

# Linux/Mac
ls -lh models/yolov8x.pt
```
If missing, re-download from Step 4.

### Issue: Video codec errors

**Solution**: Install ffmpeg:
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

## 📊 Expected Performance

| Hardware | Resolution | FPS (Detection Only) | FPS (Full Pipeline) |
|----------|-----------|---------------------|---------------------|
| RTX 4090 | 1080p | ~60 FPS | ~30 FPS |
| RTX 3080 | 1080p | ~45 FPS | ~20 FPS |
| RTX 2060 | 1080p | ~25 FPS | ~10 FPS |
| CPU (i7) | 1080p | ~0.5 FPS | ~0.2 FPS |

> **Note**: Full pipeline includes detection + tracking + identification + visualization

## 🔄 Updating the Project

Pull latest changes and reinstall dependencies:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## 🆘 Getting Help

If you encounter issues not covered here:

1. **Check logs**: Look for errors in console output
2. **Enable debug mode**: Set `log_level: "DEBUG"` in config
3. **Search issues**: Check GitHub Issues for similar problems
4. **Create issue**: Report bug with:
   - OS and Python version
   - Full error traceback
   - Config file used
   - Steps to reproduce

## 📚 Next Steps

After setup, explore:
- **[README.md](../README.md)** - Project overview
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - Technical details
- **[configs/config.yaml](../configs/config.yaml)** - Configuration reference
- **notebooks/** - Jupyter notebooks for experimentation

---

✅ **Setup complete!** You're ready to analyze football matches with computer vision.
