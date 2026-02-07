"""
Verification script for Football Analytics Pro.
Run this to check if all dependencies are installed and models can be loaded.
"""
import sys
import os
import torch

try:
    print("1. Checking Imports...")
    import ultralytics
    import supervision
    import paddleocr
    import transformers
    import box
    import cv2
    import numpy as np
    print("✅ Imports successful.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("\n2. Checking Config...")
try:
    from src.utils.config import load_config
    config = load_config("configs/config.yaml")
    print(f"✅ Config loaded. Project: {config.project.name}")
except Exception as e:
    print(f"❌ Config check failed: {e}")
    sys.exit(1)

print("\n3. Checking Hardware...")
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  CUDA not available. Running on CPU (will be slow).")

print("\n4. Checking Model Loading (This may download weights)...")
try:
    print("   - Loading YOLOv8...")
    from ultralytics import YOLO
    model = YOLO(config.detection.model_path)
    print("✅ YOLOv8 loaded.")
    
    print("   - Loading TimeSformer...")
    from transformers import TimesformerForVideoClassification
    ts_model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    print("✅ TimeSformer loaded.")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)

print("\n✅ Verification Complete! System is ready to run.")
print("Run the pipeline using: python main.py --source input.mp4")
