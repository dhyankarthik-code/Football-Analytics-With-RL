import streamlit as st
import cv2
import tempfile
import time
import os
import torch
import numpy as np
from PIL import Image

# Import Project Modules
from src.utils.logger import get_logger
from src.utils.config import load_config, get_device
from src.utils.video_io import VideoReader, VideoWriter
from src.detection import ObjectDetector
from src.tracking import Tracker
from src.identification import TeamClassifier, OCRReader
from src.visualization import Annotator

# Page Config
st.set_page_config(
    page_title="Football Analytics Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("⚽ Analytics Pro")
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
model_size = st.sidebar.selectbox("Model Size", ["n", "s", "m", "l", "x"], index=2)
enable_ocr = st.sidebar.checkbox("Enable Jersey OCR", value=False)
enable_ml_events = st.sidebar.checkbox("Enable ML Events", value=False)

def process_video(uploaded_file):
    # Save Uploaded File
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    source_path = tfile.name

    # Load Config
    config = load_config("configs/config.yaml")
    config.detection.confidence_threshold = confidence
    config.detection.model_size = model_size
    config.identification.ocr_enabled = enable_ocr
    config.events.use_ml_pipeline = enable_ml_events
    config.visualization.show_video = False
    
    # Initialize Modules
    with st.spinner("Loading AI Models..."):
        detector = ObjectDetector(config)
        tracker = Tracker(config)
        team_classifier = TeamClassifier(config)
        ocr_reader = OCRReader(config) if config.identification.ocr_enabled else None
        annotator = Annotator(config)
    
    # Video I/O
    video_reader = VideoReader(source_path)
    info = video_reader.get_info()
    st.write(f"Input Video: {info['width']}x{info['height']} @ {info['fps']:.2f} fps")
    
    # Output - Try H.264 (avc1) for best web compatibility
    # If not available, fallback to mp4v (might not play in browser but works for download)
    output_path = "output/streamlit_result.mp4"
    if not os.path.exists("output"):
        os.makedirs("output")
        
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, info['fps'], (info['width'], info['height']))
    
    if not writer.isOpened():
        st.warning("H.264 (avc1) codec not available. Falling back to 'mp4v'. Video might not play in browser.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, info['fps'], (info['width'], info['height']))
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    image_placeholder = st.empty()
    
    # Processing Loop
    player_numbers = {}
    
    # Process max 500 frames for demo speed
    max_frames = 500
    st.info(f"Processing first {max_frames} frames for quick demo...")
    
    track_history = {} # id -> list of colors
    
    try:
        for i, (frame_idx, frame) in enumerate(video_reader):
            if i >= max_frames:
                break
                
            # Debug: Check input frame
            if i == 0 and frame.mean() < 10:
                st.error("Input video frame seems suspicious (nearly black). Check source.")

            # Pipeline Steps
            detections = detector.detect(frame)
            tracked_detections = tracker.update(detections)
            team_ids = team_classifier.predict(frame, tracked_detections)
            
            # OCR (if enabled)
            if ocr_reader and frame_idx % 10 == 0:
                 for j, (track_id, bbox) in enumerate(zip(tracked_detections.tracker_id, tracked_detections.xyxy)):
                    if track_id not in player_numbers:
                        num = ocr_reader.predict(frame, bbox)
                        if num: player_numbers[track_id] = num

            # Annotate
            annotated_frame = annotator.annotate(frame, tracked_detections, team_ids, player_numbers)
            
            # Debug output frame
            if i % 100 == 0 and annotated_frame.mean() < 10:
                print(f"Frame {i}: Annotated frame is nearly black! (Mean: {annotated_frame.mean()})")

            # Write to video
            writer.write(annotated_frame)
            
            # Update UI every 5 frames
            if i % 5 == 0:
                # Convert BGR to RGB for Streamlit
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                image_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                progress = min(i / max_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing Frame {i}/{max_frames}")
                
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        writer.release()
        
    st.success("Processing Complete!")
    return output_path

# Main UI
st.title("Football Analytics Pro 🚀")
st.markdown("### Upload a match clip to analyze possession, tracking, and events.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 2 Columns for Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Video")
        st.video(uploaded_file)
        
    if st.button("Start Analysis"):
        with col2:
            st.subheader("Live Analysis")
            result_path = process_video(uploaded_file)
            
            # Show Result Video
            st.subheader("Analysis Result")
            try:
                st.video(result_path)
            except Exception as e:
                st.warning("Could not play video in browser. Please download.")

            # Download Button (MP4)
            with open(result_path, "rb") as f:
                st.download_button("Download Result (MP4)", f, "analysis_result.mp4", mime="video/mp4")
