"""
Simple video viewer for Streamlit - displays pre-processed videos.
No YOLO/tracking dependencies needed.
"""
import streamlit as st
import cv2
import tempfile
import numpy as np

st.set_page_config(page_title="Video Viewer", page_icon="🎬", layout="wide")

st.title("🎬 Football Analytics Video Viewer")
st.markdown("View pre-processed analytics videos")

uploaded_video = st.file_uploader("Upload Processed Video", type=["mp4", "mov", "avi", "MP4"])

if uploaded_video:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    st.success(f"✅ Video loaded: {uploaded_video.name}")
    
    # Display video using st.video
    st.video(video_path)
    
    # Video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    # Show video info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resolution", f"{width}x{height}")
    col2.metric("FPS", f"{fps:.1f}")
    col3.metric("Frames", f"{total_frames}")
    col4.metric("Duration", f"{duration:.1f}s")
    
    st.markdown("---")
    st.markdown("### 🎨 Features in This Video")
    st.markdown("""
    ✅ **Team-Colored Bounding Boxes** - Red (Team 1) vs Blue (Team 2)  
    ✅ **Ball Tracking** - Green box with trajectory  
    ✅ **Speed Labels** - Real-time km/h display  
    ✅ **Trajectory Prediction** - Purple trail (history) + Cyan line (future)  
    ✅ **Pitch Boundary** - Green rectangle  
    ✅ **Analytics Overlay** - Frame counter and FPS  
    """)
    
    st.markdown("---")
    st.markdown("### 📊 How to Generate More Videos")
    st.code("""
# Quick demo (standalone, no dependencies)
python standalone_demo.py --video YOUR_VIDEO.mp4 --output result.mp4

# Full pipeline (with YOLO detection)
python production_pipeline.py --video YOUR_VIDEO.mp4 --output result.mp4
    """, language="bash")

else:
    st.info("👆 Upload a video file to view it")
    
    st.markdown("### 📁 Try These Demo Videos:")
    st.markdown("""
    - `output/feature_demo.mp4` - Standalone demo with all features
    - `output/demo_analysis.mp4` - Original YOLO output
    - `demo_clip.mp4` - Raw input video
    """)
