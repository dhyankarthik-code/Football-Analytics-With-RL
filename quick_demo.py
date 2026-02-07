"""
Quick demo script for production features (no config dependency).
Shows team colors, speed, trajectory, and analytics.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analytics import (
    PitchBoundaryFilter,
    SpeedCalculator,
    TrajectoryPredictor
)
from src.visualization.enhanced_annotator import EnhancedAnnotator

print("✅ Imports successful!")


def demo_analytics(video_path: str, output_path: str, max_frames: int = 300):
    """
    Quick demo of analytics features without full pipeline.
    """
    print(f"\n🎬 Processing: {video_path}")
    print(f"📁 Output: {output_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📐 Resolution: {width}x{height} @ {fps} FPS")
    
    # Initialize components
    pitch_filter = PitchBoundaryFilter(use_auto_detection=True)
    speed_calc = SpeedCalculator(fps=fps)
    speed_calc.auto_calibrate_from_frame(width, height)
    trajectory = TrajectoryPredictor(prediction_horizon=10)
    annotator = EnhancedAnnotator()
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    pitch_initialized = False
    
    print("\n⚙️  Processing frames...")
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Initialize pitch on first frame
        if not pitch_initialized:
            pitch_filter.initialize_pitch_boundary(frame)
            pitch_initialized = True
            print("✓ Pitch boundary detected")
        
        # Simulated detections for demo (random positions)
        # In real pipeline, these come from YOLO
        demo_frame = frame.copy()
        
        # Draw pitch boundary
        demo_frame = pitch_filter.draw_pitch_boundary(demo_frame, color=(0, 255, 0), thickness=3)
        
        # Simulate ball position (moving across screen for demo)
        ball_x = int(width * (0.3 + 0.4 * np.sin(frame_idx / 30)))
        ball_y = int(height * 0.5)
        ball_pos = np.array([ball_x, ball_y], dtype=float)
        
        # Update trajectory
        trajectory.update(ball_pos)
        
        # Draw trajectory
        demo_frame = trajectory.draw_history(demo_frame, color=(255, 0, 255))
        demo_frame = trajectory.draw_trajectory(demo_frame, color=(0, 255, 255))
        
        # Draw simulated ball
        ball_box = np.array([ball_x - 20, ball_y - 20, ball_x + 20, ball_y + 20])
        demo_frame = annotator.draw_ball_bbox(demo_frame, ball_box, confidence=0.87)
        
        # Calculate ball speed
        ball_speed = speed_calc.calculate_speed(9999, ball_pos, frame_idx)
        if ball_speed:
            demo_frame = annotator.draw_speed(demo_frame, ball_box, ball_speed, color=(0, 255, 0))
        
        # Simulate some players with team colors
        for i in range(4):
            # Random player positions
            px = int(width * (0.2 + i * 0.2))
            py = int(height * 0.6)
            player_box = np.array([px - 30, py - 60, px + 30, py + 60])
            
            team_id = i % 2  # Alternate teams
            tracker_id = i + 1
            
            # Draw player
            demo_frame = annotator.draw_team_bbox(demo_frame, player_box, team_id, tracker_id)
            
            # Calculate player speed
            player_pos = np.array([px, py], dtype=float)
            player_speed = speed_calc.calculate_speed(tracker_id, player_pos, frame_idx)
            if player_speed:
                demo_frame = annotator.draw_speed(demo_frame, player_box, player_speed)
        
        # Draw analytics overlay
        demo_frame = annotator.draw_analytics_overlay(demo_frame, frame_idx, fps)
        
        # Write frame
        out.write(demo_frame)
        
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{max_frames}")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Done! Processed {frame_idx} frames")
    print(f"📹 Output saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick analytics demo")
    parser.add_argument("--video", default="demo_clip.mp4", help="Input video")
    parser.add_argument("--output", default="output/quick_demo.mp4", help="Output video")
    parser.add_argument("--frames", type=int, default=300, help="Max frames to process")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 PRODUCTION ANALYTICS DEMO")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Pitch boundary detection (green field)")
    print("  ✓ Team-colored bounding boxes (Red vs Blue)")
    print("  ✓ Speed calculation (km/h)")
    print("  ✓ Ball trajectory prediction")
    print("  ✓ Analytics overlay (Frame, FPS, etc.)")
    print("\n" + "="*60 + "\n")
    
    demo_analytics(args.video, args.output, args.frames)
