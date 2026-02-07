"""
Standalone demo - Shows analytics features with ZERO dependencies.
Pure OpenCV + NumPy only.
"""
import cv2
import numpy as np
from collections import deque


print("\n" + "="*70)
print("🚀 FOOTBALL ANALYTICS DEMO - STANDALONE")
print("="*70)
print("\nFeatures Demonstrated:")
print("  ✅ Team-colored bounding boxes (Red vs Blue)")
print("  ✅ Speed calculation and display (km/h)")
print("  ✅ Ball trajectory prediction with history")
print("  ✅ Pitch boundary visualization")
print("  ✅ Analytics overlay (Frame #, FPS)")
print("="*70 + "\n")


def main(video_path, output_path, max_frames=200):
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Input: {video_path}")
    print(f"📐 Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"💾 Output: {output_path}\n")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Ball trajectory history
    ball_history = deque(maxlen=15)
    
    # Player positions for speed calc
    player_history = {}
    
    # Team colors (BGR)
    TEAM_COLORS = {
        0: (0, 0, 255),    # Team 1: Red
        1: (255, 0, 0),    # Team 2: Blue
    }
    BALL_COLOR = (0, 255, 0)  # Green
    
    frame_idx = 0
    print("⚙️  Processing frames...")
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated = frame.copy()
        
        # Draw pitch boundary (simple green rectangle)
        pitch_margin = 50
        cv2.rectangle(
            annotated,
            (pitch_margin, pitch_margin),
            (width - pitch_margin, height - pitch_margin),
            (0, 255, 0),
            3
        )
        
        # Simulate ball position (moving in pattern)
        ball_x = int(width * (0.3 + 0.4 * np.sin(frame_idx / 30.0)))
        ball_y = int(height * (0.4 + 0.2 * np.cos(frame_idx / 20.0)))
        ball_pos = (ball_x, ball_y)
        
        # Update ball history
        ball_history.append(ball_pos)
        
        # Draw ball trajectory history (purple trail)
        if len(ball_history) > 1:
            for i in range(1, len(ball_history)):
                alpha = i / len(ball_history)
                thickness = max(1, int(3 * alpha))
                color_intensity = int(255 * alpha)
                trail_color = (color_intensity, 0, 255 - color_intensity)  # Purple
                
                cv2.line(annotated, ball_history[i-1], ball_history[i], trail_color, thickness)
        
        # Predict trajectory (simple linear extrapolation)
        if len(ball_history) >= 3:
            # Estimate velocity
            velocity = np.array(ball_history[-1], dtype=float) - np.array(ball_history[-3], dtype=float)
            velocity /= 2  # Average over 2 frames
            
            # Draw predicted path
            current_pos = np.array(ball_pos, dtype=float)
            for step in range(10):
                next_pos = current_pos + velocity
                
                alpha = 1.0 - (step / 10.0)
                pt1 = tuple(current_pos.astype(int))
                pt2 = tuple(next_pos.astype(int))
                
                color = tuple(int(c * alpha) for c in (0, 255, 255))  # Fading cyan
                cv2.line(annotated, pt1, pt2, color, 2)
                
                current_pos = next_pos
        
        # Draw ball
        ball_bbox = (ball_x - 15, ball_y - 15, ball_x + 15, ball_y + 15)
        cv2.rectangle(annotated, (ball_bbox[0], ball_bbox[1]), (ball_bbox[2], ball_bbox[3]), BALL_COLOR, 3)
        cv2.putText(annotated, "Ball", (ball_bbox[0], ball_bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, BALL_COLOR, 2)
        
        # Calculate ball speed
        if len(ball_history) >= 3:
            dist = np.linalg.norm(np.array(ball_history[-1]) - np.array(ball_history[-3]))
            time_s = 2 / fps  # 2 frames
            speed_px_per_s = dist / time_s
            # Rough calibration: assume 800px = 100m pitch
            meters_per_px = 100.0 / (width * 0.8)
            speed_ms = speed_px_per_s * meters_per_px
            speed_kmh = speed_ms * 3.6
            
            cv2.putText(annotated, f"{speed_kmh:.1f} km/h", (ball_bbox[0], ball_bbox[3] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        # Simulate players
        for i in range(6):
            # Player positions (moving slowly)
            px = int(width * (0.15 + i * 0.15 + 0.02 * np.sin((frame_idx + i*10) / 50.0)))
            py = int(height * (0.55 + 0.1 * np.cos((frame_idx + i*15) / 40.0)))
            
            team_id = i % 2
            color = TEAM_COLORS[team_id]
            
            # Draw player box
            player_bbox = (px - 25, py - 50, px + 25, py + 50)
            cv2.rectangle(annotated, (player_bbox[0], player_bbox[1]), 
                         (player_bbox[2], player_bbox[3]), color, 2)
            
            # Tracker ID
            cv2.putText(annotated, f"ID:{i+1}", (player_bbox[0], player_bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate player speed
            if i not in player_history:
                player_history[i] = deque(maxlen=5)
            
            player_history[i].append((px, py))
            
            if len(player_history[i]) >= 3:
                p1 = np.array(player_history[i][-1])
                p2 = np.array(player_history[i][-3])
                dist = np.linalg.norm(p1 - p2)
                time_s = 2 / fps
                speed_px_per_s = dist / time_s
                meters_per_px = 100.0 / (width * 0.8)
                speed_ms = speed_px_per_s * meters_per_px
                speed_kmh = speed_ms * 3.6
                
                cv2.putText(annotated, f"{speed_kmh:.1f} km/h", 
                           (player_bbox[0], player_bbox[3] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Analytics overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (250, 80), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
        
        cv2.putText(annotated, f"Frame: {frame_idx}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(annotated)
        
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"  ✓ Frame {frame_idx}/{max_frames}")
    
    cap.release()
    out.release()
    
    print(f"\n✅ SUCCESS! Processed {frame_idx} frames")
    print(f"📹 Output video: {output_path}")
    print("\n" + "="*70)
    print("You can now view the video with:")
    print(f"  • VLC/Windows Media Player")
    print(f"  • Or upload to dashboard: streamlit run rl_dashboard.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="demo_clip.mp4")
    parser.add_argument("--output", default="output/standalone_demo.mp4")
    parser.add_argument("--frames", type=int, default=200)
    
    args = parser.parse_args()
    
    main(args.video, args.output, args.frames)
