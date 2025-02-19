import cv2
import os
import numpy as np
from pathlib import Path

def process_video(video_path, output_dir, interval=1):
    """
    Load video, extract frames at specified interval, rotate them 180 degrees and save
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        interval: Interval in seconds between frames to extract
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Calculate frame interval
    frame_interval = int(fps * interval)
    
    frame_number = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame at specified interval
        if frame_number % frame_interval == 0:
            # Rotate frame 180 degrees
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Generate output filename
            timestamp = int((frame_number / fps) * 1000)
            output_path = output_dir / f"frame_{timestamp:08d}ms.jpg"
            
            # Save the frame
            cv2.imwrite(str(output_path), rotated_frame)
            saved_count += 1
            
            print(f"Saved frame at {timestamp:08d}ms to {output_path}")
            
        frame_number += 1
    
    # Release resources
    cap.release()
    print(f"\nProcessing complete. Saved {saved_count} frames.")

def main():
    # Configuration
    video_path = "/app/data/route28/route28-0218-1.mp4"  # Change this to your video path
    output_dir = "/app/data/route28/route28-0218-1/"  # Change this to your desired output directory
    interval = 0.5  # Interval in seconds
    
    # Process video
    process_video(video_path, output_dir, interval)

if __name__ == "__main__":
    main()
