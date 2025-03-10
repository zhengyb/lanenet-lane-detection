import cv2
import os
import numpy as np
from pathlib import Path

def process_video(video_path, output_dir, interval=1, specific_frame=None):
    """
    Load video, extract frames at specified interval, rotate them 180 degrees and save
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        interval: Interval in seconds between frames to extract
        specific_frame: Specific frame number to extract
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if specific_frame is not None:
        if not isinstance(specific_frame, list):
            raise ValueError("specific_frame must be a list")
        if not all(isinstance(i, int) for i in specific_frame):
            raise ValueError("specific_frame must contain only integers")

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

        need_to_save = False
            
        if (specific_frame is not None):
            if frame_number in specific_frame:
                need_to_save = True
        # Process frame at specified interval
        elif frame_interval > 0 and frame_number % frame_interval == 0:
            need_to_save = True

        if need_to_save:
                # Rotate frame 180 degrees
            #rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            rotated_frame = frame
            # Generate output filename
            output_path = output_dir / f"frame_{frame_number:08d}.jpg"
            
            # Save the frame
            cv2.imwrite(str(output_path), rotated_frame)
            saved_count += 1
            
            print(f"Saved frame at {frame_number:08d} to {output_path}")
            
        frame_number += 1
    
    # Release resources
    cap.release()
    print(f"\nProcessing complete. Saved {saved_count} frames.")

def main():
    # Configuration
    video_path = "/app/data/route28/road28_66_20250306_11_12_47_Pro.mp4"  # Change this to your video path
    #output_dir = "/app/data/carla/calibration_video_imgs/"  # Change this to your desired output directory
    output_dir = video_path.replace(".mp4", "_imgs")
    interval = 0.0  # Interval in seconds
    specific_frame = [19170, 10890]

    # Process video
    process_video(video_path, output_dir, interval, specific_frame)

if __name__ == "__main__":
    main()
