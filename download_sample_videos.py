#!/usr/bin/env python3
"""
Download sample surveillance videos for multi-camera tracking testing.
Uses OpenCV's sample videos and MOTChallenge dataset samples.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import urllib.request
import tarfile

# Sample video URLs from MOTChallenge dataset
MOT17_URLS = [
    "https://motchallenge.net/sequenceVideos/MOT17-13-FRCNN-raw.webm",
    "https://motchallenge.net/sequenceVideos/MOT17-11-FRCNN-raw.webm"
]

def download_file(url: str, output_path: str) -> bool:
    """Download a file from URL."""
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_tar(tar_path: str, output_dir: str) -> bool:
    """Extract a tar.gz file."""
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
        return True
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False

def create_sample_video(output_path: str, width: int = 640, height: int = 480, fps: int = 30, duration: int = 10) -> bool:
    """Create a sample video with moving shapes."""
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i in range(fps * duration):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw a moving rectangle
            x = int((i / (fps * 2)) * width) % (width - 100)
            cv2.rectangle(frame, (x, 100), (x + 100, 200), (0, 255, 0), -1)
            
            # Draw a moving circle
            y = int((i / fps) * 100) % (height - 100)
            cv2.circle(frame, (width // 4, y + 50), 30, (255, 0, 0), -1)
            
            out.write(frame)
        
        out.release()
        return True
    except Exception as e:
        print(f"Error creating sample video: {e}")
        return False

def main():
    # Create output directory
    output_dir = Path("data/sample_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic sample videos
    print("Creating sample videos...")
    sample_videos = [
        (output_dir / "camera1.mp4", 640, 480, 30, 10),
        (output_dir / "camera2.mp4", 640, 480, 30, 10)
    ]
    
    for i, (path, w, h, fps, dur) in enumerate(sample_videos, 1):
        print(f"Creating sample video {i}/{len(sample_videos)}...")
        if create_sample_video(str(path), w, h, fps, dur):
            print(f"  ✓ Created {path}")
        else:
            print(f"  ✗ Failed to create {path}")
    
    # Download MOTChallenge videos if needed
    print("\nDownloading sample videos from MOTChallenge...")
    mot_videos = []
    
    for i, url in enumerate(MOT17_URLS, 1):
        output_path = output_dir / f"mot17_{i}.mp4"
        if not output_path.exists():
            if download_file(url, str(output_path)):
                print(f"  ✓ Downloaded {output_path}")
                mot_videos.append(output_path)
            else:
                print(f"  ✗ Failed to download {url}")
        else:
            print(f"  ✓ Using existing {output_path}")
            mot_videos.append(output_path)
    
    print("\nSetup complete!")
    print(f"Sample videos available in: {output_dir.absolute()}")
    print("\nYou can now use these videos for testing the multi-camera tracking system.")
    print("Example usage:")
    print(f"  python multi_camera_tracking/main.py --videos {output_dir}/camera1.mp4 {output_dir}/camera2.mp4 --output outputs")

if __name__ == "__main__":
    main()
