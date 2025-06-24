"""
Video synchronization handler for multi-camera tracking.
Ensures frames from multiple videos are properly aligned in time.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Container for video metadata and capture object."""
    path: str
    cap: cv2.VideoCapture
    width: int
    height: int
    fps: float
    frame_count: int
    frame_idx: int = 0
    
    @classmethod
    def from_path(cls, video_path: str) -> 'VideoInfo':
        """Create a VideoInfo instance from a video file path."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return cls(
            path=video_path,
            cap=cap,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count
        )
    
    def release(self) -> None:
        """Release video capture resources."""
        if self.cap.isOpened():
            self.cap.release()
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the video."""
        if self.frame_idx >= self.frame_count:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_idx += 1
        return ret, frame


class VideoSynchronizer:
    """
    Handles synchronization of multiple video streams.
    Ensures frames from different cameras are aligned in time.
    """
    
    def __init__(self, video_path1: str, video_path2: str, max_frame_skip: int = 5, frame_skip: int = 0):
        """
        Initialize the video synchronizer with two video paths.
        
        Args:
            video_path1: Path to the first video file
            video_path2: Path to the second video file
            max_frame_skip: Maximum number of frames to skip when synchronizing
            frame_skip: Number of frames to skip between processed frames (0 = no skipping)
        """
        self.video1 = VideoInfo.from_path(video_path1)
        self.video2 = VideoInfo.from_path(video_path2)
        
        # Verify video properties
        self._validate_videos()
        
        # Synchronization parameters
        self.max_frame_skip = max(1, max_frame_skip)
        self.frame_skip = max(0, frame_skip)
        
        # Current frame index (aligned between videos)
        self.current_frame_idx = 0
        
        # Frame timestamp tracking
        self.last_timestamp1 = 0
        self.last_timestamp2 = 0
        
        # Debug info
        self.sync_offsets = []
        self.sync_attempts = 0
    
    def _validate_videos(self) -> None:
        """Validate that the videos have compatible properties."""
        # Check if FPS is similar (within 5%)
        fps_ratio = self.video1.fps / self.video2.fps
        if not 0.95 <= fps_ratio <= 1.05:
            print(f"Warning: Video FPS differs significantly: "
                  f"{self.video1.fps:.2f} vs {self.video2.fps:.2f}")
        
        # Check if frame counts are similar (within 5% of the shorter video)
        min_frames = min(self.video1.frame_count, self.video2.frame_count)
        max_frames = max(self.video1.frame_count, self.video2.frame_count)
        if (max_frames - min_frames) / min_frames > 0.05:
            print(f"Warning: Video lengths differ significantly: "
                  f"{self.video1.frame_count} vs {self.video2.frame_count} frames")
    
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the next synchronized frame pair from both videos.
        
        Returns:
            Tuple of (frame1, frame2) if successful, None if either video has ended
        """
        # Skip frames if needed
        if self.frame_skip > 0 and self.current_frame_idx > 0:
            for _ in range(self.frame_skip):
                ret1, _ = self.video1.read_frame()
                ret2, _ = self.video2.read_frame()
                self.current_frame_idx += 1
                if not (ret1 and ret2):
                    return None
        
        # Read initial frames
        ret1, frame1 = self.video1.read_frame()
        ret2, frame2 = self.video2.read_frame()
        
        # If either video has ended, return None
        if not (ret1 and ret2):
            return None
            
        # Calculate timestamps based on frame index and FPS
        timestamp1 = (self.video1.frame_idx - 1) / self.video1.fps * 1000  # in ms
        timestamp2 = (self.video2.frame_idx - 1) / self.video2.fps * 1000  # in ms
        
        # Simple synchronization logic - try to align frames within max_frame_skip
        frames_skipped = 0
        time_threshold = 1000 / max(self.video1.fps, self.video2.fps)  # One frame time of the faster video
        
        while abs(timestamp1 - timestamp2) > time_threshold and frames_skipped < self.max_frame_skip:
            self.sync_attempts += 1
            if timestamp1 < timestamp2:
                ret1, frame1 = self.video1.read_frame()
                if not ret1:
                    return None
                timestamp1 = (self.video1.frame_idx - 1) / self.video1.fps * 1000
            else:
                ret2, frame2 = self.video2.read_frame()
                if not ret2:
                    return None
                timestamp2 = (self.video2.frame_idx - 1) / self.video2.fps * 1000
            frames_skipped += 1
        
        if frames_skipped > 0:
            self.sync_offsets.append(abs(timestamp1 - timestamp2))
        
        # Update frame counter and timestamps
        self.current_frame_idx += 1
        self.last_timestamp1 = timestamp1
        self.last_timestamp2 = timestamp2
        
        return frame1, frame2
    
    def seek(self, frame_idx: int) -> None:
        """
        Seek to a specific frame in both videos.
        
        Args:
            frame_idx: Frame index to seek to
        """
        if frame_idx < 0 or frame_idx >= min(self.video1.frame_count, self.video2.frame_count):
            raise ValueError(f"Frame index {frame_idx} out of range")
        
        self.video1.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.video2.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.video1.frame_idx = frame_idx
        self.video2.frame_idx = frame_idx
        self.current_frame_idx = frame_idx
    
    def get_current_time(self) -> float:
        """Get the current timestamp in seconds."""
        return self.current_frame_idx / self.video1.fps
    
    def get_progress(self) -> float:
        """Get the current progress as a fraction [0, 1]."""
        min_frames = min(self.video1.frame_count, self.video2.frame_count)
        return self.current_frame_idx / min_frames
    
    def release(self) -> None:
        """Release all video resources and print synchronization stats."""
        self.video1.release()
        self.video2.release()
        
        # Print synchronization statistics if we did any synchronization
        if self.sync_attempts > 0 and len(self.sync_offsets) > 0:
            avg_offset = sum(self.sync_offsets) / len(self.sync_offsets)
            print(f"\nSynchronization statistics:")
            print(f"  - Total sync attempts: {self.sync_attempts}")
            print(f"  - Average time offset: {avg_offset:.2f} ms")
            print(f"  - Max frame skip: {self.max_frame_skip}")
            print(f"  - Frame skip: {self.frame_skip}")
            print(f"  - Total frames processed: {self.current_frame_idx}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    @property
    def width1(self) -> int:
        """Get the width of the first video."""
        return self.video1.width
    
    @property
    def height1(self) -> int:
        """Get the height of the first video."""
        return self.video1.height
    
    @property
    def width2(self) -> int:
        """Get the width of the second video."""
        return self.video2.width
    
    @property
    def height2(self) -> int:
        """Get the height of the second video."""
        return self.video2.height
    
    @property
    def fps(self) -> float:
        """Get the frame rate of the videos (average if different)."""
        return (self.video1.fps + self.video2.fps) / 2
    
    @property
    def frame_count1(self) -> int:
        """Get the frame count of the first video."""
        return self.video1.frame_count
    
    @property
    def frame_count2(self) -> int:
        """Get the frame count of the second video."""
        return self.video2.frame_count


def get_video_properties(video_path: str) -> Dict[str, Any]:
    """
    Get basic properties of a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video properties
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'path': video_path,
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration_sec': duration,
        'resolution': f"{width}x{height}",
        'codec': Path(video_path).suffix[1:].upper()
    }
