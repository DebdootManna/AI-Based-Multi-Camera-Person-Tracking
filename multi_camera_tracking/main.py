#!/usr/bin/env python3
"""
Multi-Camera Person Tracking System

This script performs person detection, tracking, and cross-camera re-identification
on synchronized video streams using YOLOv8, StrongSORT, and OSNet ReID.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.absolute()))

from multi_camera_tracking.detector.yolov8_detector import YOLOv8Detector
from multi_camera_tracking.tracker.multi_tracker import MultiCameraTracker
from multi_camera_tracking.reid.reid_model import ReIDModel
from multi_camera_tracking.utils.visualize import draw_boxes
from multi_camera_tracking.utils.sync_handler import VideoSynchronizer
from multi_camera_tracking.utils.global_id_manager import GlobalIDManager


def parse_arguments():
    parser = argparse.ArgumentParser(description='Multi-Camera Person Tracking')
    
    # Required arguments
    parser.add_argument('--video1', type=str, required=True, 
                      help='Path to first video file (CAM1)')
    parser.add_argument('--video2', type=str, required=True,
                      help='Path to second video file (CAM2)')
    
    # Model options
    parser.add_argument('--model_size', type=str, default='x',
                      choices=['n', 's', 'm', 'l', 'x', 'x6'],
                      help='YOLOv8 model size: n(nano), s(small), m(medium), l(large), x(xlarge), x6(xlarge6)')
    
    # Processing options
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save output files')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                      help='Confidence threshold for detection (0.0-1.0)')
    parser.add_argument('--iou_threshold', type=float, default=0.4,
                      help='IoU threshold for NMS (0.0-1.0)')
    parser.add_argument('--reid_threshold', type=float, default=0.7,
                      help='Cosine similarity threshold for ReID matching (0.0-1.0)')
    parser.add_argument('--max_iou_distance', type=float, default=0.6,
                      help='Maximum IoU distance for tracking (0.0-1.0)')
    parser.add_argument('--max_cosine_distance', type=float, default=0.3,
                      help='Maximum cosine distance for appearance matching (0.0-1.0)')
    parser.add_argument('--max_age', type=int, default=45,
                      help='Maximum number of frames to keep a track without updates')
    parser.add_argument('--n_init', type=int, default=2,
                      help='Number of consecutive detections before a track is confirmed')
    parser.add_argument('--min_global_confidence', type=float, default=0.6,
                      help='Minimum confidence for global ID assignment (0.0-1.0)')
    
    # Output options
    parser.add_argument('--visualize', action='store_true', default=True,
                      help='Show real-time visualization')
    parser.add_argument('--save_video', action='store_true', default=True,
                      help='Save output videos')
    parser.add_argument('--save_thumbnails', action='store_true', default=True,
                      help='Save thumbnails of detected objects')
    parser.add_argument('--debug', action='store_true', default=True,
                      help='Enable debug logging and visualization')
    parser.add_argument('--show_detections', action='store_true', default=True,
                      help='Show detection bounding boxes')
    parser.add_argument('--show_tracks', action='store_true', default=True,
                      help='Show tracking information')
    parser.add_argument('--show_global_ids', action='store_true', default=True,
                      help='Show global ID assignments')
    parser.add_argument('--show_fps', action='store_true', default=True,
                      help='Show FPS counter')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to run on (cpu, mps, cuda). Auto-detects if None')
    
    # Performance options
    parser.add_argument('--frame_skip', type=int, default=0,
                      help='Number of frames to skip between processing (for faster processing)')
    parser.add_argument('--max_frames', type=int, default=0,
                      help='Maximum number of frames to process (0 for all)')
    
    return parser.parse_args()


def setup_device(device_str=None):
    """Setup device for inference, preferring MPS on Apple Silicon if available."""
    if device_str is None:
        if torch.backends.mps.is_available():
            device_str = 'mps'
        elif torch.cuda.is_available():
            device_str = 'cuda'
        else:
            device_str = 'cpu'
    
    print(f"Using device: {device_str.upper()}")
    return torch.device(device_str)


def initialize_models(args, device):
    """Initialize all required models."""
    print("Initializing models...")
    
    # Model name based on size
    model_name = f'yolov8{args.model_size}.pt'
    print(f"Using YOLOv8 model: {model_name}")
    
    # Initialize detector with specified parameters
    detector = YOLOv8Detector(
        model_name=model_name,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=device
    )
    
    # Initialize ReID model
    reid_model = ReIDModel(device=device)
    
    # Initialize tracker with optimized parameters
    tracker = MultiCameraTracker(
        detector=detector,
        reid_model=reid_model,
        device=device,
        max_iou_distance=args.max_iou_distance,
        max_cosine_distance=args.max_cosine_distance,
        max_age=args.max_age,
        n_init=args.n_init,
        min_global_confidence=args.min_global_confidence
    )
    
    return detector, reid_model, tracker


def process_videos(args, device):
    """
    Main processing loop for the multi-camera tracking system.
    
    Args:
        args: Command line arguments
        device: Torch device to run models on
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    detector, reid_model, tracker = initialize_models(args, device)
    
    # Initialize video synchronizer
    print("Initializing video streams...")
    print(f"Video 1: {os.path.abspath(args.video1)}")
    print(f"Video 2: {os.path.abspath(args.video2)}")
    
    video_sync = VideoSynchronizer(args.video1, args.video2, max_frame_skip=5)
    
    # Get video properties from the first video
    video1 = cv2.VideoCapture(args.video1)
    fps = video1.get(cv2.CAP_PROP_FPS)
    width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    video1.release()
    
    # Create video writers if needed
    output_video1 = None
    output_video2 = None
    if args.save_video:
        output_path1 = os.path.join(output_dir, 'output_cam1.mp4')
        output_path2 = os.path.join(output_dir, 'output_cam2.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video1 = cv2.VideoWriter(output_path1, fourcc, fps, (width, height))
        output_video2 = cv2.VideoWriter(output_path2, fourcc, fps, (width, height))
    
    # Create debug directory
    debug_dir = os.path.join(output_dir, 'debug')
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)
    
    # Create thumbnails directory
    thumbnails_dir = os.path.join(output_dir, 'thumbnails')
    if args.save_thumbnails:
        os.makedirs(thumbnails_dir, exist_ok=True)
    
    # Initialize statistics
    stats = {
        'total_frames': 0,
        'processing_time': 0,
        'fps': 0,
        'total_detections': 0,
        'total_tracks': 0,
        'unique_global_ids': set(),
        'detection_stats': [],
        'tracking_stats': []
    }
    
    # Color palette for visualization
    color_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
        (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0)
    ]
    
    # Process frames
    print(f"Processing frames...")
    start_time = time.time()
    frame_idx = 0
    processed_frames = 0
    
    # Calculate total frames based on frame_skip
    total_frames = frame_count
    if args.frame_skip > 0:
        total_frames = frame_count // (args.frame_skip + 1)
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            frame_start_time = time.time()
            
            # Get synchronized frames
            frames = video_sync.get_frames()
            if frames is None:
                break
                
            frame1, frame2 = frames
            frame_idx += 1
            
            # Skip frames if needed
            if args.frame_skip > 0 and (frame_idx - 1) % (args.frame_skip + 1) != 0:
                pbar.update(1)
                continue
                
            processed_frames += 1
            
            # Update progress bar
            pbar.update(1)
            
            # Check for max frames
            if processed_frames >= args.max_frames if hasattr(args, 'max_frames') and args.max_frames else False:
                break
            
            # Process frame 1
            detections1 = detector(frame1)
            # Update tracker with frame and camera ID
            tracks1 = tracker.update(frame1, camera_id=0)
            
            # Process frame 2
            detections2 = detector(frame2)
            # Update tracker with frame and camera ID
            tracks2 = tracker.update(frame2, camera_id=1)
            
            # Calculate frame processing time
            frame_time = time.time() - frame_start_time
            
            # Update statistics
            stats['total_frames'] += 1
            stats['detection_stats'].append({
                'frame': frame_idx,
                'camera1_detections': len(detections1[0]),
                'camera2_detections': len(detections2[0]),
                'frame_time': frame_time
            })
            stats['tracking_stats'].append({
                'frame': frame_idx,
                'camera1_tracks': len(tracks1),
                'camera2_tracks': len(tracks2),
                'global_ids': len(set(t['global_id'] for t in tracks1 + tracks2 if t['global_id'] is not None))
            })
            
            stats['total_detections'] += len(detections1[0]) + len(detections2[0])
            stats['total_tracks'] += len(tracks1) + len(tracks2)
            stats['unique_global_ids'].update([t['global_id'] for t in tracks1 + tracks2 if t['global_id'] is not None])
            
            # Draw results if needed
            if args.visualize or args.save_video or args.save_thumbnails or args.debug:
                frame1_disp = frame1.copy()
                frame2_disp = frame2.copy()
                
                # Draw detections if enabled
                if args.show_detections:
                    for det_box in detections1[0]:
                        x1, y1, x2, y2 = map(int, det_box[:4])
                        cv2.rectangle(frame1_disp, (x1, y1), (x2, y2), (0, 165, 255), 1)  # Orange for detections
                    
                    for det_box in detections2[0]:
                        x1, y1, x2, y2 = map(int, det_box[:4])
                        cv2.rectangle(frame2_disp, (x1, y1), (x2, y2), (0, 165, 255), 1)  # Orange for detections
                
                # Draw tracks if enabled
                if args.show_tracks:
                    for track in tracks1 + tracks2:
                        try:
                            bbox = track.get('bbox', [0, 0, 0, 0])
                            track_id = track.get('track_id', -1)
                            global_id = track.get('global_id', None)
                            confidence = track.get('confidence', 1.0)  # Default to 1.0 if not present
                            camera_id = track.get('camera_id', 0)  # Default to camera 0 if not present
                        except (KeyError, AttributeError) as e:
                            print(f"Warning: Missing track information: {e}")
                            continue
                        
                        # Get the correct frame
                        frame_disp = frame1_disp if camera_id == 0 else frame2_disp
                        
                        # Get color based on global ID
                        if global_id is not None:
                            color_idx = global_id % len(color_palette)
                            color = color_palette[color_idx]
                        else:
                            color = (128, 128, 128)  # Gray for unassigned
                        
                        # Draw track boundary
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw track info
                        info = []
                        if args.show_global_ids and global_id is not None:
                            info.append(f"G:{global_id}")
                        info.append(f"T:{track_id}")
                        
                        # Add confidence if in debug mode
                        if args.debug:
                            info.append(f"{confidence:.2f}")
                        
                        # Draw text background
                        label = " ".join(info)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame_disp, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                        cv2.putText(frame_disp, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add FPS counter if enabled
                if args.show_fps:
                    fps = 1.0 / frame_time if frame_time > 0 else 0
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(frame1_disp, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame2_disp, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add camera labels
                cv2.putText(frame1_disp, "CAM 1", (width - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame2_disp, "CAM 2", (width - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Save debug frames if enabled
                if args.debug and frame_idx % 10 == 0:
                    debug_frame = np.hstack((frame1_disp, frame2_disp))
                    debug_path = os.path.join(debug_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(debug_path, debug_frame)
                
                # Save thumbnails if needed
                if args.save_thumbnails:
                    save_thumbnails(thumbnails_dir, frame1_disp, tracks1)
                    save_thumbnails(thumbnails_dir, frame2_disp, tracks2)
                
                # Show visualization
                if args.visualize:
                    # Stack frames side by side
                    vis_frame = np.hstack((frame1_disp, frame2_disp))
                    
                    # Add status bar
                    status_bar = np.zeros((30, width * 2, 3), dtype=np.uint8)
                    status_text = f"Frame: {frame_idx} | Tracks: {len(tracks1) + len(tracks2)} | " \
                                 f"Global IDs: {len(stats['unique_global_ids'])} | FPS: {1.0/frame_time:.1f}"
                    cv2.putText(status_bar, status_text, (10, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    vis_frame = np.vstack((vis_frame, status_bar))
                    
                    # Show frame
                    cv2.imshow("Multi-Camera Tracking", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):  # Pause on 'p' key
                        while True:
                            if cv2.waitKey(1) & 0xFF == ord('p'):
                                break
                
                # Write to output videos
                if args.save_video:
                    output_video1.write(frame1_disp)
                    output_video2.write(frame2_disp)
            
            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix({
                'tracks': len(tracks1) + len(tracks2),
                'global_ids': len(stats['unique_global_ids']),
                'fps': f"{1.0/frame_time:.1f}" if frame_time > 0 else 'N/A'
            })
            
            # Check for max frames
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    
    # Calculate final statistics
    end_time = time.time()
    stats['processing_time'] = end_time - start_time
    stats['fps'] = frame_idx / stats['processing_time'] if stats['processing_time'] > 0 else 0
    stats['unique_global_ids'] = list(stats['unique_global_ids'])
    
    # Save summary and statistics
    save_summary(output_dir, tracker, stats)
    
    # Save detailed statistics
    if args.debug:
        import pandas as pd
        pd.DataFrame(stats['detection_stats']).to_csv(os.path.join(output_dir, 'detection_stats.csv'), index=False)
        pd.DataFrame(stats['tracking_stats']).to_csv(os.path.join(output_dir, 'tracking_stats.csv'), index=False)
    
    # Release resources
    if output_video1 is not None:
        output_video1.release()
    if output_video2 is not None:
        output_video2.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete. Results saved to: {output_dir}")
    print(f"Processed {frame_idx} frames at {stats['fps']:.2f} FPS")
    print(f"Tracked {len(stats['unique_global_ids'])} unique identities")


def save_thumbnails(thumbnails_dir, frame, tracks):
    """Save thumbnails of detected objects.
    
    Args:
        thumbnails_dir: Directory to save thumbnails (can be str or Path)
        frame: Input frame
        tracks: List of track dictionaries
    """
    # Ensure the directory exists
    thumbnails_dir = Path(thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    
    for track in tracks:
        try:
            if track.get('global_id') is not None:
                bbox = track.get('bbox', [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                    
                x1, y1, x2, y2 = map(int, bbox)
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:  # Ensure valid crop dimensions
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:  # Ensure we have a valid image
                        timestamp = int(time.time() * 1000)  # Use milliseconds for uniqueness
                        thumb_path = thumbnails_dir / f"id_{track['global_id']}_{timestamp}.jpg"
                        cv2.imwrite(str(thumb_path), crop)
        except Exception as e:
            print(f"Warning: Failed to save thumbnail: {e}")
            continue


def save_summary(output_dir, tracker, stats):
    """
    Save tracking summary to JSON file.
    
    Args:
        output_dir: Directory to save summary
        tracker: MultiCameraTracker instance
        stats: Dictionary of tracking statistics
    """
    import json
    from datetime import datetime
    
    # Get global ID counts
    global_id_counts = tracker.get_global_id_counts()
    
    # Prepare summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'processing_time_seconds': stats.get('processing_time', 0),
        'total_frames': stats.get('frames_processed', 0),
        'processing_fps': stats.get('fps', 0),
        'total_tracked_objects': stats.get('tracked_objects', 0),
        'unique_global_ids': len(global_id_counts),
        'global_id_distribution': global_id_counts,
        'camera_ids': tracker.get_camera_ids(),
        'tracks_per_camera': {
            cam_id: len(tracks) 
            for cam_id, tracks in tracker.tracks.items()
        }
    }
    
    # Save to file
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Summary saved to: {summary_path}")


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup device
    device = setup_device(args.device)
    
    # Process videos
    process_videos(args, device)



if __name__ == "__main__":
    main()
