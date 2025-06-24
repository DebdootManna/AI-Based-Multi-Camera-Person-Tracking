"""
Global ID manager for maintaining consistent person identities across multiple cameras.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import time


class GlobalIDManager:
    """
    Manages global IDs for tracked objects across multiple camera views.
    
    This class is responsible for maintaining consistent identities of people
    across different camera views by matching detections based on appearance
    features and temporal consistency.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 max_age: float = 5.0,
                 min_similarity: float = 0.5,
                 max_missed: int = 10,
                 min_confidence: float = 0.7):
        """
        Initialize the Global ID Manager.
        
        Args:
            similarity_threshold: Minimum similarity score to match tracks
            max_age: Maximum age (in seconds) to keep a track without updates
            min_similarity: Minimum similarity score to consider for matching
            max_missed: Maximum number of consecutive misses before removing a track
            min_confidence: Minimum confidence score for a detection to be considered
        """
        self.similarity_threshold = similarity_threshold
        self.max_age = max_age
        self.min_similarity = min_similarity
        self.max_missed = max_missed
        self.min_confidence = min_confidence
        
        # Track storage
        self.next_global_id = 1
        self.tracks: Dict[int, Dict] = {}  # global_id -> track info
        self.last_update_time = time.time()
        
        # Camera-specific track info
        self.cam_tracks: Dict[int, Dict[int, int]] = {}  # cam_id -> {local_id: global_id}
        
        # Appearance features for ReID
        self.appearance_features: Dict[int, List[np.ndarray]] = {}  # global_id -> [features]
        self.max_features_per_id = 10  # Maximum number of features to store per ID
    
    def update(self, 
               frame: np.ndarray, 
               tracked_objects: List[Dict],
               camera_id: int = 0) -> List[int]:
        """
        Update tracks with new detections from a camera.
        
        Args:
            frame: Current video frame (for visualization)
            tracked_objects: List of tracked objects with detections
            camera_id: ID of the camera providing the detections
            
        Returns:
            List of global IDs for the input detections
        """
        current_time = time.time()
        
        # Initialize camera tracks if not exists
        if camera_id not in self.cam_tracks:
            self.cam_tracks[camera_id] = {}
        
        # Get active global IDs for this camera
        active_global_ids = []
        
        # Process each detection
        for obj in tracked_objects:
            if obj['confidence'] < self.min_confidence:
                continue
                
            bbox = obj['bbox']
            local_id = obj['track_id']
            feature = obj.get('feature', None)
            
            # Check if we've seen this local ID before
            if local_id in self.cam_tracks[camera_id]:
                # Update existing track
                global_id = self.cam_tracks[camera_id][local_id]
                if global_id in self.tracks:
                    self._update_track(global_id, bbox, feature, current_time, camera_id)
            else:
                # New detection, try to match with existing global IDs
                global_id = self._match_detection(bbox, feature, camera_id, current_time)
                if global_id is None:
                    # Assign new global ID
                    global_id = self._create_new_track(bbox, feature, camera_id, current_time)
                
                # Update camera tracks
                self.cam_tracks[camera_id][local_id] = global_id
            
            active_global_ids.append(global_id)
        
        # Clean up old tracks
        self._cleanup_tracks(current_time)
        
        return active_global_ids
    
    def _match_detection(self, 
                        bbox: np.ndarray, 
                        feature: Optional[np.ndarray],
                        camera_id: int,
                        current_time: float) -> Optional[int]:
        """
        Match a detection with existing global tracks.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            feature: Appearance feature vector
            camera_id: ID of the camera
            current_time: Current timestamp
            
        Returns:
            Matched global ID, or None if no good match
        """
        if feature is None:
            return None
            
        best_score = self.min_similarity
        best_match = None
        
        # Convert feature to numpy array if needed
        if not isinstance(feature, np.ndarray):
            feature = np.array(feature, dtype=np.float32)
        
        # Normalize feature
        feature = feature / (np.linalg.norm(feature) + 1e-6)
        
        # Compare with existing tracks
        for global_id, track in self.tracks.items():
            # Skip if this track was recently seen from the same camera
            if track['last_camera'] == camera_id and \
               (current_time - track['last_update']) < 1.0:  # 1 second cooldown
                continue
            
            # Skip if track is too old
            if (current_time - track['last_update']) > self.max_age:
                continue
            
            # Calculate appearance similarity
            if track['features']:
                # Use the most recent feature for matching
                ref_feature = track['features'][-1]
                similarity = np.dot(feature, ref_feature)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = global_id
        
        return best_match if best_score >= self.similarity_threshold else None
    
    def _create_new_track(self, 
                         bbox: np.ndarray, 
                         feature: Optional[np.ndarray],
                         camera_id: int,
                         current_time: float) -> int:
        """
        Create a new global track.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            feature: Appearance feature vector
            camera_id: ID of the camera
            current_time: Current timestamp
            
        Returns:
            New global ID
        """
        global_id = self.next_global_id
        self.next_global_id += 1
        
        # Initialize track
        self.tracks[global_id] = {
            'id': global_id,
            'bbox': bbox,
            'features': [],
            'last_update': current_time,
            'last_camera': camera_id,
            'missed_frames': 0,
            'total_detections': 1
        }
        
        # Store appearance feature if available
        if feature is not None:
            if not isinstance(feature, np.ndarray):
                feature = np.array(feature, dtype=np.float32)
            feature = feature / (np.linalg.norm(feature) + 1e-6)
            self.tracks[global_id]['features'].append(feature)
        
        return global_id
    
    def _update_track(self, 
                     global_id: int, 
                     bbox: np.ndarray, 
                     feature: Optional[np.ndarray],
                     current_time: float,
                     camera_id: int) -> None:
        """
        Update an existing global track.
        
        Args:
            global_id: Global ID of the track to update
            bbox: New bounding box
            feature: New appearance feature
            current_time: Current timestamp
            camera_id: ID of the camera
        """
        if global_id not in self.tracks:
            return
            
        track = self.tracks[global_id]
        
        # Update track state
        track['bbox'] = bbox
        track['last_update'] = current_time
        track['last_camera'] = camera_id
        track['missed_frames'] = 0
        track['total_detections'] += 1
        
        # Update appearance feature
        if feature is not None:
            if not isinstance(feature, np.ndarray):
                feature = np.array(feature, dtype=np.float32)
            feature = feature / (np.linalg.norm(feature) + 1e-6)
            
            # Add to features list, keeping only the most recent ones
            track['features'].append(feature)
            if len(track['features']) > self.max_features_per_id:
                track['features'].pop(0)
    
    def _cleanup_tracks(self, current_time: float) -> None:
        """
        Remove tracks that are too old or have too many missed detections.
        
        Args:
            current_time: Current timestamp
        """
        to_remove = []
        
        for global_id, track in self.tracks.items():
            # Check if track is too old
            if (current_time - track['last_update']) > self.max_age:
                to_remove.append(global_id)
                continue
            
            # Check for too many missed frames
            if track['missed_frames'] > self.max_missed:
                to_remove.append(global_id)
        
        # Remove old tracks
        for global_id in to_remove:
            # Remove from camera tracks
            for cam_id in self.cam_tracks:
                local_ids = [lid for lid, gid in self.cam_tracks[cam_id].items() if gid == global_id]
                for local_id in local_ids:
                    del self.cam_tracks[cam_id][local_id]
            
            # Remove from tracks
            if global_id in self.tracks:
                del self.tracks[global_id]
    
    def get_global_ids(self, camera_id: int) -> Dict[int, int]:
        """
        Get the mapping from local track IDs to global IDs for a camera.
        
        Args:
            camera_id: ID of the camera
            
        Returns:
            Dictionary mapping local track IDs to global IDs
        """
        return self.cam_tracks.get(camera_id, {})
    
    def get_all_global_ids(self) -> List[int]:
        """
        Get all active global IDs.
        
        Returns:
            List of active global IDs
        """
        return list(self.tracks.keys())
    
    def get_track_info(self, global_id: int) -> Optional[Dict]:
        """
        Get information about a specific global track.
        
        Args:
            global_id: Global ID of the track
            
        Returns:
            Track information dictionary, or None if not found
        """
        return self.tracks.get(global_id)
    
    def get_active_tracks(self) -> Dict[int, Dict]:
        """
        Get information about all active tracks.
        
        Returns:
            Dictionary mapping global IDs to track information
        """
        return self.tracks.copy()
    
    def reset(self) -> None:
        """Reset the global ID manager, clearing all tracks."""
        self.tracks.clear()
        self.cam_tracks.clear()
        self.next_global_id = 1
        self.last_update_time = time.time()
