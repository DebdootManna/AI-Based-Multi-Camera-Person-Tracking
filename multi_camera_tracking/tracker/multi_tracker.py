"""
Multi-camera tracker implementation using YOLOv8 for detection and StrongSORT for tracking.
Handles multiple camera views with global ID management.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, DefaultDict, Set
import numpy as np
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from multi_camera_tracking.detector.yolov8_detector import YOLOv8Detector
from multi_camera_tracking.reid.reid_model import ReIDModel
from multi_camera_tracking.tracker.strongsort.track import Track

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
CameraID = Union[int, str]
BBox = np.ndarray  # [x1, y1, x2, y2]
Feature = np.ndarray  # [D] dimensional feature vector

@dataclass
class GlobalIDManager:
    """
    Manages global IDs across multiple camera views.
    
    This class maintains a gallery of appearance features and assigns
    global IDs to tracks based on appearance similarity.
    """
    next_id: int = 1
    gallery: Dict[int, List[Feature]] = field(default_factory=dict)
    gallery_means: Dict[int, Feature] = field(default_factory=dict)
    
    def add(self, feature: Feature) -> int:
        """
        Add a new global ID with the given feature.
        
        Args:
            feature: Appearance feature vector
            
        Returns:
            Assigned global ID
        """
        global_id = self.next_id
        self.gallery[global_id] = [feature]
        self.gallery_means[global_id] = feature
        self.next_id += 1
        return global_id
    
    def update(self, global_id: int, feature: Feature) -> None:
        """
        Update an existing global ID with a new feature.
        
        Args:
            global_id: Global ID to update
            feature: New appearance feature
        """
        if global_id not in self.gallery:
            logger.warning(f"Global ID {global_id} not found in gallery")
            return
            
        self.gallery[global_id].append(feature)
        # Update mean feature using exponential moving average
        alpha = 0.1  # Learning rate
        old_mean = self.gallery_means[global_id]
        self.gallery_means[global_id] = (1 - alpha) * old_mean + alpha * feature
    
    def match(self, 
              query_feature: Feature, 
              min_confidence: float = 0.7) -> Tuple[Optional[int], float]:
        """
        Find the best matching global ID for a query feature.
        
        Args:
            query_feature: Query feature vector
            min_confidence: Minimum confidence threshold for matching
            
        Returns:
            Tuple of (best_matching_global_id, confidence) or (None, 0.0) if no match
        """
        if not self.gallery_means:
            return None, 0.0
            
        # Normalize query feature
        query_feature = query_feature / np.linalg.norm(query_feature)
        
        # Calculate cosine similarity with all gallery means
        similarities = []
        for global_id, mean_feature in self.gallery_means.items():
            mean_feature = mean_feature / np.linalg.norm(mean_feature)
            similarity = np.dot(query_feature, mean_feature)
            similarities.append((global_id, similarity))
        
        if not similarities:
            return None, 0.0
            
        # Find best match
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_id, best_similarity = similarities[0]
        
        # Apply threshold
        confidence = float(best_similarity)
        if confidence >= min_confidence:
            return best_id, confidence
        return None, confidence
    
    def remove(self, global_id: int) -> None:
        """
        Remove a global ID from the gallery.
        
        Args:
            global_id: Global ID to remove
        """
        if global_id in self.gallery:
            del self.gallery[global_id]
        if global_id in self.gallery_means:
            del self.gallery_means[global_id]
    
    def get_all_global_ids(self) -> Set[int]:
        """
        Get all active global IDs.
        
        Returns:
            Set of active global IDs
        """
        return set(self.gallery.keys())
    
    def clear(self) -> None:
        """Clear all global IDs and features."""
        self.gallery.clear()
        self.gallery_means.clear()
        self.next_id = 1


class Track:
    """Track class to store information about a tracked object across multiple cameras."""
    
    def __init__(self, 
                 track_id: int, 
                 bbox: BBox, 
                 embedding: Feature,
                 camera_id: CameraID,
                 class_id: int = 0,
                 max_age: int = 30,
                 n_init: int = 3,
                 global_id: Optional[int] = None):
        """
        Initialize a new track.
        
        Args:
            track_id: Local identifier for the track (unique within a camera)
            bbox: Bounding box in [x1, y1, x2, y2] format
            embedding: Feature embedding for the object
            camera_id: ID of the camera this track belongs to
            class_id: Class ID of the object (0 for person)
            max_age: Maximum number of consecutive misses before the track is deleted
            n_init: Number of consecutive detections before the track is confirmed
            global_id: Global ID for multi-camera tracking (assigned later if None)
        """
        self.track_id = track_id  # Local track ID (unique within camera)
        self.global_id = global_id  # Global ID (assigned by GlobalIDManager)
        self.camera_id = camera_id
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.class_id = class_id
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.max_age = max_age
        self.n_init = n_init
        self.state = 0  # 0: tentative, 1: confirmed, -1: deleted
        self.last_seen = time.time()
        
        # Store history
        self.history = deque(maxlen=100)  # Store recent bboxes
        self.history.append(self.bbox.copy())
        
        # For appearance updates
        self.features = [embedding]  # Store recent features for appearance updates
    
    def predict(self) -> None:
        """Increment age and time since update."""
        self.age += 1
        self.time_since_update += 1
    
    def update(self, bbox: BBox, embedding: Feature) -> None:
        """
        Update the track with a new detection.
        
        Args:
            bbox: New bounding box in [x1, y1, x2, y2] format
            embedding: New feature embedding
        """
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.embedding = np.asarray(embedding, dtype=np.float32)
        
        # Safe normalization with zero-norm check
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-12:  # Only normalize if norm is sufficiently large
            self.embedding = self.embedding / norm
        else:
            # If norm is zero or very small, use a small random vector
            self.embedding = np.random.randn(*self.embedding.shape).astype(np.float32)
            self.embedding = self.embedding / np.linalg.norm(self.embedding)
        
        # Keep track of recent features for appearance updates
        self.features.append(self.embedding)
        if len(self.features) > 10:  # Keep last 10 features
            self.features.pop(0)
            
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.age = 0
        self.last_seen = time.time()
        
        # Update history
        self.history.append(self.bbox.copy())
        
        # Mark as confirmed if we've had enough hits
        if self.state == 0 and self.hits >= self.n_init:
            self.state = 1  # Tentative -> Confirmed
        
        # Update history and features
        self.history.append(self.bbox.copy())
        self.features.append(embedding)
        if len(self.features) > 100:  # Keep only recent features
            self.features = self.features[-100:]
    
    def mark_missed(self) -> None:
        """Mark this track as missed (no detection at this time step)."""
        if self.state == 0:  # Tentative
            self.state = -1  # Mark for deletion
        elif self.time_since_update > self.max_age:
            self.state = -1  # Mark for deletion
    
    def is_tentative(self) -> bool:
        """Check if the track is tentative."""
        return self.state == 0
    
    def is_confirmed(self) -> bool:
        """Check if the track is confirmed."""
        return self.state == 1
    
    def is_deleted(self) -> bool:
        """Check if the track is marked for deletion."""
        return self.state == -1
        
    def get_global_id(self) -> Optional[int]:
        """Get the global ID of this track."""
        return self.global_id
        
    def set_global_id(self, global_id: int) -> None:
        """Set the global ID of this track."""
        self.global_id = global_id


class MultiCameraTracker:
    """
    Multi-camera tracker that manages tracks across multiple camera views.
    Handles per-camera tracking and global ID assignment.
    """
    
    def __init__(self, 
                 detector: YOLOv8Detector,
                 reid_model: ReIDModel,
                 device: torch.device,
                 max_iou_distance: float = 0.7,
                 max_age: int = 30,
                 n_init: int = 3,
                 nn_budget: Optional[int] = 100,
                 max_cosine_distance: float = 0.4,
                 matching_threshold: float = 0.2,
                 min_global_confidence: float = 0.7):
        """
        Initialize the multi-camera tracker.
        
        Args:
            detector: YOLOv8 detector instance
            reid_model: ReID model for feature extraction
            device: Device to run the models on
            max_iou_distance: Maximum IoU distance for association
            max_age: Maximum number of misses before a track is deleted
            n_init: Number of consecutive detections before a track is confirmed
            nn_budget: Maximum size of the appearance descriptor gallery
            max_cosine_distance: Maximum cosine distance for ReID matching
            matching_threshold: Matching threshold for track confirmation
            min_global_confidence: Minimum confidence for global ID assignment
        """
        self.detector = detector
        self.reid_model = reid_model
        self.device = device
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.max_cosine_distance = max_cosine_distance
        self.matching_threshold = matching_threshold
        self.min_global_confidence = min_global_confidence
        
        # Track management
        self.next_ids: Dict[CameraID, int] = defaultdict(lambda: 1)  # Per-camera track ID counter
        self.tracks: Dict[CameraID, Dict[int, Track]] = defaultdict(dict)  # camera_id -> {track_id -> Track}
        self.deleted_tracks: List[Track] = []  # For potential recovery
        
        # Global ID management
        self.global_id_manager = GlobalIDManager()
        
        # Appearance features for ReID with budget
        self.samples: Dict[int, List[Feature]] = defaultdict(list)  # global_id -> list of features
        
        # Camera parameters (can be updated later)
        self.camera_params: Dict[CameraID, Dict[str, Any]] = {}
        
        # Track statistics
        self.frame_count = 0
        
    def update(self, 
              frame: np.ndarray, 
              camera_id: CameraID,
              timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Update tracks with detections from a new frame.
        
        Args:
            frame: Input frame (BGR format)
            camera_id: ID of the camera this frame comes from
            timestamp: Optional timestamp of the frame
            
        Returns:
            List of active tracks with their information
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.frame_count += 1
        
        # Run detection
        bboxes, scores, class_ids = self.detector.detect(frame)
        
        # Extract features for each detection
        detections = []
        if len(bboxes) > 0:
            features = self.reid_model.extract_features_batch(frame, bboxes)
            detections = [
                (bbox, score, class_id, feature) 
                for bbox, score, class_id, feature in zip(bboxes, scores, class_ids, features)
            ]
        
        # Predict existing tracks for this camera
        self._predict_tracks(camera_id)
        
        # Match detections to existing tracks and update
        matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
            camera_id, detections
        )
        
        # Update matched tracks with detections
        for track_idx, det_idx in matched_pairs:
            track = list(self.tracks[camera_id].values())[track_idx]
            bbox, score, class_id, feature = detections[det_idx]
            track.update(bbox, feature)
            
            # Update global ID if needed
            self._update_global_id(track, feature)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            bbox, score, class_id, feature = detections[det_idx]
            self._create_new_track(camera_id, bbox, feature, class_id)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            track = list(self.tracks[camera_id].values())[track_idx]
            track.mark_missed()
        
        # Clean up deleted tracks for this camera
        self._delete_deleted_tracks(camera_id)
        
        # Periodically clean up old tracks across all cameras
        if self.frame_count % 30 == 0:  # Every 30 frames
            self._cleanup_old_tracks()
        
        # Return active tracks for this camera
        return self._get_active_tracks(camera_id)
    
    def _predict_tracks(self, camera_id: CameraID) -> None:
        """Predict the next state of all tracks for a camera."""
        for track in self.tracks.get(camera_id, {}).values():
            track.predict()
    
    def _match_detections_to_tracks(
        self,
        camera_id: CameraID,
        detections: List[Tuple[BBox, float, int, Feature]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks for a specific camera.
        
        Args:
            camera_id: ID of the camera
            detections: List of (bbox, score, class_id, feature) tuples
            
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if camera_id not in self.tracks or not self.tracks[camera_id]:
            # No existing tracks, all detections are unmatched
            return [], list(range(len(detections))), []
            
        if not detections:
            # No detections, all tracks are unmatched
            return [], [], list(range(len(self.tracks[camera_id])))
        
        # Get confirmed and unconfirmed tracks
        tracks = list(self.tracks[camera_id].values())
        confirmed_tracks = [i for i, t in enumerate(tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(tracks) if t.is_tentative()]
        
        # Match confirmed tracks using appearance and motion
        matches_a, unmatched_tracks_a, unmatched_detections = self._match_with_reid(
            [tracks[i] for i in confirmed_tracks],
            detections
        )
        
        # Match remaining detections to unconfirmed tracks using IoU only
        if unmatched_detections and unconfirmed_tracks:
            # Create a mapping from filtered detections back to original indices
            filtered_detections = [detections[i] for i in unmatched_detections]
            matches_b, unmatched_tracks_b, remaining_detections = self._match_with_iou(
                [tracks[i] for i in unconfirmed_tracks],
                filtered_detections
            )
            
            # Update indices for the original track list
            matched_pairs = []
            for i, j in matches_a:
                matched_pairs.append((confirmed_tracks[i], j))
                
            for i, j in matches_b:
                # Map back to the original unmatched_detections indices
                matched_pairs.append((unconfirmed_tracks[i], unmatched_detections[j]))
            
            # Update unmatched detections with the remaining ones after second matching
            unmatched_detections = [unmatched_detections[i] for i in remaining_detections]
            
            # Get unmatched track indices from both matching phases
            unmatched_tracks = [confirmed_tracks[i] for i in unmatched_tracks_a] + \
                             [unconfirmed_tracks[i] for i in unmatched_tracks_b]
        else:
            # No unconfirmed tracks or no unmatched detections
            matched_pairs = [(confirmed_tracks[i], j) for i, j in matches_a]
            unmatched_tracks = [confirmed_tracks[i] for i in unmatched_tracks_a]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _match_with_reid(
        self,
        tracks: List[Track],
        detections: List[Tuple[BBox, float, int, Feature]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match tracks to detections using both appearance and IoU.
        
        Args:
            tracks: List of Track objects
            detections: List of (bbox, score, class_id, feature) tuples
            
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
            
        # Calculate IoU distance matrix
        track_boxes = np.array([t.bbox for t in tracks])
        det_boxes = np.array([d[0] for d in detections])
        iou_matrix = self._iou_distance(track_boxes, det_boxes)
        
        # Calculate appearance distance matrix
        track_features = np.array([t.embedding for t in tracks])
        det_features = np.array([d[3] for d in detections])
        
        # Safe normalization with zero-norm checks
        def safe_normalize(features):
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            # Replace zero norms with 1 to avoid division by zero
            safe_norms = np.maximum(norms, 1e-12)
            normalized = features / safe_norms
            
            # For features that had zero norm, replace with random unit vectors
            zero_mask = (norms.squeeze() < 1e-12)
            if np.any(zero_mask):
                rand_vecs = np.random.randn(np.sum(zero_mask), features.shape[1]).astype(np.float32)
                rand_norms = np.linalg.norm(rand_vecs, axis=1, keepdims=True)
                rand_vecs = rand_vecs / np.maximum(rand_norms, 1e-12)
                normalized[zero_mask] = rand_vecs
                
            return normalized
        
        # Normalize track and detection features
        track_features = safe_normalize(track_features)
        det_features = safe_normalize(det_features)
        
        # Calculate cosine similarity with numerical stability
        cos_sim = np.dot(track_features, det_features.T)
        cos_sim = np.clip(cos_sim, -1.0 + 1e-8, 1.0 - 1e-8)  # Clip to avoid numerical issues
        cos_dist = 1.0 - cos_sim  # Convert to distance (0 = identical, 2 = opposite)
        
        # Combine IoU and appearance distances
        cost_matrix = 0.5 * cos_dist + 0.5 * iou_matrix
        
        # Apply gating
        cost_matrix[cos_dist > self.max_cosine_distance] = 1.0 + 1e-5
        cost_matrix[iou_matrix > self.max_iou_distance] = 1.0 + 1e-5
        
        # Use Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        
        # Convert to cost (lower is better)
        cost_matrix[cost_matrix > 1.0] = 1.0 + 1e-5
        
        # Apply matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by max distance
        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= 1.0:
                matches.append((row, col))
        
        # Find unmatched tracks and detections
        matched_track_indices = set(row for row, _ in matches)
        matched_det_indices = set(col for _, col in matches)
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _match_with_iou(
        self,
        tracks: List[Track],
        detections: List[Tuple[BBox, float, int, Feature]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match tracks to detections using IoU only.
        
        Args:
            tracks: List of Track objects
            detections: List of (bbox, score, class_id, feature) tuples
            
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate IoU distance matrix
        track_boxes = np.array([t.bbox for t in tracks])
        det_boxes = np.array([d[0] for d in detections])
        iou_matrix = self._iou_distance(track_boxes, det_boxes)
        
        # Apply gating
        iou_matrix[iou_matrix > self.max_iou_distance] = 1.0 + 1e-5
        
        # Use Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        
        # Apply matching
        row_indices, col_indices = linear_sum_assignment(iou_matrix)
        
        # Filter matches by max distance
        matches = []
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] <= self.max_iou_distance:
                matches.append((row, col))
        
        # Find unmatched tracks and detections
        matched_track_indices = set(row for row, _ in matches)
        matched_det_indices = set(col for _, col in matches)
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _create_new_track(
        self,
        camera_id: CameraID,
        bbox: BBox,
        feature: Feature,
        class_id: int = 0
    ) -> Track:
        """
        Create a new track from a detection.
        
        Args:
            camera_id: ID of the camera
            bbox: Bounding box in [x1, y1, x2, y2] format
            feature: Feature embedding
            class_id: Class ID of the object
            
        Returns:
            The newly created track
        """
        # Get next available track ID for this camera
        track_id = self.next_ids[camera_id]
        self.next_ids[camera_id] += 1
        
        # Create new track
        track = Track(
            track_id=track_id,
            bbox=bbox,
            embedding=feature,
            camera_id=camera_id,
            class_id=class_id,
            max_age=self.max_age,
            n_init=self.n_init
        )
        
        # Add to active tracks
        self.tracks[camera_id][track_id] = track
        
        return track
    
    def _update_global_id(self, track: Track, feature: Feature) -> None:
        """
        Update the global ID of a track based on appearance features.
        
        Args:
            track: The track to update
            feature: Current feature embedding
        """
        if not track.is_confirmed():
            return  # Only update global IDs for confirmed tracks
            
        if track.global_id is None:
            # Try to match with existing global IDs
            global_id, confidence = self.global_id_manager.match(
                feature, 
                min_confidence=self.min_global_confidence
            )
            
            if global_id is not None:
                # Found a match with sufficient confidence
                track.set_global_id(global_id)
                self.samples[global_id].append(feature)
            else:
                # Create new global ID
                global_id = self.global_id_manager.add(feature)
                track.set_global_id(global_id)
                self.samples[global_id] = [feature]
        else:
            # Update existing global ID with new feature
            self.global_id_manager.update(track.global_id, feature)
            self.samples[track.global_id].append(feature)
            
            # Enforce budget
            if self.nn_budget is not None and len(self.samples[track.global_id]) > self.nn_budget:
                self.samples[track.global_id] = self.samples[track.global_id][-self.nn_budget:]
    
    def _delete_deleted_tracks(self, camera_id: CameraID) -> None:
        """
        Remove tracks marked for deletion for a specific camera.
        
        Args:
            camera_id: ID of the camera
        """
        if camera_id not in self.tracks:
            return
            
        # Find tracks marked for deletion
        deleted_track_ids = [
            track_id for track_id, track in self.tracks[camera_id].items() 
            if track.is_deleted()
        ]
        
        # Move to deleted tracks (for potential recovery)
        for track_id in deleted_track_ids:
            track = self.tracks[camera_id].pop(track_id)
            self.deleted_tracks.append(track)
            
        # Keep only recent deleted tracks
        if len(self.deleted_tracks) > 1000:  # Keep last 1000 deleted tracks
            self.deleted_tracks = self.deleted_tracks[-1000:]
    
    def _cleanup_old_tracks(self) -> None:
        """Clean up old tracks across all cameras."""
        current_time = time.time()
        max_age_seconds = self.max_age / 30.0  # Assuming 30 FPS
        
        for camera_id in list(self.tracks.keys()):
            for track_id, track in list(self.tracks[camera_id].items()):
                if current_time - track.last_seen > max_age_seconds:
                    track.mark_missed()
            
            # Remove deleted tracks
            self._delete_deleted_tracks(camera_id)
    
    def _get_active_tracks(self, camera_id: CameraID) -> List[Dict[str, Any]]:
        """
        Get information about active tracks for a camera.
        
        Args:
            camera_id: ID of the camera
            
        Returns:
            List of dictionaries containing track information
        """
        active_tracks = []
        
        if camera_id not in self.tracks:
            return active_tracks
            
        for track in self.tracks[camera_id].values():
            if track.is_confirmed() or track.is_tentative():
                active_tracks.append({
                    'track_id': track.track_id,
                    'global_id': track.global_id,
                    'camera_id': track.camera_id,
                    'bbox': track.bbox.tolist(),
                    'class_id': track.class_id,
                    'age': track.age,
                    'hits': track.hits,
                    'time_since_update': track.time_since_update,
                    'state': 'confirmed' if track.is_confirmed() else 'tentative'
                })
                
        return active_tracks
    
    def _initiate_track(self, camera_id: CameraID, bbox: BBox, feature: Feature, class_id: int) -> Track:
        """
        Initialize a new track (legacy method, use _create_new_track instead).
        
        Args:
            camera_id: ID of the camera
            bbox: Bounding box in [x1, y1, x2, y2] format
            feature: Feature embedding for the object
            class_id: Class ID of the object
            
        Returns:
            The newly created track
        """
        return self._create_new_track(camera_id, bbox, feature, class_id)
    
    def get_all_active_tracks(self) -> Dict[CameraID, List[Dict[str, Any]]]:
        """
        Get information about all active tracks across all cameras.
        
        Returns:
            Dictionary mapping camera IDs to lists of track information
        """
        return {
            camera_id: self._get_active_tracks(camera_id)
            for camera_id in self.tracks.keys()
        }
    
    def get_global_id_counts(self) -> Dict[int, int]:
        """
        Get count of tracks per global ID.
        
        Returns:
            Dictionary mapping global IDs to track counts
        """
        counts = defaultdict(int)
        for camera_tracks in self.tracks.values():
            for track in camera_tracks.values():
                if track.global_id is not None and track.is_confirmed():
                    counts[track.global_id] += 1
        return dict(counts)
    
    def get_camera_ids(self) -> List[CameraID]:
        """
        Get list of camera IDs being tracked.
        
        Returns:
            List of camera IDs
        """
        return list(self.tracks.keys())
    
    def reset(self) -> None:
        """Reset the tracker, clearing all tracks and state."""
        self.tracks.clear()
        self.deleted_tracks.clear()
        self.samples.clear()
        self.next_ids.clear()
        self.frame_count = 0
        self.global_id_manager = GlobalIDManager()
    
    @staticmethod
    def _iou_distance(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU-based distance between two sets of bounding boxes.
        
        Args:
            bboxes1: First set of bounding boxes (N x 4)
            bboxes2: Second set of bounding boxes (M x 4)
            
        Returns:
            Distance matrix (N x M) where lower means more similar
        """
        # Calculate intersection areas
        x1 = np.maximum(bboxes1[:, None, 0], bboxes2[:, 0])
        y1 = np.maximum(bboxes1[:, None, 1], bboxes2[:, 1])
        x2 = np.minimum(bboxes1[:, None, 2], bboxes2[:, 2])
        y2 = np.minimum(bboxes1[:, None, 3], bboxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union areas
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        union = area1[:, None] + area2 - intersection
        
        # Calculate IoU and distance
        iou = np.zeros_like(intersection, dtype=np.float32)
        mask = union > 0
        iou[mask] = intersection[mask] / union[mask]
        
        # Distance is 1 - IoU
        return 1.0 - iou
