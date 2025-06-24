"""
StrongSORT: DeepSORT with Strong Association Metrics

This module implements the StrongSORT algorithm, which enhances DeepSORT with:
- Appearance-based ReID features
- Motion compensation
- Camera motion compensation
- Track interpolation
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Dict, Any
import warnings

from .detection import Detection
from .track import Track
from . import matching
from .nn_matching import NearestNeighborDistanceMetric
from .preprocessing import non_max_suppression
from .kalman_filter import KalmanFilter
from . import linear_assignment
from . import iou_matching


class StrongSORT:
    """
    This is the multi-target tracker.
    """
    
    def __init__(
        self,
        metric: NearestNeighborDistanceMetric,
        max_iou_distance: float = 0.7,
        max_age: int = 30,
        n_init: int = 3,
        ema_alpha: float = 0.9,
        mc_lambda: float = 0.995,
        max_unmatched_preds: int = 0,
        motion_compensation: bool = True,
        max_past_hits: int = 1,
        max_past_misses: int = 0,
        gating_only_position: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize the StrongSORT tracker.
        
        Args:
            metric: A metric function for comparing appearance features
            max_iou_distance: Maximum IoU distance for matching
            max_age: Maximum number of missed misses before a track is deleted
            n_init: Number of consecutive detections before the track is confirmed
            ema_alpha: Exponential moving average alpha parameter for feature updates
            mc_lambda: Motion compensation lambda parameter
            max_unmatched_preds: Maximum number of unmatched predictions
            motion_compensation: Whether to use motion compensation
            max_past_hits: Maximum past hits for track confirmation
            max_past_misses: Maximum past misses for track deletion
            gating_only_position: Whether to only use position for gating
        """
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        self.max_unmatched_preds = max_unmatched_preds
        self.motion_compensation = motion_compensation
        self.max_past_hits = max_past_hits
        self.max_past_misses = max_past_misses
        self.gating_only_position = gating_only_position
        
        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_count = 0
        
        # For motion compensation
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
    def predict(self) -> None:
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Perform measurement update and track management.
        
        Args:
            detections: List of detections at current time step
            
        Returns:
            List of active tracks after update
        """
        self.frame_count += 1
        
        # Run matching cascade
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        
        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, 
                detections[detection_idx],
                self.ema_alpha
            )
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Initialize new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # Delete dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Update distance metric
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id] * len(track.features)
            track.features = [track.features[-1]]  # Only keep the latest feature
            
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )
        
        return self.tracks
    
    def _match(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match tracks with detections.
        
        Args:
            detections: List of detections at current time step
            
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        def gated_metric(tracks, dets, track_indices, detection_indices):
            """Apply gating to distance matrix."""
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            # Compute distance matrix
            cost_matrix = self.metric.distance(features, targets)
            
            # Apply gating
            cost_matrix = matching.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, 
                detection_indices, self.gating_only_position
            )
            
            return cost_matrix
        
        # Split track set into confirmed and unconfirmed tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        # Associate confirmed tracks using appearance features
        matches_a, unmatched_tracks_a, unmatched_detections = matching.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age,
            self.tracks, detections, confirmed_tracks
        )
        
        # Associate remaining tracks together with unconfirmed tracks using IoU
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]
        
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance,
                self.tracks, detections, iou_track_candidates, unmatched_detections
            )
        
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _initiate_track(self, detection: Detection) -> None:
        """Initialize a new track from a detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, 
            covariance, 
            self._next_id, 
            self.n_init, 
            self.max_age,
            detection.feature,
            self.mc_lambda,
            self.max_unmatched_preds,
            self.max_past_hits,
            self.max_past_misses
        ))
        self._next_id += 1
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active (confirmed and unconfirmed) tracks."""
        return [t for t in self.tracks if t.is_active()]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get all confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed()]
    
    def get_unconfirmed_tracks(self) -> List[Track]:
        """Get all unconfirmed tracks."""
        return [t for t in self.tracks if not t.is_confirmed()]
    
    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.metric.reset()


def create_tracker(
    max_dist: float = 0.2,
    max_iou_distance: float = 0.7,
    max_age: int = 30,
    n_init: int = 3,
    nn_budget: Optional[int] = 100,
    ema_alpha: float = 0.9,
    mc_lambda: float = 0.995,
    max_unmatched_preds: int = 0,
    motion_compensation: bool = True,
    max_past_hits: int = 1,
    max_past_misses: int = 0,
    gating_only_position: bool = False
) -> StrongSORT:
    """
    Create a StrongSORT tracker with the specified parameters.
    
    Args:
        max_dist: Maximum cosine distance for matching
        max_iou_distance: Maximum IoU distance for matching
        max_age: Maximum number of missed misses before a track is deleted
        n_init: Number of consecutive detections before the track is confirmed
        nn_budget: Maximum size of the appearance descriptor gallery
        ema_alpha: Exponential moving average alpha parameter for feature updates
        mc_lambda: Motion compensation lambda parameter
        max_unmatched_preds: Maximum number of unmatched predictions
        motion_compensation: Whether to use motion compensation
        max_past_hits: Maximum past hits for track confirmation
        max_past_misses: Maximum past misses for track deletion
        gating_only_position: Whether to only use position for gating
        
    Returns:
        A StrongSORT tracker instance
    """
    # Create the metric for appearance features
    metric = NearestNeighborDistanceMetric(
        metric="cosine",
        matching_threshold=max_dist,
        budget=nn_budget
    )
    
    # Create the tracker
    tracker = StrongSORT(
        metric=metric,
        max_iou_distance=max_iou_distance,
        max_age=max_age,
        n_init=n_init,
        ema_alpha=ema_alpha,
        mc_lambda=mc_lambda,
        max_unmatched_preds=max_unmatched_preds,
        motion_compensation=motion_compensation,
        max_past_hits=max_past_hits,
        max_past_misses=max_past_misses,
        gating_only_position=gating_only_position
    )
    
    return tracker
