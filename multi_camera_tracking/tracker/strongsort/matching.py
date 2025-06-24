"""
Matching functions for StrongSORT.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Any, Optional, Callable

from . import kalman_filter
from .track import Track
from .detection import Detection


def min_cost_matching(
    distance_metric: Callable,
    max_distance: float,
    tracks: List[Track],
    detections: List[Detection],
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve linear assignment problem using the Hungarian algorithm.
    
    Args:
        distance_metric: A function that computes distance between tracks and detections.
        max_distance: Maximum distance for matching.
        tracks: List of tracks.
        detections: List of detections.
        track_indices: Indices of tracks to match. Defaults to all tracks.
        detection_indices: Indices of detections to match. Defaults to all detections.
        
    Returns:
        Tuple of (matches, unmatched_tracks, unmatched_detections).
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices
    
    # Compute cost matrix
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices
    )
    
    # Apply gating
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    
    # Solve the linear assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Extract matched and unmatched indices
    matches = []
    unmatched_tracks = []
    unmatched_detections = []
    
    # Collect unmatched track indices
    for t in track_indices:
        if t not in row_indices:
            unmatched_tracks.append(t)
    
    # Collect unmatched detection indices
    for d in detection_indices:
        if d not in col_indices:
            unmatched_detections.append(d)
    
    # Filter out matches with large distances
    for i, j in zip(row_indices, col_indices):
        track_idx = track_indices[i]
        det_idx = detection_indices[j]
        
        if cost_matrix[i, j] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(det_idx)
        else:
            matches.append((track_idx, det_idx))
    
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric: Callable,
    max_distance: float,
    cascade_depth: int,
    tracks: List[Track],
    detections: List[Detection],
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Run matching cascade.
    
    Args:
        distance_metric: A function that computes distance between tracks and detections.
        max_distance: Maximum distance for matching.
        cascade_depth: Maximum number of missed misses before a track is deleted.
        tracks: List of tracks.
        detections: List of detections.
        track_indices: Indices of tracks to match. Defaults to all tracks.
        detection_indices: Indices of detections to match. Defaults to all detections.
        
    Returns:
        Tuple of (matches, unmatched_tracks, unmatched_detections).
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    
    unmatched_detections = detection_indices.copy()
    matches = []
    
    # Sort tracks by number of hits (descending)
    track_indices = sorted(
        track_indices,
        key=lambda i: tracks[i].hits,
        reverse=True
    )
    
    # Match tracks in cascade order
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
            
        # Select tracks with the same number of hits
        track_indices_l = [
            i for i in track_indices
            if tracks[i].time_since_update == 1 + level
        ]
        
        if len(track_indices_l) == 0:
            continue
            
        # Match tracks with detections
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections
        )
        
        matches += matches_l
    
    # Collect unmatched tracks
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    kf: kalman_filter.KalmanFilter,
    cost_matrix: np.ndarray,
    tracks: List[Track],
    detections: List[Detection],
    track_indices: List[int],
    detection_indices: List[int],
    only_position: bool = False,
) -> np.ndarray:
    """
    Apply gating to the cost matrix based on Mahalanobis distance.
    
    Args:
        kf: Kalman filter instance.
        cost_matrix: Cost matrix to gate.
        tracks: List of tracks.
        detections: List of detections.
        track_indices: Indices of tracks.
        detection_indices: Indices of detections.
        only_position: If True, only use position for gating.
        
    Returns:
        Gated cost matrix.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    
    measurements = np.array([detections[i].to_xyah() for i in detection_indices])
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        
        # Compute gating distance
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        
        # Apply gating
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    
    return cost_matrix


def iou_cost(
    tracks: List[Track],
    detections: List[Detection],
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute intersection-over-union distance between tracks and detections.
    
    Args:
        tracks: List of tracks.
        detections: List of detections.
        track_indices: Indices of tracks to match. Defaults to all tracks.
        detection_indices: Indices of detections to match. Defaults to all detections.
        
    Returns:
        Cost matrix of shape (len(track_indices), len(detection_indices)).
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    
    for i, track_idx in enumerate(track_indices):
        track_box = tracks[track_idx].to_tlwh()
        
        for j, det_idx in enumerate(detection_indices):
            det_box = detections[det_idx].to_tlwh()
            cost_matrix[i, j] = 1.0 - iou(track_box, det_box)
    
    return cost_matrix


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute intersection over union of two bounding boxes in [x, y, w, h] format.
    
    Args:
        bbox1: First bounding box [x, y, w, h].
        bbox2: Second bounding box [x, y, w, h].
        
    Returns:
        IoU value.
    """
    # Convert to [x1, y1, x2, y2]
    bbox1 = bbox1.copy()
    bbox2 = bbox2.copy()
    
    bbox1[2:] += bbox1[:2]  # Convert [x1, y1, w, h] to [x1, y1, x2, y2]
    bbox2[2:] += bbox2[:2]
    
    # Compute intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Compute union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / max(union, 1e-6)
