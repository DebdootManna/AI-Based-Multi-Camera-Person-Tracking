"""
Linear assignment for StrongSORT.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Any, Optional, Callable

from . import matching
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
    
    This is a thin wrapper around the matching.min_cost_matching function
    to maintain backward compatibility.
    
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
    return matching.min_cost_matching(
        distance_metric=distance_metric,
        max_distance=max_distance,
        tracks=tracks,
        detections=detections,
        track_indices=track_indices,
        detection_indices=detection_indices,
    )


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
    
    This is a thin wrapper around the matching.matching_cascade function
    to maintain backward compatibility.
    
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
    return matching.matching_cascade(
        distance_metric=distance_metric,
        max_distance=max_distance,
        cascade_depth=cascade_depth,
        tracks=tracks,
        detections=detections,
        track_indices=track_indices,
        detection_indices=detection_indices,
    )


def gate_cost_matrix(
    kf,
    cost_matrix: np.ndarray,
    tracks: List[Track],
    detections: List[Detection],
    track_indices: List[int],
    detection_indices: List[int],
    only_position: bool = False,
) -> np.ndarray:
    """
    Apply gating to the cost matrix based on Mahalanobis distance.
    
    This is a thin wrapper around the matching.gate_cost_matrix function
    to maintain backward compatibility.
    
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
    return matching.gate_cost_matrix(
        kf=kf,
        cost_matrix=cost_matrix,
        tracks=tracks,
        detections=detections,
        track_indices=track_indices,
        detection_indices=detection_indices,
        only_position=only_position,
    )
