"""
IOU matching for StrongSORT.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable

from . import matching
from .track import Track
from .detection import Detection


def iou_cost(
    tracks: List[Track],
    detections: List[Detection],
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute intersection-over-union distance between tracks and detections.
    
    This is a thin wrapper around the matching.iou_cost function
    to maintain backward compatibility.
    
    Args:
        tracks: List of tracks.
        detections: List of detections.
        track_indices: Indices of tracks to match. Defaults to all tracks.
        detection_indices: Indices of detections to match. Defaults to all detections.
        
    Returns:
        Cost matrix of shape (len(track_indices), len(detection_indices)).
    """
    return matching.iou_cost(
        tracks=tracks,
        detections=detections,
        track_indices=track_indices,
        detection_indices=detection_indices,
    )


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute intersection over union of two bounding boxes in [x, y, w, h] format.
    
    This is a thin wrapper around the matching.iou function
    to maintain backward compatibility.
    
    Args:
        bbox1: First bounding box [x, y, w, h].
        bbox2: Second bounding box [x, y, w, h].
        
    Returns:
        IoU value.
    """
    return matching.iou(bbox1, bbox2)
