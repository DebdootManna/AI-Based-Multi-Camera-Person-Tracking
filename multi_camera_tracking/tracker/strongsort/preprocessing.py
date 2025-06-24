"""
Preprocessing utilities for StrongSORT.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    max_detections: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-maximum suppression (NMS) for object detection.
    
    Args:
        boxes: Array of shape (num_boxes, 4) containing bounding boxes in [x1, y1, x2, y2] format.
        scores: Array of shape (num_boxes,) containing confidence scores.
        classes: Optional array of shape (num_boxes,) containing class indices.
        iou_threshold: IoU threshold for NMS.
        score_threshold: Minimum score threshold.
        max_detections: Maximum number of detections to keep.
        
    Returns:
        Tuple of (boxes, scores, classes) after NMS.
    """
    # Filter out boxes with low scores
    keep = scores > score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    
    if classes is not None:
        classes = classes[keep]
    
    # If no boxes remain, return empty arrays
    if len(boxes) == 0:
        empty = np.array([], dtype=np.int32)
        return (
            np.zeros((0, 4), dtype=boxes.dtype),
            np.zeros((0,), dtype=scores.dtype),
            np.zeros((0,), dtype=classes.dtype) if classes is not None else empty
        )
    
    # Sort boxes by score (descending)
    idxs = np.argsort(scores)[::-1]
    boxes = boxes[idxs]
    scores = scores[idxs]
    if classes is not None:
        classes = classes[idxs]
    
    # Initialize list of picked indices
    pick = []
    
    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute area of bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Apply NMS
    while len(pick) < max_detections and len(idxs) > 0:
        # Pick the box with the highest score
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)
        
        # Compute IoU of the picked box with the rest
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersection = w * h
        iou = intersection / (area[i] + area[idxs[1:]] - intersection)
        
        # Keep boxes with IoU <= threshold
        keep = np.where(iou <= iou_threshold)[0]
        idxs = idxs[keep + 1]  # +1 because we excluded idxs[0]
    
    # Return only the picked boxes, scores, and classes
    if classes is not None:
        return boxes[pick], scores[pick], classes[pick]
    return boxes[pick], scores[pick], np.zeros(len(pick), dtype=np.int32)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from [x, y, w, h] to [x1, y1, x2, y2] format.
    
    Args:
        boxes: Array of shape (..., 4) containing bounding boxes in [x, y, w, h] format.
        
    Returns:
        Array of shape (..., 4) containing bounding boxes in [x1, y1, x2, y2] format.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    boxes_out = np.zeros_like(boxes)
    boxes_out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1 = x - w/2
    boxes_out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1 = y - h/2
    boxes_out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2 = x + w/2
    boxes_out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2 = y + h/2
    return boxes_out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x, y, w, h] format.
    
    Args:
        boxes: Array of shape (..., 4) containing bounding boxes in [x1, y1, x2, y2] format.
        
    Returns:
        Array of shape (..., 4) containing bounding boxes in [x, y, w, h] format.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    boxes_out = np.zeros_like(boxes)
    boxes_out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2  # x = (x1 + x2) / 2
    boxes_out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # y = (y1 + y2) / 2
    boxes_out[..., 2] = boxes[..., 2] - boxes[..., 0]  # w = x2 - x1
    boxes_out[..., 3] = boxes[..., 3] - boxes[..., 1]  # h = y2 - y1
    return boxes_out


def clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries.
    
    Args:
        boxes: Array of shape (..., 4) containing bounding boxes in [x1, y1, x2, y2] format.
        shape: Tuple of (height, width) of the image.
        
    Returns:
        Clipped bounding boxes in [x1, y1, x2, y2] format.
    """
    height, width = shape
    boxes_out = boxes.copy()
    boxes_out[..., 0::2] = np.clip(boxes[..., 0::2], 0, width - 1)  # x1, x2
    boxes_out[..., 1::2] = np.clip(boxes[..., 1::2], 0, height - 1)  # y1, y2
    return boxes_out


def scale_boxes(
    boxes: np.ndarray, 
    scale: Union[float, Tuple[float, float]]
) -> np.ndarray:
    """
    Scale bounding boxes by a factor.
    
    Args:
        boxes: Array of shape (..., 4) containing bounding boxes in [x1, y1, x2, y2] format.
        scale: Scale factor (sx, sy) or single scale for both dimensions.
        
    Returns:
        Scaled bounding boxes in [x1, y1, x2, y2] format.
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    
    boxes = np.asarray(boxes, dtype=np.float32)
    boxes_out = boxes.copy()
    
    # Scale center coordinates
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    wh = boxes[..., 2:] - boxes[..., :2]
    
    # Apply scale
    wh_scaled = wh * np.array(scale, dtype=np.float32)
    
    # Compute new coordinates
    boxes_out[..., :2] = center - wh_scaled / 2
    boxes_out[..., 2:] = center + wh_scaled / 2
    
    return boxes_out
