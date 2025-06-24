"""
Visualization utilities for multi-camera person tracking.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import random

# Color palette for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
    (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192)
]

def get_color(idx: int) -> Tuple[int, int, int]:
    """
    Get a consistent color for a track ID.
    
    Args:
        idx: Track ID
        
    Returns:
        BGR color tuple
    """
    return COLORS[idx % len(COLORS)]


def draw_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    labels: List[str] = None,
    scores: List[float] = None,
    colors: List[Tuple[int, int, int]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX
) -> np.ndarray:
    """
    Draw bounding boxes with optional labels and scores on an image.
    
    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes in [x1, y1, x2, y2] format
        labels: Optional list of labels for each box
        scores: Optional list of confidence scores for each box
        colors: Optional list of BGR colors for each box
        line_thickness: Thickness of the box lines
        font_scale: Font scale for the text
        font_thickness: Thickness of the text
        font: OpenCV font type
        
    Returns:
        Image with drawn boxes and labels
    """
    if len(boxes) == 0:
        return image
        
    # Make a copy of the image to avoid modifying the original
    img = image.copy()
    
    # Ensure scores is a list
    if not isinstance(scores, (list, tuple)):
        scores = [scores] if scores is not None else []
    
    # Ensure labels is a list
    if not isinstance(labels, (list, tuple)):
        labels = [labels] if labels is not None else []
    
    # Ensure colors is a list
    if not isinstance(colors, (list, tuple)):
        colors = [colors] if colors is not None else []
    
    # If no boxes, return the original image
    if not boxes:
        return image
        
    # Initialize scores with None if not provided or empty
    if not scores:
        scores = [None] * len(boxes)
    
    # Initialize labels with empty strings if not provided or empty
    if not labels:
        labels = [''] * len(boxes)
        
    # Initialize colors with default colors if not provided or empty
    if not colors:
        colors = [get_color(i) for i in range(len(boxes))]
    
    # Ensure all lists have the same length as boxes
    scores = (scores[:len(boxes)] + [None] * (len(boxes) - len(scores)))[:len(boxes)]
    labels = (labels[:len(boxes)] + [''] * (len(boxes) - len(labels)))[:len(boxes)]
    colors = (colors[:len(boxes)] + [get_color(i) for i in range(len(colors), len(boxes))])[:len(boxes)]
    
    # Draw each box
    for i, ((x1, y1, x2, y2), color) in enumerate(zip(boxes, colors)):
        # Draw the bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
        
        # Prepare the label text
        label = str(labels[i]) if (i < len(labels) and labels[i] is not None) else ''
        score = scores[i] if i < len(scores) else None
        
        if score is not None and score != '':
            try:
                # Try to convert score to float and format it
                score_float = float(score)
                score_str = f"{score_float:.2f}"
            except (ValueError, TypeError):
                # If conversion fails, use the score as is
                score_str = str(score) if score is not None else ''
            label = f"{label} {score_str}" if label else score_str
            
        # Ensure label is a string
        label = str(label) if label is not None else ''
        
        if label:
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                img,
                (int(x1), int(y1) - text_height - 5),
                (int(x1) + text_width + 2, int(y1)),
                color,
                -1  # Filled rectangle
            )
            
            # Put text on the image
            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 5),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
                cv2.LINE_AA
            )
    
    return img

def draw_bbox(
    image: np.ndarray,
    bbox: List[float],
    track_id: int = None,
    label: str = None,
    color: Tuple[int, int, int] = None,
    line_thickness: int = 2,
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 5,
    show_label: bool = True,
    show_confidence: bool = False,
    confidence: float = None
) -> np.ndarray:
    """
    Draw a bounding box with optional label and confidence on an image.
    
    Args:
        image: Input image (BGR format)
        bbox: Bounding box in [x1, y1, x2, y2] format
        track_id: Track ID to display
        label: Text label to display
        color: BGR color tuple
        line_thickness: Thickness of the bounding box lines
        text_scale: Font scale for the text
        text_thickness: Thickness of the text
        text_padding: Padding around the text
        show_label: Whether to show the label
        show_confidence: Whether to show the confidence score
        confidence: Confidence score to display
        
    Returns:
        Image with the bounding box drawn
    """
    img = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    
    # Generate a consistent color based on track_id if not provided
    if color is None:
        if track_id is not None:
            color = get_color(track_id)
        else:
            color = (0, 255, 0)  # Default to green
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
    
    # Prepare text
    text_parts = []
    if track_id is not None and show_label:
        text_parts.append(f"ID: {track_id}")
    if label is not None and show_label:
        text_parts.append(label)
    if confidence is not None and show_confidence:
        text_parts.append(f"{confidence:.2f}")
    
    if text_parts:
        text = " ".join(text_parts)
        
        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img,
            (x1, y1 - text_h - 2 * text_padding),
            (x1 + text_w + 2 * text_padding, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            img,
            text,
            (x1 + text_padding, y1 - text_padding // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),  # White text
            text_thickness,
            cv2.LINE_AA
        )
    
    return img

def draw_tracking_info(
    image: np.ndarray,
    tracked_objects: List[Dict[str, Any]],
    global_ids: List[int] = None,
    camera_label: str = None,
    show_confidence: bool = False,
    show_track_id: bool = True,
    show_global_id: bool = True,
    line_thickness: int = 2,
    text_scale: float = 0.5,
    text_thickness: int = 1
) -> np.ndarray:
    """
    Draw tracking information on an image.
    
    Args:
        image: Input image (BGR format)
        tracked_objects: List of tracked objects with detections
        global_ids: List of global IDs corresponding to the tracked objects
        camera_label: Label to display for the camera
        show_confidence: Whether to show confidence scores
        show_track_id: Whether to show track IDs
        show_global_id: Whether to show global IDs
        line_thickness: Thickness of the bounding box lines
        text_scale: Font scale for the text
        text_thickness: Thickness of the text
        
    Returns:
        Image with tracking information drawn
    """
    img = image.copy()
    
    # Draw camera label if provided
    if camera_label:
        cv2.putText(
            img,
            camera_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),  # Red text
            2,
            cv2.LINE_AA
        )
    
    # Draw each tracked object
    for i, obj in enumerate(tracked_objects):
        bbox = obj.get('bbox')
        if bbox is None or len(bbox) < 4:
            continue
        
        # Get track ID and global ID
        track_id = obj.get('track_id')
        global_id = global_ids[i] if global_ids and i < len(global_ids) else None
        
        # Prepare label
        label_parts = []
        if show_track_id and track_id is not None:
            label_parts.append(f"T:{track_id}")
        if show_global_id and global_id is not None:
            label_parts.append(f"G:{global_id}")
        
        label = " ".join(label_parts) if label_parts else None
        
        # Get color based on ID
        color = None
        if global_id is not None:
            color = get_color(global_id)
        elif track_id is not None:
            color = get_color(track_id)
        
        # Draw bounding box and label
        img = draw_bbox(
            img,
            bbox,
            track_id=track_id,
            label=label,
            color=color,
            line_thickness=line_thickness,
            text_scale=text_scale,
            text_thickness=text_thickness,
            show_confidence=show_confidence,
            confidence=obj.get('confidence')
        )
    
    return img

def draw_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    matches: List[Tuple[int, int]],
    tracked_objects1: List[Dict[str, Any]],
    tracked_objects2: List[Dict[str, Any]],
    global_ids1: List[int],
    global_ids2: List[int],
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw matches between two images with tracking information.
    
    Args:
        image1: First image (BGR format)
        image2: Second image (BGR format)
        matches: List of (idx1, idx2) tuples indicating matches
        tracked_objects1: Tracked objects in the first image
        tracked_objects2: Tracked objects in the second image
        global_ids1: Global IDs for objects in the first image
        global_ids2: Global IDs for objects in the second image
        line_color: Color of the match lines
        line_thickness: Thickness of the match lines
        
    Returns:
        Concatenated image with matches drawn
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create a new image with both images side by side
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = image1
    result[:h2, w1:w1+w2] = image2
    
    # Draw matches
    for idx1, idx2 in matches:
        if idx1 >= len(tracked_objects1) or idx2 >= len(tracked_objects2):
            continue
            
        obj1 = tracked_objects1[idx1]
        obj2 = tracked_objects2[idx2]
        
        # Get center points of bounding boxes
        x1 = int((obj1['bbox'][0] + obj1['bbox'][2]) / 2)
        y1 = int((obj1['bbox'][1] + obj1['bbox'][3]) / 2)
        x2 = int((obj2['bbox'][0] + obj2['bbox'][2]) / 2) + w1
        y2 = int((obj2['bbox'][1] + obj2['bbox'][3]) / 2)
        
        # Draw line between matches
        cv2.line(result, (x1, y1), (x2, y2), line_color, line_thickness)
        
        # Draw circles at the center of each box
        cv2.circle(result, (x1, y1), 5, line_color, -1)
        cv2.circle(result, (x2, y2), 5, line_color, -1)
    
    return result

def create_blank_image(width: int, height: int, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Create a blank image with the specified color.
    
    Args:
        width: Width of the image
        height: Height of the image
        color: BGR color tuple
        
    Returns:
        Blank image with the specified color
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    return img

def add_text_to_image(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = None,
    font_scale: float = 1.0,
    thickness: int = 1,
    padding: int = 5
) -> np.ndarray:
    """
    Add text to an image with optional background.
    
    Args:
        image: Input image
        text: Text to add
        position: (x, y) position of the text
        text_color: BGR color of the text
        background_color: BGR color of the background (None for no background)
        font_scale: Font scale
        thickness: Thickness of the text
        padding: Padding around the text (if background is used)
        
    Returns:
        Image with text added
    """
    img = image.copy()
    x, y = position
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Draw background if specified
    if background_color is not None:
        cv2.rectangle(
            img,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + padding),
            background_color,
            -1  # Filled rectangle
        )
    
    # Draw text
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )
    
    return img
