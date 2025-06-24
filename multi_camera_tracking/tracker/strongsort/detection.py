"""
Detection class for StrongSORT.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class Detection:
    """
    This class represents a bounding box detection in a single image.
    """
    
    def __init__(
        self, 
        bbox: np.ndarray, 
        confidence: float, 
        feature: np.ndarray,
        class_id: int = 0
    ) -> None:
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format
            confidence: Detection confidence score
            feature: Feature vector for appearance matching
            class_id: Class ID of the detected object
        """
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.class_id = int(class_id)
    
    def to_tlwh(self) -> np.ndarray:
        """Convert bounding box to format `(top left x, top left y, width, height)`."""
        ret = self.bbox.copy()
        ret[2:] = ret[2:] - ret[:2]
        return ret
    
    def to_xyah(self) -> np.ndarray:
        """Convert bounding box to format `(center x, center y, aspect ratio, height)`,
        where the aspect ratio is `width / height`.
        """
        ret = self.to_tlwh()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]  # aspect ratio = width / height
        return ret
    
    def to_xywh(self) -> np.ndarray:
        """Convert bounding box to format `(center x, center y, width, height)`."""
        ret = self.to_tlwh()
        ret[:2] += ret[2:] / 2
        return ret
    
    def to_tlbr(self) -> np.ndarray:
        """Convert bounding box to format `(top left x, top left y, bottom right x, bottom right y)`."""
        return self.bbox.copy()
    
    @classmethod
    def from_yolo(
        cls, 
        yolo_det: List[float], 
        feature: np.ndarray, 
        img_shape: Tuple[int, int],
        class_id: int = 0
    ) -> 'Detection':
        """
        Create a Detection from YOLO format detection.
        
        Args:
            yolo_det: YOLO detection [x_center, y_center, width, height, conf, ...]
            feature: Feature vector for appearance matching
            img_shape: Tuple of (height, width) of the input image
            class_id: Class ID of the detected object
            
        Returns:
            Detection instance
        """
        img_h, img_w = img_shape
        
        # Convert from center_x, center_y, width, height to x1, y1, x2, y2
        x_center, y_center, w, h = yolo_det[:4]
        
        # Scale to image dimensions
        x_center *= img_w
        y_center *= img_h
        w *= img_w
        h *= img_h
        
        # Convert to x1, y1, x2, y2
        x1 = max(0, x_center - w / 2)
        y1 = max(0, y_center - h / 2)
        x2 = min(img_w - 1, x_center + w / 2)
        y2 = min(img_h - 1, y_center + h / 2)
        
        confidence = yolo_det[4] if len(yolo_det) > 4 else 1.0
        
        return cls(
            bbox=[x1, y1, x2, y2],
            confidence=confidence,
            feature=feature,
            class_id=class_id
        )
    
    @classmethod
    def from_xyxy(
        cls, 
        bbox: List[float], 
        confidence: float, 
        feature: np.ndarray,
        class_id: int = 0
    ) -> 'Detection':
        """
        Create a Detection from [x1, y1, x2, y2] format.
        
        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format
            confidence: Detection confidence score
            feature: Feature vector for appearance matching
            class_id: Class ID of the detected object
            
        Returns:
            Detection instance
        """
        return cls(
            bbox=bbox,
            confidence=confidence,
            feature=feature,
            class_id=class_id
        )
    
    def __repr__(self) -> str:
        return f"Detection(bbox={self.bbox}, confidence={self.confidence:.2f}, class_id={self.class_id})"
