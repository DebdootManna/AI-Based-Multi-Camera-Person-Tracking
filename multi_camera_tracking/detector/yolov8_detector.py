"""
YOLOv8 detector implementation for person detection.
"""

import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional


class YOLOv8Detector:
    def __init__(self, model_name: str = 'yolov8n.pt', 
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.4,
                 device: Optional[torch.device] = None,
                 max_det: int = 100,
                 agnostic_nms: bool = False,
                 augment: bool = False):
        """
        Initialize YOLOv8 detector with enhanced detection settings.
        
        Args:
            model_name: Name of the YOLOv8 model (e.g., 'yolov8n.pt', 'yolov8s.pt')
            conf_threshold: Confidence threshold for detection (lower = more detections)
            iou_threshold: IoU threshold for NMS (lower = more aggressive NMS)
            device: Device to run the model on (cpu, mps, cuda)
            max_det: Maximum number of detections per image
            agnostic_nms: Use class-agnostic NMS
            augment: Apply test time augmentation
        """
        print(f"Initializing YOLOv8 detector with model: {model_name}")
        print(f"Detection parameters - conf: {conf_threshold}, iou: {iou_threshold}, max_det: {max_det}")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device is not None else torch.device('cpu')
        
        # Load the YOLOv8 model
        try:
            self.model = YOLO(model_name)
            self.model.to(self.device)
            print(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {e}")
        
        # Set model parameters with enhanced settings
        self.model.overrides = {
            'conf': conf_threshold,
            'iou': iou_threshold,
            'classes': [0],  # Only detect persons (class 0)
            'max_det': max_det,
            'agnostic_nms': agnostic_nms,
            'augment': augment,
            'verbose': False
        }
    
    @torch.no_grad()
    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect persons in the input image with enhanced detection settings.
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            boxes: Bounding boxes in format [x1, y1, x2, y2]
            scores: Confidence scores
            class_ids: Class IDs (always 0 for person)
        """
        if img is None or img.size == 0:
            print("Warning: Empty image provided to detector")
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)
            
        try:
            # Run inference with enhanced settings
            results = self.model.predict(
                img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],  # Person class only
                verbose=False,
                max_det=self.model.overrides['max_det'],
                agnostic_nms=self.model.overrides['agnostic_nms'],
                augment=self.model.overrides['augment']
            )
            
            # Process results
            boxes = []
            scores = []
            class_ids = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # Convert to numpy arrays
                    boxes.append(result.boxes.xyxy.cpu().numpy())
                    scores.append(result.boxes.conf.cpu().numpy())
                    class_ids.append(result.boxes.cls.cpu().numpy())
            
            if boxes:
                boxes = np.concatenate(boxes, axis=0)
                scores = np.concatenate(scores, axis=0)
                class_ids = np.concatenate(class_ids, axis=0)
                
                # Filter out any invalid boxes
                valid_indices = [i for i, box in enumerate(boxes) 
                               if (box[2] > box[0]) and (box[3] > box[1])]
                
                if valid_indices:
                    boxes = boxes[valid_indices]
                    scores = scores[valid_indices]
                    class_ids = class_ids[valid_indices]
                    return boxes, scores, class_ids.astype(int)
                
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Alias for detect method for backward compatibility."""
        return self.detect(img)
    
    def warmup(self, img_size: Tuple[int, int] = (640, 640), batch_size: int = 1):
        """
        Warmup the model with dummy data.
        
        Args:
            img_size: Input image size (height, width)
            batch_size: Batch size for warmup
        """
        dummy_img = np.zeros((*img_size, 3), dtype=np.uint8)
        for _ in range(2):  # Run a few times
            self.detect(dummy_img)
