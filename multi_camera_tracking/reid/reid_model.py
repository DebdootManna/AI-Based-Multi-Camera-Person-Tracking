"""
Re-Identification (ReID) model implementation using OSNet.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import MLP
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path

# Import OSNet implementation (we'll vendor a lightweight version)
from .osnet import osnet_x1_0


class ReIDModel:
    def __init__(self, 
                 model_name: str = 'osnet_x1_0',
                 model_path: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the ReID model.
        
        Args:
            model_name: Name of the model architecture ('osnet_x1_0')
            model_path: Path to the pretrained model weights
            device: Device to run the model on (cpu, mps, cuda)
        """
        self.device = device if device is not None else torch.device('cpu')
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        
        # Initialize model
        self._init_model(model_path)
        
        # Initialize preprocessing
        self._init_preprocess()
    
    def _init_model(self, model_path: Optional[str] = None):
        """Initialize the OSNet model."""
        if self.model_name == 'osnet_x1_0':
            # Load model without classifier weights to avoid shape mismatch
            self.model = osnet_x1_0(
                pretrained=False,  # Don't load pretrained weights to avoid classifier mismatch
                num_classes=0,  # Remove the classifier layer
                loss='softmax',
                use_gpu=self.device.type == 'cuda',
                dropout=0.2
            )
            
            # Try to load pretrained weights from multiple sources
            pretrained_loaded = False
            
            # Try loading from torch hub (official source)
            if model_path is None:
                try:
                    print("Attempting to load OSNet weights from torch hub...")
                    # Official torch hub model
                    pretrained_model = torch.hub.load(
                        'KaiyangZhou/deep-person-reid',
                        'osnet_x1_0',
                        pretrained=True,
                        device='cpu'
                    )
                    # Extract the state dict
                    pretrained_dict = pretrained_model.state_dict()
                    # Remove the classifier weights
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                    if not k.startswith('classifier')}
                    # Load the weights
                    self.model.load_state_dict(pretrained_dict, strict=False)
                    print("Successfully loaded pretrained weights from torch hub")
                    pretrained_loaded = True
                except Exception as e:
                    print(f"Warning: Could not load weights from torch hub: {e}")
            
            # If torch hub loading failed, try loading from local cache if available
            if not pretrained_loaded and model_path is None:
                try:
                    print("Attempting to load OSNet weights from local cache...")
                    # Try to find the weights in the default cache location
                    from torch.hub import get_dir as get_hub_dir
                    hub_dir = Path(get_hub_dir()) / 'checkpoints'
                    cache_files = list(hub_dir.glob('osnet_*.pth'))
                    
                    if cache_files:
                        # Sort by modification time and get the most recent
                        cache_file = sorted(cache_files, key=os.path.getmtime, reverse=True)[0]
                        print(f"Found cached weights: {cache_file}")
                        state_dict = torch.load(cache_file, map_location='cpu')
                        
                        if 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                        
                        # Remove module prefix if present (for DDP models)
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                        # Remove classifier weights
                        state_dict = {k: v for k, v in state_dict.items() 
                                    if not k.startswith('classifier')}
                        
                        # Load the weights
                        self.model.load_state_dict(state_dict, strict=False)
                        print("Successfully loaded weights from local cache")
                        pretrained_loaded = True
                except Exception as e:
                    print(f"Warning: Could not load weights from cache: {e}")
            
            # If no pretrained weights were loaded, initialize with random weights
            if not pretrained_loaded:
                print("Warning: Using randomly initialized weights for OSNet. "
                      "Tracking performance may be degraded.")
            
            # Load custom weights if provided (overrides any previous loading)
            if model_path is not None and os.path.exists(model_path):
                try:
                    print(f"Loading custom weights from {model_path}...")
                    state_dict = torch.load(model_path, map_location='cpu')
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    # Remove module prefix if present (for DDP models)
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    # Remove classifier weights
                    state_dict = {k: v for k, v in state_dict.items() 
                                if not k.startswith('classifier')}
                    # Load the weights
                    self.model.load_state_dict(state_dict, strict=False)
                    print("Successfully loaded custom weights")
                except Exception as e:
                    print(f"Error loading custom weights: {e}")
                    raise
            
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _init_preprocess(self):
        """Initialize image preprocessing pipeline."""
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # Standard size for person ReID
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_features(self, img: np.ndarray, bbox: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract features from an image or image crop.
        
        Args:
            img: Input image in BGR format (numpy array)
            bbox: Optional bounding box in format [x1, y1, x2, y2]. If None, use the entire image.
            
        Returns:
            Feature vector (numpy array)
        """
        # Crop image to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(512, dtype=np.float32)
                
            img = img[y1:y2, x1:x2]
        
        # Skip if image is too small
        if img.size == 0 or min(img.shape[:2]) < 10:
            return np.zeros(512, dtype=np.float32)
        
        # Preprocess image
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Extract features
        with torch.no_grad():
            # Forward pass through the model
            features = self.model(img_tensor)
            
            # If features are multi-dimensional (e.g., from a feature map), apply global average pooling
            if features.dim() > 2:
                features = F.adaptive_avg_pool2d(features, 1)
                features = features.view(features.size(0), -1)
            
            # Normalize features to unit length
            features = F.normalize(features, p=2, dim=1)
        
        # Ensure we return a 1D numpy array
        return features.squeeze().cpu().numpy().flatten()
    
    def extract_features_batch(self, img: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract ReID features for multiple persons in a batch.
        
        Args:
            img: Input image in BGR format
            bboxes: Array of bounding boxes in format [[x1, y1, x2, y2], ...]
            
        Returns:
            features: Array of extracted feature vectors (L2 normalized)
        """
        if len(bboxes) == 0:
            return np.zeros((0, 512), dtype=np.float32)
            
        features = []
        for bbox in bboxes:
            # Extract features for each bounding box
            feat = self.extract_features(img, bbox)
            features.append(feat)
        
        return np.vstack(features) if features else np.zeros((0, 512), dtype=np.float32)
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            features1: First feature vector or array of vectors
            features2: Second feature vector or array of vectors
            
        Returns:
            similarity: Cosine similarity score(s)
        """
        if len(features1) == 0 or len(features2) == 0:
            return np.array([])
        
        # Ensure features are 2D arrays
        features1 = np.atleast_2d(features1)
        features2 = np.atleast_2d(features2)
        
        # Compute cosine similarity
        similarity = np.dot(features1, features2.T)
        
        return similarity.squeeze()
    
    def __call__(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Alias for extract_features for backward compatibility."""
        return self.extract_features(img, bbox)
