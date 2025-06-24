"""
Nearest neighbor matching for StrongSORT.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque


class NearestNeighborDistanceMetric:
    """
    A nearest neighbor distance metric that uses cosine distance for feature matching.
    
    Parameters:
    -----------
    metric : str
        The distance metric to use. Currently only 'cosine' is supported.
    matching_threshold : float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest examples when the budget is reached.
    """
    
    def __init__(
        self,
        metric: str = "cosine",
        matching_threshold: float = 0.2,
        budget: Optional[int] = 100,
    ) -> None:
        self.metric = metric
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples: Dict[int, Dict[str, Any]] = {}
    
    def partial_fit(
        self, 
        features: np.ndarray, 
        targets: np.ndarray, 
        active_targets: Optional[List[int]] = None
    ) -> None:
        """
        Update the distance metric with new data.
        
        Parameters:
        -----------
        features : ndarray
            An NxM matrix of N features of dimension M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int], optional
            A list of targets that are currently present in the scene.
        """
        if active_targets is None:
            active_targets = []
        
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, {"features": deque(maxlen=self.budget), "count": 0})
            self.samples[target]["features"].append(feature)
            self.samples[target]["count"] += 1
        
        # Remove samples of targets that are no longer active
        if active_targets:
            self.samples = {
                k: v for k, v in self.samples.items()
                if k in active_targets
            }
    
    def distance(
        self, 
        features: np.ndarray, 
        targets: np.ndarray,
    ) -> np.ndarray:
        """
        Compute distance between features and targets.
        
        Parameters:
        -----------
        features : ndarray
            An NxM matrix of N features of dimension M.
        targets : ndarray
            An array of target indices.
            
        Returns:
        --------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the squared distance between
            `targets[i]` and `features[j]`.
        """
        if self.metric == "cosine":
            return self._cosine_distance(features, targets)
        else:
            raise ValueError(f"Invalid metric: {self.metric}")
    
    def _cosine_distance(
        self, 
        features: np.ndarray, 
        targets: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine distance between features and targets.
        
        Parameters:
        -----------
        features : ndarray
            An NxM matrix of N features of dimension M.
        targets : ndarray
            An array of target indices.
            
        Returns:
        --------
        ndarray
            Returns a cost matrix of shape len(targets), len(features).
        """
        # Normalize features
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Initialize cost matrix
        cost_matrix = np.zeros((len(targets), len(features)))
        
        for i, target in enumerate(targets):
            if target not in self.samples:
                cost_matrix[i, :] = np.inf
                continue
                
            # Get stored features for this target
            target_features = np.array(self.samples[target]["features"])
            
            # Normalize target features
            target_features = target_features / np.linalg.norm(
                target_features, axis=1, keepdims=True
            )
            
            # Compute cosine similarity (1 - cosine distance)
            similarity = np.dot(target_features, features.T)
            
            # Use the minimum distance (maximum similarity) across all stored features
            cost_matrix[i, :] = 1.0 - np.max(similarity, axis=0)
        
        return cost_matrix
    
    def reset(self) -> None:
        """Reset the metric to its initial state."""
        self.samples = {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metric='{self.metric}', matching_threshold={self.matching_threshold}, budget={self.budget})"
