"""
Track class for StrongSORT.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from collections import deque


class TrackState:
    """
    Enumeration type for the single target track state.
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    """
    
    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        n_init: int,
        max_age: int,
        feature: np.ndarray,
        mc_lambda: float = 0.995,
        max_unmatched_preds: int = 0,
        max_past_hits: int = 1,
        max_past_misses: int = 0,
    ) -> None:
        """
        Initialize a track with initial mean, covariance, and feature.
        
        Args:
            mean: Mean vector of the initial state distribution.
            covariance: Covariance matrix of the initial state distribution.
            track_id: A unique track identifier.
            n_init: Number of consecutive detections before the track is confirmed.
            max_age: Maximum number of consecutive misses before the track is deleted.
            feature: Feature vector of the detection this track originates from.
            mc_lambda: Motion compensation lambda parameter.
            max_unmatched_preds: Maximum number of unmatched predictions.
            max_past_hits: Maximum past hits for track confirmation.
            max_past_misses: Maximum past misses for track deletion.
        """
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.features: List[np.ndarray] = []
        
        if feature is not None:
            self.features.append(feature)
        
        self._n_init = n_init
        self._max_age = max_age
        self._mc_lambda = mc_lambda
        self._max_unmatched_preds = max_unmatched_preds
        self._max_past_hits = max_past_hits
        self._max_past_misses = max_past_misses
        
        # For motion compensation
        self.mc_mean = mean.copy()
        self.mc_covariance = covariance.copy()
        
        # For track interpolation
        self.past_observations = deque(maxlen=30)  # Store past observations
        self.predicted_observations = deque(maxlen=30)  # Store predicted observations
        self.unmatched_preds = 0  # Number of consecutive unmatched predictions
        
        # For track confirmation/deletion
        self.past_hits = deque([1], maxlen=max_past_hits)
        self.past_misses = deque([0], maxlen=max_past_misses)
    
    def to_tlwh(self) -> np.ndarray:
        """
        Get current position in bounding box format `(top left x, top left y, width, height)`.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    def to_tlbr(self) -> np.ndarray:
        """
        Get current position in bounding box format `(top left x, top left y, bottom right x, bottom right y)`.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    def to_xyah(self) -> np.ndarray:
        """
        Get current position in bounding box format `(center x, center y, aspect ratio, height)`,
        where the aspect ratio is `width / height`.
        """
        return self.mean[:4].copy()
    
    def predict(self, kf) -> None:
        """
        Predict the next state using the Kalman filter.
        
        Args:
            kf: The Kalman filter.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        
        # Store predicted observation for interpolation
        self.predicted_observations.append((self.age, self.to_tlbr()))
        
        # Update motion compensation
        if self.time_since_update > 1:
            self.mc_mean, self.mc_covariance = kf.update(
                self.mc_mean, self.mc_covariance, self.to_xyah()
            )
    
    def update(self, kf, detection: 'Detection', ema_alpha: float = 0.9) -> None:
        """
        Update the state with a new detection.
        
        Args:
            kf: The Kalman filter.
            detection: The associated detection.
            ema_alpha: EMA alpha parameter for feature updates.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        
        # Update feature
        if detection.feature is not None:
            if len(self.features) > 0:
                # Update feature with EMA
                self.features[0] = ema_alpha * self.features[0] + (1.0 - ema_alpha) * detection.feature
                self.features[0] /= np.linalg.norm(self.features[0])  # Normalize
            else:
                self.features.append(detection.feature)
        
        # Update state
        self.hits += 1
        self.time_since_update = 0
        
        # Update past hits/misses
        if len(self.past_hits) > 0 and self.past_hits[-1] == 1:
            self.past_hits[-1] += 1
        else:
            self.past_hits.append(1)
        
        if len(self.past_misses) > 0 and self.past_misses[-1] > 0:
            self.past_misses.append(0)
        
        # Update motion compensation
        self.mc_mean = self.mean.copy()
        self.mc_covariance = self.covariance.copy()
        
        # Store observation for interpolation
        self.past_observations.append((self.age, detection.to_tlbr()))
        self.unmatched_preds = 0
    
    def mark_missed(self) -> None:
        """Mark this track as missed (no associated detection at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        else:
            self.unmatched_preds += 1
            
            # Update past misses
            if len(self.past_misses) > 0 and self.past_misses[-1] > 0:
                self.past_misses[-1] += 1
            else:
                self.past_misses.append(1)
            
            # Check if track should be deleted
            if (self._max_unmatched_preds > 0 and 
                self.unmatched_preds > self._max_unmatched_preds):
                self.state = TrackState.Deleted
            
            # Check if track has been missing for too long
            if self.time_since_update > self._max_age:
                self.state = TrackState.Deleted
    
    def is_tentative(self) -> bool:
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative
    
    def is_confirmed(self) -> bool:
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed
    
    def is_deleted(self) -> bool:
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    
    def is_active(self) -> bool:
        """Returns True if this track is active (confirmed or tentative)."""
        return self.state in (TrackState.Tentative, TrackState.Confirmed)
    
    def confirm(self) -> None:
        """Confirm the track."""
        self.state = TrackState.Confirmed
    
    def get_feature(self) -> Optional[np.ndarray]:
        """Get the most recent feature vector."""
        if len(self.features) > 0:
            return self.features[-1]
        return None
    
    def get_prediction(self, kf, dt: int = 1) -> np.ndarray:
        """
        Predict the track's state after dt time steps.
        
        Args:
            kf: The Kalman filter.
            dt: Number of time steps to predict ahead.
            
        Returns:
            Predicted bounding box in [x1, y1, x2, y2] format.
        """
        mean, _ = kf.predict(self.mean, self.covariance, dt=dt)
        
        # Convert to bounding box
        ret = mean[:4].copy()
        ret[2] *= ret[3]  # aspect_ratio * height = width
        
        # Convert to x1, y1, x2, y2
        ret[:2] -= ret[2:] / 2
        ret[2:] += ret[:2]
        
        return ret
    
    def get_interpolated_prediction(self, kf, dt: int = 1) -> np.ndarray:
        """
        Get interpolated prediction using past observations.
        
        Args:
            kf: The Kalman filter.
            dt: Number of time steps to predict ahead.
            
        Returns:
            Interpolated bounding box in [x1, y1, x2, y2] format.
        """
        if len(self.past_observations) < 2 or len(self.predicted_observations) < 2:
            return self.get_prediction(kf, dt)
        
        # Get last two observations and predictions
        t1, obs1 = self.past_observations[-2]
        t2, obs2 = self.past_observations[-1]
        
        p1, pred1 = self.predicted_observations[-2]
        p2, pred2 = self.predicted_observations[-1]
        
        # Calculate velocity in observation space
        obs_vel = (np.array(obs2) - np.array(obs1)) / max(1, t2 - t1)
        
        # Calculate velocity in prediction space
        pred_vel = (np.array(pred2) - np.array(pred1)) / max(1, p2 - p1)
        
        # Calculate compensation
        compensation = obs_vel - pred_vel
        
        # Get predicted state
        pred = self.get_prediction(kf, dt)
        
        # Apply compensation
        compensated = pred + compensation * dt
        
        return compensated
