"""
Kalman filter for StrongSORT.
"""

import numpy as np
from scipy.linalg import cholesky
from typing import Tuple, Optional


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The state is defined as the bounding box [x, y, a, h], where (x, y) is the center,
    a is the aspect ratio, and h is the height.
    """
    
    def __init__(self) -> None:
        # State transition matrix (motion model)
        self._motion_mat = np.eye(8, 8)
        self._motion_mat[0, 4] = 1.0  # x + dx
        self._motion_mat[1, 5] = 1.0  # y + dy
        self._motion_mat[2, 6] = 1.0  # a + da
        self._motion_mat[3, 7] = 1.0  # h + dh
        
        # Observation matrix (measurement model)
        self._update_mat = np.eye(4, 8)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        
        Args:
            measurement: Bounding box in [x, y, a, h] format
            
        Returns:
            Mean vector (8 dimensional) and covariance matrix (8x8) of the new track.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]  # [x, y, a, h, vx, vy, va, vh]
        
        std = [
            2 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[3],  # y
            1e-2,  # a
            2 * self._std_weight_position * measurement[3],  # h
            10 * self._std_weight_velocity * measurement[3],  # vx
            10 * self._std_weight_velocity * measurement[3],  # vy
            1e-5,  # va
            10 * self._std_weight_velocity * measurement[3],  # vh
        ]
        
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray, dt: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        
        Args:
            mean: Mean vector of the state (8 dimensional)
            covariance: Covariance matrix of the state (8x8)
            dt: Time step (default: 1)
            
        Returns:
            The predicted mean and covariance matrix.
        """
        # Adjust motion model for time step
        motion_mat = self._motion_mat.copy()
        motion_mat[:4, 4:] = np.diag([dt] * 4)
        
        # Predict
        mean = np.dot(motion_mat, mean)
        covariance = np.linalg.multi_dot((
            motion_mat, covariance, motion_mat.T
        ))
        
        # Add process noise
        std_pos = [
            self._std_weight_position * mean[3],  # x
            self._std_weight_position * mean[3],  # y
            1e-2,  # a
            self._std_weight_position * mean[3],  # h
        ]
        
        std_vel = [
            self._std_weight_velocity * mean[3],  # vx
            self._std_weight_velocity * mean[3],  # vy
            1e-5,  # va
            self._std_weight_velocity * mean[3],  # vh
        ]
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        covariance += motion_cov
        
        return mean, covariance
    
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        
        Args:
            mean: Mean vector of the state (8 dimensional)
            covariance: Covariance matrix of the state (8x8)
            
        Returns:
            The projected mean and covariance matrix.
        """
        std = [
            self._std_weight_position * mean[3],  # x
            self._std_weight_position * mean[3],  # y
            1e-1,  # a
            self._std_weight_position * mean[3],  # h
        ]
        
        # Project to measurement space
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        ))
        
        # Add measurement noise
        noise_cov = np.diag(np.square(std))
        projected_cov += noise_cov
        
        return projected_mean, projected_cov
    
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.
        
        Args:
            mean: Mean vector of the state (8 dimensional)
            covariance: Covariance matrix of the state (8x8)
            measurement: Measurement vector (4 dimensional)
            
        Returns:
            The updated mean and covariance matrix.
        """
        # Project state to measurement space
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Compute Kalman gain
        chol_factor, lower = cholesky(projected_cov, lower=True, check_finite=False)
        kalman_gain = np.linalg.lstsq(
            projected_cov.T, 
            np.dot(covariance, self._update_mat.T).T,
            rcond=None
        )[0].T
        
        # Update state
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        
        return new_mean, new_covariance
    
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.
        
        Args:
            mean: Mean vector of the state (8 dimensional)
            covariance: Covariance matrix of the state (8x8)
            measurements: Array of N measurements in [x, y, a, h] format
            only_position: If True, only x, y are used in the calculation
            
        Returns:
            Array of length N containing the squared Mahalanobis distance for each measurement.
        """
        mean, covariance = self.project(mean, covariance)
        
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        cholesky_factor = cholesky(covariance, lower=True, check_finite=False)
        d = measurements - mean
        z = np.linalg.solve(cholesky_factor, d.T).T
        squared_maha = np.sum(z * z, axis=1)
        
        return squared_maha
