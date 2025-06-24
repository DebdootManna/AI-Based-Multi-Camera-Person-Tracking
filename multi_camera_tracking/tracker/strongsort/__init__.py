"""
StrongSORT: DeepSORT with Strong Association Metrics

This module implements the StrongSORT algorithm, which enhances DeepSORT with:
- Appearance-based ReID features
- Motion compensation
- Camera motion compensation
- Track interpolation
"""

from .strong_sort import StrongSORT
from .nn_matching import NearestNeighborDistanceMetric
from .track import Track
from .detection import Detection

__all__ = ['StrongSORT', 'NearestNeighborDistanceMetric', 'Track', 'Detection']
