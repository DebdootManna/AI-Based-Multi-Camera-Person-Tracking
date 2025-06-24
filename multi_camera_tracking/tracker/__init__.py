"""
Tracking module for the Multi-Camera Person Tracking system.

This module provides multi-object tracking functionality using StrongSORT
and handles cross-camera tracking with global ID management.
"""

from .multi_tracker import MultiCameraTracker  # noqa: F401

__all__ = ['MultiCameraTracker']
