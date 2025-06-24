"""
Utility functions for the Multi-Camera Person Tracking system.

This module provides various utility functions including:
- Global ID management
- Frame synchronization
- Visualization utilities
- Helper functions
"""

from .global_id_manager import GlobalIDManager  # noqa: F401
from .sync_handler import VideoSynchronizer  # noqa: F401
from .visualize import draw_boxes, draw_bbox  # noqa: F401

__all__ = ['GlobalIDManager', 'VideoSynchronizer', 'draw_boxes', 'draw_bbox']
