"""
Object detection module for the Multi-Camera Person Tracking system.

This module provides interfaces and implementations for object detection,
with a focus on person detection using YOLOv8.
"""

from .yolov8_detector import YOLOv8Detector  # noqa: F401

__all__ = ['YOLOv8Detector']
