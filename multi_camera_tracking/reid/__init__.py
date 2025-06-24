"""
Re-Identification module for the Multi-Camera Person Tracking system.

This module provides interfaces and implementations for person re-identification
using OSNet and other ReID models.
"""

from .reid_model import ReIDModel  # noqa: F401

__all__ = ['ReIDModel']
