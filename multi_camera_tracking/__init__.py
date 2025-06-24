"""
Multi-Camera Person Tracking System

A robust system for tracking persons across multiple camera views using YOLOv8,
StrongSORT, and OSNet ReID. The system supports hardware acceleration on
Apple Silicon (MPS), CUDA, and CPU.
"""

__version__ = "0.1.0"
__author__ = "Debdoot Manna"
__license__ = "MIT"

# Import main components for easier access
from .main import main  # noqa: F401
