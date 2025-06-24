"""
Test script for the Multi-Camera Person Tracking system.

This script verifies that all components are properly imported and can be initialized.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing module imports...")
    
    try:
        # Test PyTorch import
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Test Ultralytics import (for YOLO)
        try:
            import ultralytics
            from ultralytics import YOLO
            ultralytics_version = getattr(ultralytics, '__version__', 'unknown')
            print(f"✅ Ultralytics {ultralytics_version} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import Ultralytics: {e}")
            return False
        
        # Test OpenCV import
        try:
            import cv2
            print(f"✅ OpenCV {cv2.__version__} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import OpenCV: {e}")
            return False
            
        # Test our custom modules
        print("\nTesting custom module imports...")
        
        # Test detector import
        try:
            from multi_camera_tracking.detector import YOLOv8Detector
            print("✅ YOLOv8Detector imported successfully")
        except Exception as e:
            print(f"❌ Failed to import YOLOv8Detector: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # Test ReID model import
        try:
            from multi_camera_tracking.reid import ReIDModel
            print("✅ ReIDModel imported successfully")
        except Exception as e:
            print(f"❌ Failed to import ReIDModel: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # Test tracker import
        try:
            from multi_camera_tracking.tracker import MultiCameraTracker
            print("✅ MultiCameraTracker imported successfully")
        except Exception as e:
            print(f"❌ Failed to import MultiCameraTracker: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # Test utility imports
        try:
            from multi_camera_tracking.utils import GlobalIDManager, VideoSynchronizer
            print("✅ Utility modules imported successfully")
        except Exception as e:
            print(f"❌ Failed to import utility modules: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Critical error during imports: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_setup():
    """Test if PyTorch can detect available devices."""
    print("\nTesting device setup...")
    
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA is available. Using {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ MPS (Apple Silicon) is available")
    else:
        device = torch.device("cpu")
        print("ℹ️ Using CPU (no GPU acceleration available)")
    
    print(f"Device set to: {device}")
    return device

def test_model_initialization(device):
    """Test if models can be initialized."""
    print("\nTesting model initialization...")
    
    try:
        from multi_camera_tracking.detector import YOLOv8Detector
        from multi_camera_tracking.reid import ReIDModel
        
        print("Initializing YOLOv8 detector...")
        detector = YOLOv8Detector(device=device)
        print("✅ YOLOv8 detector initialized")
        
        print("Initializing ReID model...")
        reid_model = ReIDModel(device=device)
        print("✅ ReID model initialized")
        
        return True, detector, reid_model
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False, None, None

def main():
    """Run all tests."""
    print("="*50)
    print("Multi-Camera Person Tracking - System Test")
    print("="*50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Some imports failed. Please check the error messages above.")
        return
    
    # Test device setup
    device = test_device_setup()
    
    # Test model initialization
    success, detector, reid_model = test_model_initialization(device)
    
    if success:
        print("\n✅ All tests passed! The system is ready to use.")
        print("\nTo run the full system with test videos, use:")
        print("  ./run.sh test_videos/camera1.mp4 test_videos/camera2.mp4 --visualize")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
