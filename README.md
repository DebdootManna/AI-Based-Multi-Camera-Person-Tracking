# Multi-Camera Person Tracking System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust multi-camera person tracking system that uses YOLOv8 for detection, StrongSORT for tracking, and OSNet for cross-camera re-identification. The system is optimized to run on both CPU and Apple Silicon (MPS) with fallback to CUDA when available.

## Features

- 🎯 **Person Detection**: Utilizes YOLOv8 for fast and accurate person detection
- 🔄 **Multi-Camera Tracking**: Tracks individuals across multiple synchronized camera views
- 🆔 **Cross-Camera Re-Identification**: Uses OSNet for robust person re-identification
- 🚀 **Hardware Acceleration**: Optimized for Apple Silicon (MPS), CUDA, and CPU
- 📊 **Detailed Analytics**: Provides tracking statistics and visualizations
- 🖼️ **Thumbnail Extraction**: Saves thumbnails of detected persons
- 📹 **Video Export**: Saves processed videos with tracking information

## Prerequisites

- Python 3.8 or higher
- macOS (Intel or Apple Silicon) or Linux
- (Optional) NVIDIA GPU with CUDA support for acceleration

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DebdootManna/multi-camera-tracking.git
   cd multi-camera-tracking
   ```

2. **Make the setup script executable**:
   ```bash
   chmod +x run.sh
   ```

3. **Run the setup script** (no arguments needed for setup):
   ```bash
   ./run.sh
   ```
   This will:
   - Create a Python virtual environment
   - Install all required dependencies
   - Detect your hardware configuration (CPU/GPU/Apple Silicon)
   - Install the appropriate version of PyTorch

## Usage

### Basic Usage

```bash
./run.sh video1.mp4 video2.mp4 [options]
```

### Options

- `--output-dir DIR`: Directory to save output files (default: `outputs`)
- `--visualize`: Show real-time visualization
- `--no-save-video`: Don't save output videos
- `--save-thumbnails`: Save thumbnails of detected objects
- `--max-frames N`: Process only N frames (for testing)
- `--reid-threshold FLOAT`: Threshold for ReID matching (0.0-1.0, default: 0.7)
- `--detection-confidence FLOAT`: Minimum confidence for detection (0.0-1.0, default: 0.5)

### Example

Process two synchronized videos with visualization and save results:

```bash
./run.sh \
    videos/camera1.mp4 \
    videos/camera2.mp4 \
    --output-dir results \
    --visualize \
    --save-thumbnails
```

## Output

The system generates the following outputs in the specified output directory:

- `camera1_tracked.mp4`, `camera2_tracked.mp4`: Processed videos with tracking information
- `thumbnails/`: Directory containing extracted thumbnails of detected persons
- `summary.json`: JSON file with tracking statistics and metrics

## Project Structure

```
multi-camera-tracking/
├── multi_camera_tracking/    # Main package
│   ├── detector/            # Object detection modules
│   ├── reid/                # Re-identification models
│   ├── tracker/             # Tracking algorithms
│   ├── utils/               # Utility functions
│   ├── __init__.py
│   └── main.py              # Main application
├── outputs/                 # Default output directory
├── run.sh                   # Setup and execution script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Performance Tips

- For best performance on Apple Silicon, ensure you're using Python 3.8 or higher
- Reduce the input video resolution if processing is too slow
- Use `--max-frames` for quick testing
- On Apple Silicon, the system automatically uses MPS acceleration
- For NVIDIA GPUs, ensure you have the appropriate CUDA version installed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [StrongSORT](https://github.com/dyhBUPT/StrongSORT) for multi-object tracking
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid) for person re-identification
- [PyTorch](https://pytorch.org/) for deep learning framework
