#!/bin/bash

# Multi-Camera Person Tracking Setup Script
# ---------------------------------------
# This script sets up the Python environment and runs the multi-camera tracking system.
# It automatically detects the hardware configuration and installs the appropriate
# version of PyTorch for optimal performance.

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
REQUIREMENTS="$PROJECT_ROOT/requirements.txt"
PYTHON_CMD=python3

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is required but not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Function to detect hardware and set PyTorch installation command
detect_hardware() {
    echo -e "${GREEN}Detecting hardware configuration...${NC}"
    
    # Check for Apple Silicon
    if [[ $(uname -m) == "arm64" && $(uname) == "Darwin" ]]; then
        echo -e "${GREEN}Detected Apple Silicon (M1/M2)${NC}"
        TORCH_EXTRA="--pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu"
        DEVICE="mps"
    
    # Check for CUDA GPU
    elif command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}Detected NVIDIA GPU${NC}"
        TORCH_EXTRA="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        DEVICE="cuda"
    
    # Default to CPU
    else
        echo -e "${YELLOW}No GPU detected. Using CPU.${NC}"
        TORCH_EXTRA="torch torchvision torchaudio"
        DEVICE="cpu"
    fi
}

# Function to create and activate virtual environment
setup_venv() {
    echo -e "${GREEN}Setting up Python virtual environment...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON_CMD -m venv "$VENV_DIR"
        echo -e "${GREEN}Virtual environment created at $VENV_DIR${NC}"
    else
        echo -e "${YELLOW}Using existing virtual environment at $VENV_DIR${NC}"
    fi
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "darwin"* ]]; then
        source "$VENV_DIR/bin/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi
    
    # Upgrade pip
    pip install --upgrade pip
}

# Function to install dependencies
install_dependencies() {
    echo -e "${GREEN}Installing dependencies...${NC}"
    
    # Install PyTorch with appropriate backend
    echo -e "${GREEN}Installing PyTorch for $DEVICE...${NC}"
    pip install $TORCH_EXTRA
    
    # Install other requirements
    if [ -f "$REQUIREMENTS" ]; then
        pip install -r "$REQUIREMENTS"
    else
        echo -e "${YELLOW}requirements.txt not found. Installing common dependencies...${NC}"
        pip install \
            ultralytics \
            opencv-python \
            numpy \
            scipy \
            tqdm \
            scikit-learn \
            filterpy \
            lap \
            easydict \
            pyyaml \
            loguru
    fi
    
    # Install project in development mode
    pip install -e .
}

# Function to run the tracking system
run_tracking() {
    echo -e "${GREEN}Starting multi-camera tracking system...${NC}"
    echo -e "${YELLOW}Using device: $DEVICE${NC}"
    
    # Check if video files are provided
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: No video files provided.${NC}"
        echo -e "Usage: $0 [options] video1.mp4 video2.mp4 ..."
        echo -e "\nOptions:"
        echo -e "  --output-dir DIR      Directory to save output files (default: outputs)"
        echo -e "  --visualize           Show real-time visualization"
        echo -e "  --no-save-video       Don't save output videos"
        echo -e "  --save-thumbnails     Save thumbnails of detected objects"
        echo -e "  --max-frames N        Process only N frames (for testing)"
        exit 1
    fi
    
    # Build command
    CMD="python -m multi_camera_tracking.main --device $DEVICE"
    
    # Add video files (all non-option arguments)
    for arg in "$@"; do
        if [[ "$arg" != --* ]]; then
            CMD="$CMD $arg"
        fi
    done
    
    # Add options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output-dir)
                CMD="$CMD --output-dir $2"
                shift 2
                ;;
            --visualize)
                CMD="$CMD --visualize"
                shift
                ;;
            --no-save-video)
                CMD="$CMD --no-save-video"
                shift
                ;;
            --save-thumbnails)
                CMD="$CMD --save-thumbnails"
                shift
                ;;
            --max-frames)
                CMD="$CMD --max-frames $2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    echo -e "${GREEN}Running: $CMD${NC}"
    eval $CMD
}

# Main execution
main() {
    # Detect hardware and set PyTorch installation command
    detect_hardware
    
    # Setup virtual environment
    setup_venv
    
    # Install dependencies
    install_dependencies
    
    # Run tracking with provided arguments
    run_tracking "$@"
}

# Run main function with all arguments
main "$@"
