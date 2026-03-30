# Crowd Monitoring Pipeline

## Overview
This system implements real-time crowd monitoring using AI for object detection, multi-object tracking, and counting in a surveillance area. It uses YOLOv8 for detection, ByteTrack for tracking, and custom logic for loitering detection. Optimized for edge devices.

## Features
- Detection of persons, bicycles, and cars.
- Multi-object tracking with stable IDs.
- Person counting in polygon area.
- Loitering detection with velocity check.
- Real-time visualization.

## Installation
1. Install dependencies:
   ```bash
   pip install ultralytics opencv-python numpy matplotlib
   ```

2. Ensure ByteTrack is available in the ByteTrack/ directory.

## Usage
Run the main script:
# run in terminal: git clone https://github.com/ifzhang/ByteTrack.git
# cd Bytetrack ; pip install -r requirements.txt 
```bash
python main.py
```

This will process a sample image from the Shanghai dataset and display results.

## Modules
- `config.py`: Configuration constants and polygon definition.
- `detection.py`: YOLO model loading and detection functions.
- `tracking.py`: ByteTrack integration.
- `utils.py`: Polygon utility functions.
- `loitering.py`: Loitering detection logic.
- `main.py`: Main processing pipeline.

## Notes
- YOLOv8s is selected for edge device performance.
- Counting only applies to persons (class 0).
- Loitering detection combines time (>5s) and velocity (<0.5 pixels/frame).