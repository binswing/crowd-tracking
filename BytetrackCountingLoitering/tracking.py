# tracking.py
import sys
import os
import numpy as np
from argparse import Namespace

# Add ByteTrack path
sys.path.insert(0, r"D:\Anomaly-Detection-in-Surveillance-Videos\ByteTrack")
from yolox.tracker.byte_tracker import BYTETracker

# Initialize tracker
args = Namespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
tracker = BYTETracker(args)

def convert_to_bytetrack(yolo_output):
    detections = []
    for i in range(len(yolo_output.boxes_xyxy)):
        x1, y1, x2, y2 = yolo_output.boxes_xyxy[i]
        conf = yolo_output.confidences[i]
        detections.append([x1, y1, x2, y2, conf])
    return np.array(detections)