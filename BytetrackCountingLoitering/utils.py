# utils.py
import cv2
import numpy as np
from config import polygon

def inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def count_in_polygon(tracks, polygon):
    count = 0
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id = track  # Unpack class_id
        if class_id == 0:  # Only count persons (class 0)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if inside_polygon((cx, cy), polygon):
                count += 1
    return count