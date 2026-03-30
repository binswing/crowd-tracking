# main.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from config import polygon, CONF_THRESHOLD
from detection import model, detect
from tracking import tracker
from utils import count_in_polygon
from loitering import check_loitering, loiter_dict

def process_frame(frame, frame_id, loiter_dict):
    # Run YOLO tracking
    yolo_output = model.track(frame, persist=True, classes=[0, 1, 3], conf=CONF_THRESHOLD, verbose=False)

    # Get tracks from ultralytics, including class_ids
    tracks = []
    if yolo_output[0].boxes.id is not None:
        boxes = yolo_output[0].boxes.xyxy.cpu().numpy()
        ids = yolo_output[0].boxes.id.cpu().numpy().astype(int)
        classes = yolo_output[0].boxes.cls.cpu().numpy().astype(int)  # Add class_ids
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            track_id = ids[i]
            class_id = classes[i]
            tracks.append([x1, y1, x2, y2, track_id, class_id])  # Add class_id to track

    # Draw polygon for debug
    cv2.polylines(frame, [polygon], True, (255,0,0), 2)

    # Count only persons (class 0) in polygon
    count = count_in_polygon(tracks, polygon)

    # Loitering
    current_time = time.time()
    alerts = check_loitering(tracks, polygon, loiter_dict, current_time)

    # Draw tracks with color based on loitering
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id = track
        color = (0,0,255) if track_id in alerts else (0,255,0)
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        cv2.putText(frame, f"ID {track_id} C{class_id}", (int(x1),int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display count
    cv2.putText(frame, f"Count: {count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return frame, count, alerts, len(tracks)

# Example usage with image
if __name__ == "__main__":
    # Load image from Shanghai dataset
    image_path = r"Shanghaidataset\shanghaitech\shanghaitech\testing\frames\01_0014\186.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Image not found!")
    else:
        print("Image loaded successfully!")
        processed_frame, count, alerts, total_tracks = process_frame(frame, 0, loiter_dict)
        plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        plt.title("Processed Image with Detections, Tracks, and Count")
        plt.show()

        print(f"Total detections: {total_tracks}")
        print(f"Active tracks: {total_tracks}")
        print(f"Count in polygon (persons only): {count}")
        print(f"Loitering alerts: {alerts}")